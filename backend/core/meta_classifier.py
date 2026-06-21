"""
core/meta_classifier.py
Learned meta-classifier (Logistic Regression) for fusing uncertainty signals.
Fits separate models for 3-signal (full) and 2-signal (degraded) modes.
"""

import json
import logging
import os
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_ROOT, "config.yaml")
MODEL_PATH = os.path.join(_ROOT, "core", "meta_classifier.json")

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


class MetaClassifier:
    """
    Logistic Regression classifier to predict hallucination probability
    from individual uncertainty signals.
    """

    def __init__(self):
        self.coef_full = None
        self.intercept_full = None
        self.coef_2signal = None
        self.intercept_2signal = None
        self._fitted = False

    def fit(self, cal_scores: list, unc_scores: list, cc_scores: list, labels: list) -> dict:
        """
        Fit both full (3-signal) and 2-signal logistic regression models.
        Labels: 0 = correct, 1 = hallucination
        """
        from sklearn.linear_model import LogisticRegression

        X_full = np.column_stack([cal_scores, unc_scores, cc_scores])
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            logger.warning("Only one class present in labels. Cannot fit a robust classifier.")
            return {"success": False, "reason": "single_class"}

        # Fit 3-signal model
        lr_full = LogisticRegression(C=1.0)
        lr_full.fit(X_full, y)
        self.coef_full = lr_full.coef_[0].tolist()
        self.intercept_full = float(lr_full.intercept_[0])

        # Fit 2-signal model (Calibration + Semantic Uncertainty)
        X_2signal = np.column_stack([cal_scores, unc_scores])
        lr_2s = LogisticRegression(C=1.0)
        lr_2s.fit(X_2signal, y)
        self.coef_2signal = lr_2s.coef_[0].tolist()
        self.intercept_2signal = float(lr_2s.intercept_[0])

        self._fitted = True
        logger.info("Successfully fitted full and 2-signal MetaClassifiers.")

        # Compute training accuracies
        pred_full = lr_full.predict(X_full)
        pred_2s = lr_2s.predict(X_2signal)

        return {
            "success": True,
            "train_accuracy_full": float(np.mean(pred_full == y)),
            "train_accuracy_2signal": float(np.mean(pred_2s == y)),
            "coef_full": self.coef_full,
            "intercept_full": self.intercept_full,
            "coef_2signal": self.coef_2signal,
            "intercept_2signal": self.intercept_2signal,
        }

    def predict_proba(self, cal: float, unc: float, cc: float = 0.5, mode: str = "full") -> float:
        """
        Predict probability of hallucination.
        Falls back to configured weighted sum if the meta-classifier is not fitted.
        """
        if not self._fitted:
            # Fallback to current config-defined weights
            w = CONFIG.get("detection", {}).get("weights", {"calibration": 0.20, "semantic_uncertainty": 0.50, "cross_check": 0.30})
            w1 = w.get("calibration", 0.20)
            w2 = w.get("semantic_uncertainty", 0.50)
            w3 = w.get("cross_check", 0.30)

            if mode == "2-signal":
                # Redistribute cross-check weight (40/60 split like aggregator.py)
                w1 = w1 + w3 * 0.40
                w2 = w2 + w3 * 0.60
                total = w1 + w2
                w1, w2 = w1 / total, w2 / total
                score = w1 * cal + w2 * unc
            else:
                total = w1 + w2 + w3
                w1, w2, w3 = w1 / total, w2 / total, w3 / total
                score = w1 * cal + w2 * unc + w3 * cc
            return float(np.clip(score, 0.0, 1.0))

        if mode == "full":
            z = self.intercept_full + self.coef_full[0] * cal + self.coef_full[1] * unc + self.coef_full[2] * cc
        else:
            z = self.intercept_2signal + self.coef_2signal[0] * cal + self.coef_2signal[1] * unc

        prob = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(prob, 0.0, 1.0))

    def save(self, path: str = None):
        path = path or MODEL_PATH
        data = {
            "fitted": self._fitted,
            "coef_full": self.coef_full,
            "intercept_full": self.intercept_full,
            "coef_2signal": self.coef_2signal,
            "intercept_2signal": self.intercept_2signal,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"MetaClassifier saved to {path}")

    def load(self, path: str = None):
        path = path or MODEL_PATH
        if not os.path.exists(path):
            logger.warning(f"No MetaClassifier model found at {path} — running in fallback mode.")
            self._fitted = False
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._fitted = data.get("fitted", False)
            self.coef_full = data.get("coef_full")
            self.intercept_full = data.get("intercept_full")
            self.coef_2signal = data.get("coef_2signal")
            self.intercept_2signal = data.get("intercept_2signal")
            logger.info(f"MetaClassifier loaded from {path} (fitted: {self._fitted})")
        except Exception as e:
            logger.error(f"Failed to load MetaClassifier from {path}: {e}")
            self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted


_meta_classifier = None

def get_meta_classifier() -> MetaClassifier:
    """Lazy-loaded global instance of the MetaClassifier."""
    global _meta_classifier
    if _meta_classifier is None:
        _meta_classifier = MetaClassifier()
        _meta_classifier.load()
    return _meta_classifier
