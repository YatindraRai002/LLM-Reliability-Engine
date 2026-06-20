import argparse
import json
import logging
import os

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATOR_PATH = os.path.join(_ROOT, "evaluation", "platt_calibrator.json")


class PlattCalibrator:
    """
    Sigmoid calibration (Platt scaling) for the hallucination risk score.

    Fits a logistic regression:
        P(hallucination | raw_score) = sigmoid(a * raw_score + b)

    This is the same approach used in:
        Guo et al. 2017 "On Calibration of Modern Neural Networks"
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1.0, max_iter=1000)
        self._fitted = False

    def fit(self, raw_scores: list, labels: list) -> dict:
        """
        Fit the calibrator.

        Args:
            raw_scores: List of floats from aggregate_scores() — uncalibrated
            labels:     List of ints — 1 = hallucination, 0 = correct

        Returns:
            dict with before/after ECE and calibration plot data
        """
        from sklearn.calibration import calibration_curve

        X = np.array(raw_scores).reshape(-1, 1)
        y = np.array(labels)

        if len(set(y)) < 2:
            logger.warning("Only one class in labels — calibrator will be trivial")

        self.lr.fit(X, y)
        self._fitted = True

        calibrated = [self.transform(s) for s in raw_scores]

        ece_before = self._ece(raw_scores, labels)
        ece_after  = self._ece(calibrated, labels)

        frac_pos_before, mean_pred_before = calibration_curve(y, raw_scores,   n_bins=10)
        frac_pos_after,  mean_pred_after  = calibration_curve(y, calibrated,   n_bins=10)

        logger.info(f"Platt calibration: ECE {ece_before:.4f} → {ece_after:.4f}")

        return {
            "ece_before": round(ece_before, 4),
            "ece_after":  round(ece_after,  4),
            "improvement": round(ece_before - ece_after, 4),
            "reliability_before": {
                "mean_predicted": mean_pred_before.tolist(),
                "fraction_pos":   frac_pos_before.tolist(),
            },
            "reliability_after": {
                "mean_predicted": mean_pred_after.tolist(),
                "fraction_pos":   frac_pos_after.tolist(),
            },
        }

    def transform(self, raw_score: float) -> float:
        """Map raw score → calibrated probability."""
        if not self._fitted:
            return float(raw_score)
        return float(self.lr.predict_proba([[raw_score]])[0][1])

    def transform_batch(self, scores: list) -> list:
        if not self._fitted:
            return list(scores)
        X = np.array(scores).reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1].tolist()

    def save(self, path: str = None):
        path = path or CALIBRATOR_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"fitted": self._fitted}
        if self._fitted:
            data["coef"] = self.lr.coef_.tolist()
            data["intercept"] = self.lr.intercept_.tolist()
            data["classes"] = self.lr.classes_.tolist()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibrator saved to {path}")

    def load(self, path: str = None):
        path = path or CALIBRATOR_PATH
        if not os.path.exists(path):
            logger.warning(f"No calibrator found at {path} — using passthrough mode")
            return
        with open(path, "r") as f:
            data = json.load(f)
        self._fitted  = data.get("fitted", False)
        if self._fitted:
            self.lr.coef_ = np.array(data["coef"])
            self.lr.intercept_ = np.array(data["intercept"])
            self.lr.classes_ = np.array(data["classes"])
        logger.info(f"Calibrator loaded from {path}")

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @staticmethod
    def _ece(scores: list, labels: list, n_bins: int = 10) -> float:
        scores = np.array(scores)
        labels = np.array(labels, dtype=float)
        edges  = np.linspace(0.0, 1.0, n_bins + 1)
        ece    = 0.0
        for i in range(n_bins):
            mask = (scores >= edges[i]) & (scores < edges[i + 1])
            if not mask.any():
                continue
            ece += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
        return float(ece)


_calibrator = None


def get_calibrator() -> PlattCalibrator:
    """Return the global calibrator instance (loaded lazily)."""
    global _calibrator
    if _calibrator is None:
        _calibrator = PlattCalibrator()
        _calibrator.load()
    return _calibrator


def calibrate_score(raw_score: float) -> float:
    """Public API — apply Platt calibration to a raw score."""
    return get_calibrator().transform(raw_score)


def cmd_fit(input_file: str, output_path: str = None):
    """Fit calibrator from eval results and save."""
    with open(input_file) as f:
        data = json.load(f)

    results = data.get("results", data)
    labeled = [r for r in results if r.get("correctness") is not None]

    if len(labeled) < 10:
        logger.error(f"Need ≥10 labeled examples, got {len(labeled)}")
        return

    raw_scores = [r["score"] for r in labeled]
    labels     = [0 if r["correctness"] else 1 for r in labeled]

    cal = PlattCalibrator()
    metrics = cal.fit(raw_scores, labels)
    cal.save(output_path)

    print(f"\n{'='*45}")
    print(f"  Platt calibration fit")
    print(f"{'='*45}")
    print(f"  Training examples : {len(labeled)}")
    print(f"  ECE before        : {metrics['ece_before']}")
    print(f"  ECE after         : {metrics['ece_after']}")
    print(f"  ECE improvement   : {metrics['improvement']}")
    print(f"  Saved to          : {output_path or CALIBRATOR_PATH}")
    print(f"{'='*45}\n")


def cmd_test(input_file: str):
    """Test calibration quality on eval data."""
    with open(input_file) as f:
        data = json.load(f)
    results = data.get("results", data)
    labeled = [r for r in results if r.get("correctness") is not None]

    if len(labeled) < 5:
        print("Not enough labeled data to test calibration")
        return

    cal = PlattCalibrator()
    cal.load()
    if not cal.is_fitted:
        print("No calibrator found. Run: python platt_calibration.py fit --input <file>")
        return

    raw    = [r["score"] for r in labeled]
    cal_sc = cal.transform_batch(raw)
    labels = [0 if r["correctness"] else 1 for r in labeled]

    ece_r = PlattCalibrator._ece(raw, labels)
    ece_c = PlattCalibrator._ece(cal_sc, labels)

    print(f"\n  ECE (raw)       : {ece_r:.4f}")
    print(f"  ECE (calibrated): {ece_c:.4f}")
    print(f"  Improvement     : {ece_r - ece_c:.4f}")

    print("\n  Sample score mapping:")
    for raw_s, cal_s in zip(raw[:8], cal_sc[:8]):
        print(f"    {raw_s:.3f}  →  {cal_s:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    fit_p = sub.add_parser("fit",  help="Fit calibrator from eval results")
    fit_p.add_argument("--input",  required=True)
    fit_p.add_argument("--output", default=None)

    test_p = sub.add_parser("test", help="Test calibration quality")
    test_p.add_argument("--input", required=True)

    args = p.parse_args()
    if args.cmd == "fit":
        cmd_fit(args.input, args.output)
    elif args.cmd == "test":
        cmd_test(args.input)
    else:
        p.print_help()
