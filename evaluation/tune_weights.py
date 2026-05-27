"""
Weight tuner for the LLM Lie Detector aggregator.
Optimizes signal weights (calibration, uncertainty, cross_check) to maximize AUROC
using grid search over evaluation results.

Usage:
    PYTHONPATH=. python evaluation/tune_weights.py
    PYTHONPATH=. python evaluation/tune_weights.py --eval_results eval_results.json
"""
import json
import numpy as np
import logging
import yaml
import os
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "config.yaml")


def tune_weights(
    eval_results_path: str = "eval_results.json",
    write_back: bool = True,
) -> dict | None:
    """
    Optimizes aggregator weights to maximize AUROC.

    Args:
        eval_results_path: Path to the JSON file with evaluation results.
        write_back: If True, write the best weights back to config.yaml.

    Returns:
        Dict with best weights, or None if tuning fails.
    """
    # 1. Load evaluation results
    try:
        with open(eval_results_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(f"Evaluation results file not found: {eval_results_path}")
        logger.info("Run the evaluation harness first: python evaluation/truthfulqa_eval.py")
        return None

    if not results:
        logger.error("No results found to tune.")
        return None

    # 2. Filter to labeled results only
    labeled = [r for r in results if r.get("correctness") is not None]
    if len(labeled) < 5:
        logger.error(f"Only {len(labeled)} labeled results — need at least 5 for meaningful tuning.")
        return None

    # Build arrays
    labels = np.array([0 if r["correctness"] else 1 for r in labeled])
    cals = np.array([r["calibration"] for r in labeled])
    uncs = np.array([r["uncertainty"] for r in labeled])
    ccs = np.array([r["cross_check"] for r in labeled])

    # Check we have both classes
    if len(set(labels)) < 2:
        logger.error("Only one class present — cannot compute AUROC for tuning.")
        return None

    # 3. Load current weights
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        old_weights = config.get("detection", {}).get("weights", {})
    except Exception:
        old_weights = {"calibration": 0.25, "semantic_uncertainty": 0.30, "cross_check": 0.45}

    old_w1 = old_weights.get("calibration", 0.25)
    old_w2 = old_weights.get("semantic_uncertainty", 0.30)
    old_w3 = old_weights.get("cross_check", 0.45)

    # Compute old AUROC
    old_scores = old_w1 * cals + old_w2 * uncs + old_w3 * ccs
    old_auroc = roc_auc_score(labels, old_scores)

    # 4. Grid search for best weights
    best_auroc = 0.0
    best_weights = None
    step = 0.05

    for w1 in np.arange(0.05, 0.85, step):
        for w2 in np.arange(0.05, 0.85, step):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue

            scores = w1 * cals + w2 * uncs + w3 * ccs
            try:
                auroc = roc_auc_score(labels, scores)
            except ValueError:
                continue

            if auroc > best_auroc:
                best_auroc = auroc
                best_weights = {
                    "calibration": round(float(w1), 3),
                    "semantic_uncertainty": round(float(w2), 3),
                    "cross_check": round(float(w3), 3),
                }

    if best_weights is None:
        logger.error("Grid search failed to find any valid weight combination.")
        return None

    # 5. Print comparison
    print("\n" + "=" * 60)
    print("  WEIGHT TUNING RESULTS")
    print("=" * 60)
    print(f"\n  {'Signal':<25} {'Old Weight':>12} {'New Weight':>12}")
    print("  " + "-" * 50)
    print(f"  {'Calibration':<25} {old_w1:>12.3f} {best_weights['calibration']:>12.3f}")
    print(f"  {'Semantic Uncertainty':<25} {old_w2:>12.3f} {best_weights['semantic_uncertainty']:>12.3f}")
    print(f"  {'Cross-Check':<25} {old_w3:>12.3f} {best_weights['cross_check']:>12.3f}")
    print()
    print(f"  Old AUROC:  {old_auroc:.4f}")
    print(f"  New AUROC:  {best_auroc:.4f}")
    improvement = best_auroc - old_auroc
    print(f"  Improvement: {'+' if improvement >= 0 else ''}{improvement:.4f} "
          f"({'✅ Better' if improvement > 0 else '⚠️ No improvement'})")
    print("=" * 60)

    # 6. Write back to config.yaml
    if write_back and improvement > 0:
        try:
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)

            if "detection" not in config:
                config["detection"] = {}
            config["detection"]["weights"] = best_weights

            with open(CONFIG_PATH, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"✅ Updated config.yaml with new weights.")
        except Exception as e:
            logger.error(f"Failed to write config.yaml: {e}")
    elif write_back and improvement <= 0:
        logger.info("No improvement — keeping existing weights in config.yaml.")

    return best_weights


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tune aggregator weights to maximize AUROC")
    parser.add_argument("--eval_results", type=str, default="eval_results.json",
                        help="Path to evaluation results JSON")
    parser.add_argument("--no-write", action="store_true",
                        help="Don't write back to config.yaml")
    args = parser.parse_args()
    tune_weights(args.eval_results, write_back=not args.no_write)
