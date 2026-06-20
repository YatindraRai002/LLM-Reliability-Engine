"""
Script to fit the PlattCalibrator on evaluation results.
Loads raw scores and labels from eval_results.json, fits the calibrator,
and saves it to calibrator.pkl.
"""
import json
import logging
import argparse
from core.calibration import PlattCalibrator, compute_ece

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_calibrator(results_path: str = "eval_results.json", output_path: str = "calibrator.pkl"):
    """Fit Platt scaling on raw scores."""
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(f"Results file not found: {results_path}")
        return

    labeled = [r for r in results if r.get("correctness") is not None]
    if len(labeled) < 10:
        logger.error("Not enough labeled data to fit calibrator. Need at least 10 samples.")
        return

    raw_scores = [r["hallucination_score"] for r in labeled]
    labels = [0 if r["correctness"] else 1 for r in labeled]

    calibrator = PlattCalibrator()
    calibrator.fit(raw_scores, labels)
    calibrator.save(output_path)
    logger.info(f"Fitted calibrator on {len(raw_scores)} samples and saved to {output_path}")

    old_ece = compute_ece(raw_scores, labels)
    new_scores = [calibrator.transform(s) for s in raw_scores]
    new_ece = compute_ece(new_scores, labels)

    print("\n" + "=" * 50)
    print("  CALIBRATION RESULTS")
    print("=" * 50)
    print(f"  Old ECE: {old_ece:.4f}")
    print(f"  New ECE: {new_ece:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval_results.json", help="Path to evaluation results")
    parser.add_argument("--output", default="calibrator.pkl", help="Path to save calibrator model")
    args = parser.parse_args()
    fit_calibrator(args.input, args.output)
