"""
Generates structured JSON evaluation reports from evaluation results.
Reports include AUROC, AP, precision, recall, per-category breakdown,
and latency percentiles.

Usage:
    PYTHONPATH=. python evaluation/eval_report.py
    PYTHONPATH=. python evaluation/eval_report.py --input eval_results.json --output eval_report.json
"""
import json
import logging
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_report(
    results_path: str = "eval_results.json",
    output_path: str = "eval_report.json",
) -> dict:
    """
    Generate a structured evaluation report from pipeline results.

    Returns:
        A dict containing all computed metrics and metadata.
    """
    try:
        with open(results_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(f"Results file not found: {results_path}")
        return {}

    if not results:
        logger.error("No results to report on.")
        return {}

    labeled = [r for r in results if r.get("correctness") is not None]
    scores = [r["hallucination_score"] for r in labeled]
    labels = [0 if r["correctness"] else 1 for r in labeled]

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "results_file": results_path,
            "total_samples": len(results),
            "labeled_samples": len(labeled),
        },
        "overall_metrics": {},
        "latency": {},
        "per_category": {},
        "risk_distribution": {},
    }

    if len(set(labels)) >= 2:
        auroc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        predicted = [1 if s >= 0.5 else 0 for s in scores]
        prec = precision_score(labels, predicted, zero_division=0)
        rec = recall_score(labels, predicted, zero_division=0)

        report["overall_metrics"] = {
            "auroc": round(auroc, 4),
            "average_precision": round(ap, 4),
            "precision_at_0.5": round(prec, 4),
            "recall_at_0.5": round(rec, 4),
            "hallucination_count": sum(labels),
            "correct_count": len(labels) - sum(labels),
        }
    else:
        report["overall_metrics"] = {
            "auroc": None,
            "note": "Insufficient class diversity for AUROC computation",
        }

    latencies = [r["elapsed_seconds"] for r in results if "elapsed_seconds" in r]
    if latencies:
        report["latency"] = {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2),
        }

    risk_labels = [r.get("risk_label", "unknown") for r in results]
    report["risk_distribution"] = {
        "low": risk_labels.count("low"),
        "medium": risk_labels.count("medium"),
        "high": risk_labels.count("high"),
    }

    categories = set(r.get("category", "unknown") for r in labeled)
    for cat in sorted(categories):
        cat_results = [r for r in labeled if r.get("category") == cat]
        cat_scores = [r["hallucination_score"] for r in cat_results]
        cat_labels = [0 if r["correctness"] else 1 for r in cat_results]

        cat_entry = {
            "count": len(cat_results),
            "avg_score": round(float(np.mean(cat_scores)), 4),
            "hallucination_count": sum(cat_labels),
        }

        if len(set(cat_labels)) >= 2:
            cat_entry["auroc"] = round(roc_auc_score(cat_labels, cat_scores), 4)
        else:
            cat_entry["auroc"] = None

        report["per_category"][cat] = cat_entry

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {output_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--input", type=str, default="eval_results.json")
    parser.add_argument("--output", type=str, default="eval_report.json")
    args = parser.parse_args()
    report = generate_report(args.input, args.output)
    if report:
        print(json.dumps(report, indent=2))
