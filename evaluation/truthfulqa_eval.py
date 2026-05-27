"""
Evaluation harness for the LLM Lie Detector pipeline.
Supports TruthfulQA and HaluEval datasets with AUROC, precision, recall,
per-category breakdown, tqdm progress, and checkpoint resume.

Usage:
    PYTHONPATH=. python evaluation/truthfulqa_eval.py
    PYTHONPATH=. python evaluation/truthfulqa_eval.py --n_samples 100 --dataset both
"""
import json
import time
import logging
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
CHECKPOINT_FILE = "eval_checkpoint.jsonl"
OUTPUT_FILE = "eval_results.json"
ERROR_FILE = "eval_errors.json"

TARGET_CATEGORIES = ["Finance", "Health", "Law", "Conspiracies", "Misconceptions"]


# ── Checkpoint Resume ────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str) -> list[dict]:
    """Load partial results from a JSONL checkpoint file."""
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Resumed {len(results)} results from checkpoint: {checkpoint_path}")
    return results


def append_checkpoint(checkpoint_path: str, result: dict):
    """Append a single result to the JSONL checkpoint file."""
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


# ── NLI-based Correctness Scoring ────────────────────────────────────

def score_correctness_nli(model_response: str, reference_answer: str) -> bool | None:
    """
    Use the NLI model to determine if the model response is correct
    by checking entailment against the reference answer.
    Returns True if entailment > contradiction, False otherwise.
    Returns None if scoring fails.
    """
    if not model_response or not reference_answer:
        return None

    try:
        from core.cross_check import nli_score_sync
        scores = nli_score_sync(model_response, reference_answer)
        # If the model's response entails the reference answer, it's correct
        return scores["entailment"] > scores["contradiction"]
    except Exception as e:
        logger.warning(f"NLI correctness scoring failed: {e}")
        return None


# ── Dataset Loaders ──────────────────────────────────────────────────

def load_truthfulqa_samples(n_samples: int) -> list[dict]:
    """Load and filter TruthfulQA samples by target categories."""
    logger.info("Loading TruthfulQA dataset...")
    try:
        dataset = load_dataset("truthful_qa", "generation")["validation"]
    except Exception as e:
        logger.error(f"Failed to load TruthfulQA: {e}")
        return []

    filtered = [
        {
            "question": ex["question"],
            "best_answer": ex["best_answer"],
            "category": ex["category"],
            "dataset": "truthfulqa",
        }
        for ex in dataset
        if ex["category"] in TARGET_CATEGORIES
    ][:n_samples]

    logger.info(f"Loaded {len(filtered)} TruthfulQA samples across categories: "
                f"{set(s['category'] for s in filtered)}")
    return filtered


def load_halueval_samples(n_samples: int) -> list[dict]:
    """
    Load HaluEval QA samples. Each sample has a question,
    a hallucinated answer, and a correct answer.
    We create two eval entries per sample: one correct, one hallucinated.
    """
    logger.info("Loading HaluEval dataset...")
    try:
        dataset = load_dataset("pminervini/HaluEval", "qa_samples")
        split = dataset.get("data") or dataset.get("train") or list(dataset.values())[0]
    except Exception as e:
        logger.warning(f"Failed to load HaluEval (may require HF token): {e}")
        return []

    samples = []
    for ex in list(split)[:n_samples // 2]:
        question = ex.get("question", "")
        hallucinated = ex.get("hallucinated_answer", "")
        correct = ex.get("right_answer", ex.get("correct_answer", ""))

        if question and hallucinated:
            samples.append({
                "question": question,
                "best_answer": correct,
                "category": "HaluEval",
                "dataset": "halueval",
                "is_hallucination_input": False,
            })
        if question and correct:
            samples.append({
                "question": question,
                "best_answer": correct,
                "category": "HaluEval",
                "dataset": "halueval",
                "is_hallucination_input": False,
            })

    logger.info(f"Loaded {len(samples)} HaluEval samples")
    return samples[:n_samples]


# ── Main Evaluation Runner ───────────────────────────────────────────

def run_evaluation(
    n_samples: int = 50,
    dataset_choice: str = "truthfulqa",
    output_file: str = OUTPUT_FILE,
    checkpoint_file: str = CHECKPOINT_FILE,
    resume: bool = True,
) -> list[dict]:
    """
    Run the full evaluation pipeline with checkpoint resume support.

    Args:
        n_samples: Number of samples to evaluate per dataset.
        dataset_choice: "truthfulqa", "halueval", or "both".
        output_file: Path for the final JSON results.
        checkpoint_file: Path for the JSONL checkpoint file.
        resume: Whether to resume from checkpoint.
    """
    from core.aggregator import run_full_pipeline

    # 1. Load datasets
    samples = []
    if dataset_choice in ("truthfulqa", "both"):
        samples.extend(load_truthfulqa_samples(n_samples))
    if dataset_choice in ("halueval", "both"):
        samples.extend(load_halueval_samples(n_samples))

    if not samples:
        logger.error("No evaluation samples loaded. Exiting.")
        return []

    # 2. Resume from checkpoint
    completed_results = []
    completed_questions = set()
    if resume:
        completed_results = load_checkpoint(checkpoint_file)
        completed_questions = {r["question"] for r in completed_results}

    # Filter out already-completed samples
    remaining_samples = [s for s in samples if s["question"] not in completed_questions]
    logger.info(f"Total samples: {len(samples)}, "
                f"Already completed: {len(completed_results)}, "
                f"Remaining: {len(remaining_samples)}")

    # 3. Run pipeline on remaining samples
    errors = []
    for sample in tqdm(remaining_samples, desc="Evaluating", unit="sample"):
        question = sample["question"]
        category = sample["category"]
        best_answer = sample["best_answer"]

        start = time.time()
        try:
            pipeline_result = run_full_pipeline(question)
            elapsed = time.time() - start

            # Determine correctness via NLI
            model_response = pipeline_result.get("calibration_detail", {}).get("response", "")
            correctness = score_correctness_nli(model_response, best_answer)

            result = {
                "question": question,
                "category": category,
                "dataset": sample.get("dataset", "truthfulqa"),
                "best_answer": best_answer,
                "model_response": model_response,
                "hallucination_score": pipeline_result["result"].score,
                "risk_label": pipeline_result["result"].label,
                "elapsed_seconds": round(elapsed, 2),
                "calibration": pipeline_result["result"].calibration_score,
                "uncertainty": pipeline_result["result"].uncertainty_score,
                "cross_check": pipeline_result["result"].cross_check_score,
                "correctness": correctness,
            }

            completed_results.append(result)
            append_checkpoint(checkpoint_file, result)

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Pipeline failed for: {question[:60]}... — {e}")
            errors.append({
                "question": question,
                "category": category,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
            })

    # 4. Save final results
    with open(output_file, "w") as f:
        json.dump(completed_results, f, indent=2, default=str)
    logger.info(f"Saved {len(completed_results)} results to {output_file}")

    # 5. Save errors
    if errors:
        with open(ERROR_FILE, "w") as f:
            json.dump(errors, f, indent=2)
        logger.warning(f"Saved {len(errors)} errors to {ERROR_FILE}")

    # 6. Compute and print metrics
    print_metrics(completed_results)

    return completed_results


# ── Metrics Computation ──────────────────────────────────────────────

def print_metrics(results: list[dict]):
    """Compute and print AUROC, AP, precision, recall, per-category breakdown."""
    if not results:
        logger.warning("No results to compute metrics from.")
        return

    # Filter results with valid correctness labels
    labeled = [r for r in results if r.get("correctness") is not None]

    if len(labeled) < 2:
        logger.warning(f"Only {len(labeled)} labeled samples — need at least 2 for AUROC.")
        return

    # scores = hallucination likelihood, labels = 1 if hallucination (not correct)
    scores = [r["hallucination_score"] for r in labeled]
    labels = [0 if r["correctness"] else 1 for r in labeled]

    # Check if we have both classes
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class present in labels ({unique_labels}). Cannot compute AUROC.")
        _print_basic_stats(results)
        return

    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    # Threshold-based precision/recall at 0.5
    predicted = [1 if s >= 0.5 else 0 for s in scores]
    prec = precision_score(labels, predicted, zero_division=0)
    rec = recall_score(labels, predicted, zero_division=0)

    # Latency stats
    latencies = [r["elapsed_seconds"] for r in results]
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Samples Processed:  {len(results)}")
    print(f"  Labeled Samples:    {len(labeled)}")
    print(f"  Hallucinations:     {sum(labels)} / {len(labels)}")
    print()
    print(f"  AUROC:              {auroc:.4f}  {'✅' if auroc > 0.80 else '⚠️'}")
    print(f"  Average Precision:  {ap:.4f}")
    print(f"  Precision (@0.5):   {prec:.4f}  {'✅' if prec > 0.75 else '⚠️'}")
    print(f"  Recall (@0.5):      {rec:.4f}  {'✅' if rec > 0.70 else '⚠️'}")
    print()
    print(f"  Latency P50:        {p50:.2f}s")
    print(f"  Latency P95:        {p95:.2f}s  {'✅' if p95 < 30 else '⚠️'}")
    print(f"  Latency P99:        {p99:.2f}s")
    print("=" * 60)

    # Per-category breakdown
    categories = set(r["category"] for r in labeled)
    if len(categories) > 1:
        print("\n  PER-CATEGORY BREAKDOWN:")
        print(f"  {'Category':<20} {'Count':>6} {'AUROC':>8} {'Avg Score':>10}")
        print("  " + "-" * 48)

        for cat in sorted(categories):
            cat_results = [r for r in labeled if r["category"] == cat]
            cat_scores = [r["hallucination_score"] for r in cat_results]
            cat_labels = [0 if r["correctness"] else 1 for r in cat_results]
            cat_avg = np.mean(cat_scores)

            if len(set(cat_labels)) >= 2:
                cat_auroc = roc_auc_score(cat_labels, cat_scores)
                print(f"  {cat:<20} {len(cat_results):>6} {cat_auroc:>8.4f} {cat_avg:>10.4f}")
            else:
                print(f"  {cat:<20} {len(cat_results):>6} {'N/A':>8} {cat_avg:>10.4f}")

        print("=" * 60)


def _print_basic_stats(results: list[dict]):
    """Fallback stats when AUROC can't be computed."""
    high_risk = sum(1 for r in results if r["risk_label"] == "high")
    avg_latency = np.mean([r["elapsed_seconds"] for r in results])

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY (Basic — insufficient labels for AUROC)")
    print("=" * 60)
    print(f"  Samples Processed:  {len(results)}")
    print(f"  High Risk Flagged:  {high_risk}/{len(results)} "
          f"({100 * high_risk / len(results):.1f}%)")
    print(f"  Avg Latency:        {avg_latency:.2f}s per query")
    print("=" * 60)


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Lie Detector Evaluation Harness")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples per dataset (default: 50)")
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                        choices=["truthfulqa", "halueval", "both"],
                        help="Which dataset(s) to evaluate on (default: truthfulqa)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output file for results JSON")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh instead of resuming from checkpoint")
    args = parser.parse_args()

    if args.no_resume and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Cleared previous checkpoint.")

    run_evaluation(
        n_samples=args.n_samples,
        dataset_choice=args.dataset,
        output_file=args.output,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
