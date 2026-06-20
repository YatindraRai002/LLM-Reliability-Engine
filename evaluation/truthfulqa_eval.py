import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def load_truthfulqa(n: int, categories: list = None) -> list:
    from datasets import load_dataset
    logger.info("Loading TruthfulQA dataset...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    items = list(ds)
    if categories:
        items = [x for x in items if x.get("category") in categories]
    items = items[:n] if n else items
    logger.info(f"Loaded {len(items)} TruthfulQA examples")
    return [
        {
            "id": i,
            "question": x["question"],
            "correct_answers": x.get("correct_answers", []),
            "incorrect_answers": x.get("incorrect_answers", []),
            "best_answer": x.get("best_answer", ""),
            "category": x.get("category", "unknown"),
            "dataset": "truthfulqa",
        }
        for i, x in enumerate(items)
    ]


def load_halueval(n: int) -> list:
    from datasets import load_dataset
    logger.info("Loading HaluEval dataset...")
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        items = list(ds)[:n]
        logger.info(f"Loaded {len(items)} HaluEval examples")
        return [
            {
                "id": i,
                "question": x.get("question", x.get("input", "")),
                "correct_answers": [x.get("right_answer", x.get("answer", ""))],
                "incorrect_answers": [x.get("hallucinated_answer", "")],
                "best_answer": x.get("right_answer", x.get("answer", "")),
                "category": "halueval",
                "dataset": "halueval",
                "is_hallucinated": True,
            }
            for i, x in enumerate(items)
        ]
    except Exception as e:
        logger.warning(f"HaluEval load failed ({e}) — falling back to TruthfulQA")
        return load_truthfulqa(n)


def approx_correctness(response: str, correct: list, incorrect: list) -> bool | None:
    """
    Heuristic correctness check using keyword matching.
    Returns True (correct), False (incorrect), or None (ambiguous).
    A proper evaluation uses an LLM judge — this is a fast approximation.
    """
    r = response.lower().strip()
    if not r:
        return None

    for wrong in incorrect:
        if len(wrong) < 10:
            continue
        key_words = [w for w in wrong.lower().split()[:4] if len(w) > 3]
        if len(key_words) > 1 and sum(1 for w in key_words if w in r) >= 2:
            return False

    for right in correct:
        if len(right) < 5:
            continue
        key_words = [w for w in right.lower().split()[:4] if len(w) > 3]
        if len(key_words) > 1 and sum(1 for w in key_words if w in r) >= 2:
            return True

    return None


def compute_metrics(results: list) -> dict:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
    )

    scores = [r["score"] for r in results]
    labeled = [r for r in results if r["correctness"] is not None]

    if len(labeled) < 5:
        logger.warning("Not enough labeled examples for reliable metrics")
        return {"auroc": None, "avg_precision": None, "n_labeled": len(labeled)}

    y_true = [0 if r["correctness"] else 1 for r in labeled]
    y_score = [r["score"] for r in labeled]

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = None

    try:
        ap = float(average_precision_score(y_true, y_score))
    except Exception:
        ap = None

    optimal_threshold = 0.5
    best_f1 = 0.0
    try:
        prec, rec, thrs = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(np.argmax(f1_scores))
        best_f1 = float(f1_scores[best_idx])
        optimal_threshold = float(thrs[min(best_idx, len(thrs) - 1)])
    except Exception:
        pass

    preds = [1 if s >= optimal_threshold else 0 for s in y_score]
    tp = sum(1 for p, t in zip(preds, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, y_true) if p == 0 and t == 1)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    category_stats = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "scores": [], "high_risk": 0}
        category_stats[cat]["count"] += 1
        category_stats[cat]["scores"].append(r["score"])
        if r["label"] == "high":
            category_stats[cat]["high_risk"] += 1
    for cat in category_stats:
        s = category_stats[cat]["scores"]
        category_stats[cat]["mean_score"] = round(float(np.mean(s)), 3)
        del category_stats[cat]["scores"]

    latencies = [r["elapsed"] for r in results]
    lat_p50 = float(np.percentile(latencies, 50))
    lat_p95 = float(np.percentile(latencies, 95))

    return {
        "auroc": round(auroc, 4) if auroc else None,
        "avg_precision": round(ap, 4) if ap else None,
        "precision_at_threshold": round(precision, 4),
        "recall_at_threshold": round(recall, 4),
        "best_f1": round(best_f1, 4),
        "optimal_threshold": round(optimal_threshold, 3),
        "n_total": len(results),
        "n_labeled": len(labeled),
        "score_mean": round(float(np.mean(scores)), 3),
        "score_std": round(float(np.std(scores)), 3),
        "risk_distribution": {
            lbl: sum(1 for r in results if r["label"] == lbl)
            for lbl in ("low", "medium", "high")
        },
        "latency_p50_s": round(lat_p50, 2),
        "latency_p95_s": round(lat_p95, 2),
        "category_breakdown": category_stats,
    }


def run(
    n: int = 100,
    dataset: str = "truthfulqa",
    out_file: str = None,
    resume_file: str = None,
    local_sampling: bool = False,
) -> dict:
    from core.aggregator import run_full_pipeline

    if dataset == "halueval":
        examples = load_halueval(n)
    else:
        cats = CONFIG.get("evaluation", {}).get("target_categories", None)
        examples = load_truthfulqa(n, cats)

    done_ids = set()
    results = []
    errors = []
    if resume_file and os.path.exists(resume_file):
        with open(resume_file) as f:
            partial = json.load(f)
        results = partial.get("results", [])
        errors = partial.get("errors", [])
        done_ids = {r["id"] for r in results}
        logger.info(f"Resuming: {len(done_ids)} already done")

    out_file = out_file or f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    t_total = time.time()

    for ex in tqdm(examples, desc=f"Evaluating {dataset}"):
        if ex["id"] in done_ids:
            continue
        try:
            t = time.time()
            out = run_full_pipeline(
                ex["question"],
                use_local_for_uncertainty=local_sampling,
            )
            elapsed = round(time.time() - t, 2)

            resp = out["calibration_detail"].get("response", "")
            correctness = approx_correctness(
                resp,
                ex.get("correct_answers", []),
                ex.get("incorrect_answers", []),
            )

            results.append({
                "id":           ex["id"],
                "question":     ex["question"],
                "category":     ex.get("category", "unknown"),
                "dataset":      ex.get("dataset", dataset),
                "best_answer":  ex.get("best_answer", ""),
                "response":     resp,
                "correctness":  correctness,
                
                "score":        out["result"].score,
                "label":        out["result"].label,
                "elapsed":      elapsed,
                
                "hallucination_score": out["result"].score,
                "risk_label":          out["result"].label,
                "elapsed_seconds":      elapsed,
                
                "calibration":  out["result"].calibration_score,
                "uncertainty":  out["result"].uncertainty_score,
                "cross_check":  out["result"].cross_check_score,
                "n_clusters":   out["uncertainty_detail"].get("n_semantic_clusters", 1),
                "verdict":      out["cross_check_detail"].get("verdict", ""),
                "groq_ok":      out["cross_check_detail"].get("groq_available", False),
            })

        except Exception as e:
            logger.error(f"Error on [{ex['id']}]: {e}")
            errors.append({"id": ex["id"], "question": ex["question"], "error": str(e)})

        if (len(results) + len(errors)) % 10 == 0:
            _save(out_file, results, errors, {}, dataset, n)

    metrics = compute_metrics(results)
    total_time = round(time.time() - t_total, 1)

    data = _save(out_file, results, errors, metrics, dataset, n, total_time)

    print(f"\n{'='*55}")
    print(f"  Evaluation complete — {dataset.upper()}")
    print(f"{'='*55}")
    print(f"  Examples evaluated : {metrics['n_total']}")
    print(f"  Labeled examples   : {metrics['n_labeled']}")
    print(f"  AUROC              : {metrics['auroc']}")
    print(f"  Avg precision      : {metrics['avg_precision']}")
    print(f"  Precision @ thresh : {metrics['precision_at_threshold']}")
    print(f"  Recall @ thresh    : {metrics['recall_at_threshold']}")
    print(f"  Best F1            : {metrics['best_f1']}")
    print(f"  Score mean ± std   : {metrics['score_mean']} ± {metrics['score_std']}")
    print(f"  Latency p50 / p95  : {metrics['latency_p50_s']}s / {metrics['latency_p95_s']}s")
    print(f"  Risk distribution  : {metrics['risk_distribution']}")
    print(f"  Total time         : {total_time}s")
    print(f"  Results saved to   : {out_file}")
    print(f"{'='*55}\n")

    return data


def _save(path, results, errors, metrics, dataset, n, total_time=None):
    data = {
        "metadata": {
            "timestamp":    datetime.now().isoformat(),
            "dataset":      dataset,
            "n_requested":  n,
            "total_time_s": total_time,
            "config": {
                "weights":    CONFIG.get("detection", {}).get("weights", {}),
                "thresholds": CONFIG.get("detection", {}).get("thresholds", {}),
                "n_samples":  CONFIG.get("sampling", {}).get("n_samples", 6),
            },
        },
        "metrics":  metrics,
        "results":  results,
        "errors":   errors,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_checkpoint(checkpoint_path: str) -> list[dict]:
    """Load partial results from a JSONL checkpoint file (compatibility helper)."""
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
    return results


def append_checkpoint(checkpoint_path: str, result: dict):
    """Append a single result to the JSONL checkpoint file (compatibility helper)."""
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def print_metrics(results: list[dict]):
    """Print metrics (compatibility helper)."""
    print("--- EVALUATION SUMMARY ---")
    mapped = []
    for r in results:
        mapped.append({
            "score": r.get("score", r.get("hallucination_score", 0.0)),
            "correctness": r.get("correctness"),
            "label": r.get("label", r.get("risk_label", "low")),
            "category": r.get("category", "unknown"),
            "elapsed": r.get("elapsed", r.get("elapsed_seconds", 0.0)),
        })
    metrics = compute_metrics(mapped)
    print(f"AUROC: {metrics.get('auroc')}")


def _print_basic_stats(results: list[dict]):
    """Fallback stats (compatibility helper)."""
    high_risk = sum(1 for r in results if r.get("label", r.get("risk_label")) == "high")
    latencies = [r.get("elapsed", r.get("elapsed_seconds", 0.0)) for r in results]
    avg_latency = np.mean(latencies) if latencies else 0.0
    print(f"High Risk Flagged: {high_risk}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM Lie Detector evaluation harness")
    p.add_argument("--n",             type=int,   default=100,         help="Number of examples")
    p.add_argument("--dataset",       type=str,   default="truthfulqa", choices=["truthfulqa","halueval"])
    p.add_argument("--output",        type=str,   default=None,         help="Output JSON path")
    p.add_argument("--resume",        type=str,   default=None,         help="Resume from partial JSON")
    p.add_argument("--local-sampling",action="store_true",              help="Use local model for sampling")
    args = p.parse_args()

    run(
        n               = args.n,
        dataset         = args.dataset,
        out_file        = args.output,
        resume_file     = args.resume,
        local_sampling  = args.local_sampling,
    )

