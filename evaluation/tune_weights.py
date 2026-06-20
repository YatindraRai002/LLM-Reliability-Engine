import argparse
import json
import logging
import os

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_ROOT, "config.yaml")


def load_results(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    labeled = [r for r in results if r.get("correctness") is not None]
    logger.info(f"Loaded {len(results)} results, {len(labeled)} labeled")
    return results


def score_weights(results: list, w1: float, w2: float, w3: float) -> dict:
    """Compute AUROC and other metrics for a given weight triple."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    labeled = [r for r in results if r.get("correctness") is not None]
    if len(labeled) < 5:
        return {"auroc": 0.5, "ap": 0.5}

    total = w1 + w2 + w3
    if total == 0:
        return {"auroc": 0.5, "ap": 0.5}
    w1n, w2n, w3n = w1 / total, w2 / total, w3 / total

    scores = [
        np.clip(w1n * r["calibration"] + w2n * r["uncertainty"] + w3n * r["cross_check"], 0, 1)
        for r in labeled
    ]
    # Label: 1 = hallucination (wrong answer)
    y_true = [0 if r["correctness"] else 1 for r in labeled]

    try:
        auroc = float(roc_auc_score(y_true, scores))
        ap    = float(average_precision_score(y_true, scores))
    except Exception:
        auroc, ap = 0.5, 0.5

    return {"auroc": auroc, "ap": ap}


def grid_search(results: list) -> dict:
    """Exhaustive grid search over weight combinations."""
    logger.info("Running grid search over weight combinations...")
    step = 0.05
    options = np.arange(0.05, 0.91, step)

    best = {"auroc": 0.0, "w1": 0.20, "w2": 0.50, "w3": 0.30}
    n_tried = 0

    for w1 in options:
        for w2 in options:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0.05 or w3 > 0.90:
                continue
            metrics = score_weights(results, w1, w2, w3)
            n_tried += 1
            if metrics["auroc"] > best["auroc"]:
                best = {
                    "auroc": metrics["auroc"],
                    "ap":    metrics["ap"],
                    "w1":    round(w1, 2),
                    "w2":    round(w2, 2),
                    "w3":    round(w3, 2),
                }

    logger.info(f"Grid search complete — tried {n_tried} combinations")
    return best


def bayesian_search(results: list, n_calls: int = 60) -> dict:
    """Bayesian optimization using scipy — faster than full grid search."""
    from scipy.optimize import differential_evolution

    logger.info("Running Bayesian/differential evolution search...")

    def objective(x):
        w1, w2 = x
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            return 1.0  # penalize invalid combinations
        return -score_weights(results, w1, w2, w3)["auroc"]

    # Constraints: w1 + w2 <= 0.95 (so w3 >= 0.05)
    bounds = [(0.05, 0.90), (0.05, 0.90)]
    result = differential_evolution(
        objective,
        bounds,
        maxiter=n_calls,
        seed=42,
        tol=0.001,
        workers=1,
    )

    w1, w2 = round(float(result.x[0]), 2), round(float(result.x[1]), 2)
    w3 = round(1.0 - w1 - w2, 2)
    metrics = score_weights(results, w1, w2, w3)

    return {
        "auroc": metrics["auroc"],
        "ap":    metrics["ap"],
        "w1":    w1,
        "w2":    w2,
        "w3":    max(w3, 0.05),
    }


def write_weights_to_config(best: dict):
    """Update config.yaml with the best found weights."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Support both "aggregator" and "detection" key names
    weight_key = "aggregator" if "aggregator" in config else "detection"
    if weight_key not in config:
        config[weight_key] = {}
    if "weights" not in config[weight_key]:
        config[weight_key]["weights"] = {}

    config[weight_key]["weights"]["calibration"] = best["w1"]
    config[weight_key]["weights"]["semantic_uncertainty"]  = best["w2"]
    config[weight_key]["weights"]["cross_check"]  = best["w3"]

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Written to config.yaml: cal={best['w1']} unc={best['w2']} cc={best['w3']}")


def run(input_file: str, method: str = "grid", write_back: bool = True):
    try:
        results = load_results(input_file)
    except Exception as e:
        logger.error(f"Failed to load results from {input_file}: {e}")
        return None

    labeled = [r for r in results if r.get("correctness") is not None]

    if len(labeled) < 5:
        logger.error(
            f"Only {len(labeled)} labeled examples — need at least 5 for reliable tuning. "
            "Run with a larger --n in truthfulqa_eval.py first."
        )
        return None

    # Show baseline (current config weights)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    wkey = "aggregator" if "aggregator" in cfg else "detection"
    cur_w = cfg.get(wkey, {}).get("weights", {"calibration": 0.20, "semantic_uncertainty": 0.50, "cross_check": 0.30})
    baseline = score_weights(results, cur_w.get("calibration", 0.20), cur_w.get("semantic_uncertainty", 0.50), cur_w.get("cross_check", 0.30))
    logger.info(f"Baseline AUROC (current weights): {baseline['auroc']:.4f}")

    # Run optimization
    if method == "bayesian":
        best = bayesian_search(results)
    else:
        best = grid_search(results)

    improvement = best["auroc"] - baseline["auroc"]

    print(f"\n{'='*50}")
    print(f"  Weight tuning results")
    print(f"{'='*50}")
    print(f"  Method          : {method}")
    print(f"  Labeled examples: {len(labeled)}")
    print(f"  Baseline AUROC  : {baseline['auroc']:.4f}")
    print(f"  Best AUROC      : {best['auroc']:.4f}  (+{improvement:.4f})")
    print(f"  Best AP         : {best['ap']:.4f}")
    print(f"  Best weights:")
    print(f"    calibration   : {best['w1']}")
    print(f"    uncertainty   : {best['w2']}")
    print(f"    cross_check   : {best['w3']}")
    print(f"{'='*50}\n")

    if write_back:
        write_weights_to_config(best)
        print(f"  Weights written to config.yaml")
        print(f"  Restart Streamlit for changes to take effect.\n")
    else:
        print(f"  --no-write-back set — config.yaml not modified.\n")

    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tune aggregation weights to maximize AUROC")
    p.add_argument("--input",         required=True, help="Path to eval_results JSON file")
    p.add_argument("--method",        default="grid", choices=["grid", "bayesian"])
    p.add_argument("--no-write-back", action="store_true", help="Don't update config.yaml")
    args = p.parse_args()
    run(args.input, args.method, write_back=not args.no_write_back)


def tune_weights(
    eval_results_path: str = "eval_results.json",
    write_back: bool = True,
) -> dict | None:
    """Backward compatibility wrapper for tune_weights."""
    res = run(eval_results_path, method="grid", write_back=write_back)
    if res is None:
        return None
    return {
        "calibration": res["w1"],
        "semantic_uncertainty": res["w2"],
        "cross_check": res["w3"],
    }


