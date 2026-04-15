import time
import logging
import numpy as np
from core.calibration import get_generation_with_scores, compute_calibration_score
from core.semantic_uncertainty import generate_n_samples_parallel, compute_semantic_uncertainty
from core.cross_check import run_cross_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def profile_pipeline(prompt: str):
    """
    Profiles the latency of each individual component in the pipeline.
    """
    timings = {}
    
    logger.info(f"Profiling pipeline for prompt: {prompt}")
    
    # 1. Calibration Generation
    t = time.time()
    cal_res = get_generation_with_scores(prompt)
    timings["calibration_gen"] = time.time() - t
    
    # 2. Calibration Scoring
    t = time.time()
    compute_calibration_score(cal_res["token_probs"])
    timings["calibration_score"] = time.time() - t
    
    # 3. Sampling (The most expensive part)
    t = time.time()
    samples = generate_n_samples_parallel(prompt)
    timings["sampling"] = time.time() - t
    
    # 4. Semantic Clustering
    t = time.time()
    compute_semantic_uncertainty(samples)
    timings["clustering"] = time.time() - t
    
    # 5. Cross-Check (API Call)
    t = time.time()
    run_cross_check(prompt, cal_res["response"])
    timings["cross_check"] = time.time() - t
    
    total = sum(timings.values())
    print("\n=== Latency Breakdown ===")
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:.2f}s  ({100*v/total:.1f}%)")
    print(f"  {'TOTAL':<25} {total:.2f}s")
    
    return timings
