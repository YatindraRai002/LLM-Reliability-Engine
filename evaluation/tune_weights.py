import json
import itertools
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tune_weights(eval_results_path: str):
    """
    Optimizes aggregator weights to maximize the separation 
    between low-risk and high-risk responses.
    """
    with open(eval_results_path) as f:
        results = json.load(f)
    
    if not results:
        logger.error("No results found to tune.")
        return None
    
    # Grid search for weights (w1+w2+w3 = 1.0)
    # Range: 0.1 to 0.8
    best_separation = 0
    best_weights = None
    
    # Generate possible weight combinations
    for w1 in np.arange(0.1, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.8, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0.1:
                continue
            
            # Calculate final scores with these weights
            scores = [
                w1 * r["calibration"] + w2 * r["uncertainty"] + w3 * r["cross_check"]
                for r in results
            ]
            
            # Metric: Standard deviation of scores 
            # (Higher variance usually means better separation between confident/uncertain)
            separation = np.std(scores)
            
            if separation > best_separation:
                best_separation = separation
                best_weights = (w1, w2, w3)
    
    if best_weights:
        logger.info(f"Best weights found: calibration={best_weights[0]:.2f}, "
                    f"uncertainty={best_weights[1]:.2f}, cross_check={best_weights[2]:.2f}")
        logger.info(f"Score separation (std): {best_separation:.4f}")
    
    return best_weights
