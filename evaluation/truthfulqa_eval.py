import json
import time
import logging
import numpy as np
from datasets import load_dataset
from core.aggregator import run_full_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_truthfulqa_eval(n_samples: int = 50, output_file: str = "eval_results.json"):
    """
    Runs the full lie-detector pipeline on a subset of TruthfulQA.
    """
    logger.info(f"Loading TruthfulQA dataset...")
    try:
        dataset = load_dataset("truthful_qa", "generation")["validation"]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []

    # Focus on categories where LLMs commonly hallucinate
    target_categories = ["Finance", "Health", "Law", "Conspiracies", "Misconceptions"]
    filtered = [ex for ex in dataset if ex["category"] in target_categories][:n_samples]
    
    results = []
    logger.info(f"Evaluating {len(filtered)} samples...")
    
    for i, example in enumerate(filtered):
        question = example["question"]
        category = example["category"]
        best_answer = example["best_answer"]
        
        logger.info(f"[{i+1}/{len(filtered)}] Testing: {question[:60]}...")
        
        start = time.time()
        try:
            pipeline_result = run_full_pipeline(question)
            elapsed = time.time() - start
            
            results.append({
                "question": question,
                "category": category,
                "best_answer": best_answer,
                "model_response": pipeline_result["calibration_detail"]["response"],
                "hallucination_score": pipeline_result["result"].score,
                "risk_label": pipeline_result["result"].label,
                "elapsed_seconds": elapsed,
                "calibration": pipeline_result["result"].calibration_score,
                "uncertainty": pipeline_result["result"].uncertainty_score,
                "cross_check": pipeline_result["result"].cross_check_score,
            })
        except Exception as e:
            logger.error(f"Pipeline failed for question {i}: {e}")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary Stats
    if not results:
        return []
        
    high_risk = sum(1 for r in results if r["risk_label"] == "high")
    avg_latency = sum(r["elapsed_seconds"] for r in results) / len(results)
    
    logger.info("\n" + "="*30)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"Samples Processed: {len(results)}")
    logger.info(f"High Risk Flagged: {high_risk}/{len(results)} ({100*high_risk/len(results):.1f}%)")
    logger.info(f"Avg Latency: {avg_latency:.2f}s per query")
    logger.info("="*30)
    
    return results
