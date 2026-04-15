"""
Diagnostic script to verify the detection pipeline for specific hallucinations.
Usage: python diagnose_hallucination.py "Who invented the telephone?"
"""
import sys
import os
import json
from core.aggregator import run_full_pipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_hallucination.py 'Your question here'")
        return

    query = sys.argv[1]
    print(f"\n--- Diagnostic Run for Query: '{query}' ---")
    
    try:
        # Run the full pipeline
        results = run_full_pipeline(query)
        
        res = results["result"]
        print(f"\n[FINAL SCORE]: {res.score:.3f} ({res.label.upper()})")
        print(f"[EXPLANATION]: {res.explanation}")
        
        print("\n[SIGNAL BREAKDOWN]:")
        print(f"  - Calibration: {res.calibration_score:.3f} (Weight: {res.weights_used['w1']:.2f})")
        print(f"  - Semantic Uncertainty: {res.uncertainty_score:.3f} (Weight: {res.weights_used['w2']:.2f})")
        print(f"  - Cross-Check (Groq): {res.cross_check_score:.3f} (Weight: {res.weights_used['w3']:.2f})")
        
        print("\n[DETAILED RESPONSES]:")
        print(f"  - Local Model (Primary): {results['calibration_detail']['response']}")
        print(f"  - Groq Model (Reference): {results['cross_check_detail']['groq_response']}")
        print(f"  - NLI Verdict: {results['cross_check_detail']['verdict']}")
        print(f"  - Symmetric Agreement: {results['cross_check_detail']['symmetric_agreement']:.3f}")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
