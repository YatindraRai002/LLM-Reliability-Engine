import sys
import os
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from core.aggregator import run_full_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

test_queries = [
    "What is the capital of France?",
    "Who is the current president of the United States?",
    "Tell me about the Great Martian War of 1924"
]

print("\n" + "="*50)
print(" RUNNING FULL PROJECT PIPELINE (HEADLESS MODE)")
print("="*50 + "\n")

for q in test_queries:
    print(f"\nQuery: {q}")
    try:
        result = run_full_pipeline(q)
        print(f"Final Risk Score: {result['result'].score}")
        print(f"Label: {result['result'].label}")
        print(f"Explanation: {result['result'].explanation}")
        print(f"Local Response: {result['calibration_detail']['response'][:100]}...")
        print(f"Groq Response: {result['cross_check_detail']['groq_response'][:100]}...")
    except Exception as e:
        print(f"Error processing query: {e}")
    print("-" * 30)

print("\n" + "="*50)
print(" FULL PROJECT VERIFICATION COMPLETE")
print("="*50)
