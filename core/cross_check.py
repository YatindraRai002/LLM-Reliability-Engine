import torch
import numpy as np
import logging
import asyncio
from models.model_loader import get_nli_model
from models.groq_client import groq_client
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Relative path for config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

NLI_LABELS = ["contradiction", "entailment", "neutral"]

async def nli_score_async(premise: str, hypothesis: str) -> dict:
    """
    Score whether premise entails/contradicts/is neutral to hypothesis.
    Uses cross-encoder/nli-deberta-v3-base.
    """
    try:
        # Move this to a thread because it's CPU/GPU bound and not async
        def _compute():
            tokenizer, model = get_nli_model()
            max_len = 240
            prem_tokens = tokenizer.encode(premise, add_special_tokens=False)[:max_len]
            hyp_tokens = tokenizer.encode(hypothesis, add_special_tokens=False)[:max_len]
            premise_trunc = tokenizer.decode(prem_tokens)
            hypothesis_trunc = tokenizer.decode(hyp_tokens)

            inputs = tokenizer(
                premise_trunc,
                hypothesis_trunc,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(model.device)

            with torch.no_grad():
                logits = model(**inputs).logits

            probs = torch.softmax(logits[0], dim=-1).tolist()
            scores = dict(zip(NLI_LABELS, probs))

            return {
                "contradiction": scores["contradiction"],
                "entailment": scores["entailment"],
                "neutral": scores["neutral"],
                "verdict": max(scores, key=scores.get),
                "agreement_score": scores["entailment"] - scores["contradiction"]
            }

        return await asyncio.to_thread(_compute)
    except Exception as e:
        logger.error(f"NLI score error: {e}")
        return {"contradiction": 0, "entailment": 0, "neutral": 1, "verdict": "neutral", "agreement_score": 0}

async def run_cross_check_async(prompt: str, response_a: str) -> dict:
    """
    Get response from second model (Groq), compare via NLI.
    response_a is from the local model.
    """
    # Get Groq's response at temperature 0 (deterministic)
    try:
        response_b = await groq_client.generate(prompt, temperature=0.0)
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        response_b = ""

    if not response_b:
        return {
            "cross_check_uncertainty": 1.0,
            "verdict": "neutral",
            "agreement_score": 0,
            "open_model_response": response_a,
            "groq_response": "Error fetching Groq response",
            "symmetric_agreement": 0.0
        }

    # NLI in both directions - run in parallel
    ab_task = nli_score_async(response_a, response_b)
    ba_task = nli_score_async(response_b, response_a)

    ab_scores, ba_scores = await asyncio.gather(ab_task, ba_task)

    symmetric_agreement = (ab_scores["agreement_score"] + ba_scores["agreement_score"]) / 2

    # Map symmetric_agreement [-1, 1] -> uncertainty [1, 0]
    cross_check_uncertainty = float(1.0 - (symmetric_agreement + 1.0) / 2.0)

    return {
        "open_model_response": response_a,
        "groq_response": response_b,
        "ab_nli": ab_scores,
        "ba_nli": ba_scores,
        "symmetric_agreement": symmetric_agreement,
        "cross_check_uncertainty": cross_check_uncertainty,
        "verdict": "agree" if symmetric_agreement > 0.3 else
                   "neutral" if symmetric_agreement > -0.1 else "contradict"
    }
