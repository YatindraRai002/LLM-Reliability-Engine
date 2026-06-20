"""
core/cross_check.py
Signal 3 — Multi-model cross-checking via NLI.
Uses safe_groq_cross_check() so Groq failures degrade gracefully.
"""

import logging
import os

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)

_NLI_IDX = {"contradiction": 0, "entailment": 1, "neutral": 2}


def _truncate(text: str, tokenizer, max_tokens: int = 220) -> str:
    """Truncate text to fit NLI model 512-token window."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:180] + ids[-40:], skip_special_tokens=True)


def nli_score(premise: str, hypothesis: str) -> dict:
    """
    NLI score between premise and hypothesis.
    Returns contradiction, entailment, neutral probabilities.
    """
    from models.model_loader import get_nli_model
    
    tokenizer, model = get_nli_model()
    device = next(model.parameters()).device
    
    p = _truncate(premise, tokenizer)
    h = _truncate(hypothesis, tokenizer)
    
    inputs = tokenizer(
        p, h,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs   = torch.softmax(logits[0], dim=-1).cpu().tolist()
    scores  = {label: probs[idx] for label, idx in _NLI_IDX.items()}
    verdict = max(scores, key=scores.get)
    
    return {
        "contradiction": round(scores["contradiction"], 4),
        "entailment":    round(scores["entailment"],    4),
        "neutral":       round(scores["neutral"],       4),
        "verdict":       verdict,
        "agreement":     round(scores["entailment"] - scores["contradiction"], 4),
    }


def symmetric_nli(response_a: str, response_b: str) -> dict:
    """Bidirectional NLI — score A→B and B→A, then average."""
    ab = nli_score(response_a, response_b)
    ba = nli_score(response_b, response_a)
    
    sym = (ab["agreement"] + ba["agreement"]) / 2.0
    uncertainty = float(np.clip((1.0 - sym) / 2.0, 0.0, 1.0))
    
    if sym > 0.25:
        verdict = "agree"
    elif sym < -0.15:
        verdict = "contradict"
    else:
        verdict = "neutral"
    
    return {
        "ab_scores":               ab,
        "ba_scores":               ba,
        "symmetric_agreement":     round(sym, 4),
        "cross_check_uncertainty": round(uncertainty, 4),
        "verdict":                 verdict,
    }


def _clean_response_for_nli(text: str) -> str:
    """
    Ensure text ends at a sentence boundary for better NLI scoring.
    Truncated sentences confuse the NLI model and produce neutral verdicts.
    """
    if not text:
        return text
    last_end = max(
        text.rfind('.'),
        text.rfind('!'),
        text.rfind('?'),
    )
    if last_end > len(text) * 0.5:
        return text[:last_end + 1].strip()
    return text.strip()


def run_cross_check(prompt: str, local_response: str) -> dict:
    """
    Full cross-check pipeline.
    Uses safe_groq_cross_check so a bad API key never crashes the app.
    
    Returns dict with 'groq_available' flag so aggregator can
    switch to 2-signal mode when Groq is down.
    """
    from models.groq_client import safe_groq_cross_check
    
    groq_result = safe_groq_cross_check(prompt)
    
    if not groq_result["groq_available"]:
        logger.warning(
            f"Groq unavailable — cross-check degraded. "
            f"Error: {groq_result.get('error', 'unknown')}"
        )
        return {
            "local_response":          local_response,
            "groq_response":           None,
            "groq_available":          False,
            "groq_model":              CONFIG["models"]["groq"]["name"],
            "error":                   groq_result.get("error"),
            "error_type":              groq_result.get("error_type", "unknown"),
            "nli":                     None,
            "cross_check_uncertainty": 0.5,
            "verdict":                 "unavailable",
            "symmetric_agreement":     0.0,
            "ab_detail":               {},
            "ba_detail":               {},
        }
    
    groq_response = groq_result["groq_response"]
    logger.info(f"Groq response: {groq_response[:80]}...")
    
    clean_local = _clean_response_for_nli(local_response)
    clean_groq  = _clean_response_for_nli(groq_response)
    nli_result = symmetric_nli(clean_local, clean_groq)
    
    logger.info(
        f"Cross-check: verdict={nli_result['verdict']} "
        f"agreement={nli_result['symmetric_agreement']:.3f} "
        f"uncertainty={nli_result['cross_check_uncertainty']:.3f}"
    )
    
    return {
        "local_response":          local_response,
        "groq_response":           groq_response,
        "groq_available":          True,
        "groq_model":              CONFIG["models"]["groq"]["name"],
        "error":                   None,
        "nli":                     nli_result,
        "cross_check_uncertainty": nli_result["cross_check_uncertainty"],
        "verdict":                 nli_result["verdict"],
        "symmetric_agreement":     nli_result["symmetric_agreement"],
        "ab_detail":               nli_result["ab_scores"],
        "ba_detail":               nli_result["ba_scores"],
    }


nli_score_sync = nli_score
