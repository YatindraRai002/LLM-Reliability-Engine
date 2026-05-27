"""
core/calibration.py
Signal 1 — Calibration-based uncertainty via token logit extraction.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def format_prompt(user_msg: str) -> str:
    """
    Apply the correct chat template for the configured local model.
    
    CRITICAL: Using the wrong template causes the model to generate
    meta-commentary instead of actual answers. Each model family has
    its own exact template — do not mix them.
    
    TinyLlama-1.1B-Chat-v1.0  → Zephyr/ChatML template
    Mistral-7B-Instruct        → [INST] template  
    Llama-2-Chat               → <<SYS>> template
    Llama-3-Instruct           → <|begin_of_text|> template
    """
    model_name = CONFIG["models"]["local"]["name"].lower() # Adjusted key based on config.yaml
    msg = user_msg.strip()
    
    if "tinyllama" in model_name:
        # TinyLlama-1.1B-Chat-v1.0 was fine-tuned with Zephyr chat template
        # Source: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
        return (
            "<|system|>\n"
            "You are a helpful, accurate assistant. "
            "Answer questions directly and concisely.</s>\n"
            f"<|user|>\n{msg}</s>\n"
            "<|assistant|>\n"
        )
    
    elif "mistral" in model_name:
        return f"<s>[INST] {msg} [/INST]"
    
    elif "llama-2" in model_name or "llama2" in model_name:
        return (
            f"<s>[INST] <<SYS>>\n"
            "You are a helpful assistant.\n"
            f"<</SYS>>\n\n{msg} [/INST]"
        )
    
    elif "llama-3" in model_name or "llama3" in model_name:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    else:
        # Generic fallback — no template, just raw message
        logger.warning(f"Unknown model '{model_name}' — using raw prompt. May produce poor results.")
        return msg


def get_generation_with_scores(
    prompt: str,
    max_new_tokens: Optional[int] = None,
) -> dict:
    """
    Generate response from local model AND capture per-token probabilities.
    
    Key requirements:
      - return_dict_in_generate=True  (enables outputs.scores)
      - output_scores=True            (captures logits at each step)
      - do_sample=False               (greedy = deterministic for calibration)
    
    Returns dict always containing 'token_probs' key (may be empty list on failure).
    """
    from models.model_loader import get_open_model # using my loader name

    # Fallback to my config structure if the user's doesn't match
    max_tokens = max_new_tokens or 64 
    tokenizer, model = get_open_model()

    # Apply correct chat template
    formatted = format_prompt(prompt)
    logger.debug(f"Formatted prompt (first 100 chars): {formatted[:100]}")

    try:
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[1]
        logger.debug(f"Input token count: {input_len}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,               # greedy decoding
                return_dict_in_generate=True,  # REQUIRED for .scores access
                output_scores=True,            # REQUIRED for logit capture
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,        # reduce repetition loops
            )

        # Verify scores were actually captured
        if not hasattr(outputs, 'scores') or outputs.scores is None:
            logger.error("output_scores not returned by model.generate(). "
                        "Check that return_dict_in_generate=True is set.")
            response = tokenizer.decode(
                outputs.sequences[0][input_len:], skip_special_tokens=True
            ).strip()
            return _fallback_result(response, reason="no_scores")

        # Decode generated text
        generated_ids = outputs.sequences[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        logger.debug(f"Raw response: {response[:100]}")

        # Extract per-token probabilities
        token_probs = []
        eos_id = tokenizer.eos_token_id
        
        for score_tensor, token_id in zip(outputs.scores, generated_ids):
            # score_tensor: [1, vocab_size] → squeeze → [vocab_size]
            probs = torch.softmax(score_tensor.squeeze(0), dim=-1)
            chosen_prob = probs[token_id].item()
            token_probs.append(chosen_prob)
            if token_id.item() == eos_id:
                break  # stop at end-of-sequence

        if not token_probs:
            logger.warning("No token probs captured — response may be empty")
            return _fallback_result(response, reason="empty_probs")

        arr = np.array(token_probs, dtype=np.float64)
        
        result = {
            "response":         response,
            "token_probs":      token_probs,        # always present
            "mean_confidence":  float(arr.mean()),
            "min_confidence":   float(arr.min()),
            "confidence_std":   float(arr.std()),
            "n_tokens":         len(token_probs),
            "success":          True,
        }
        
        logger.info(
            f"Calibration: {len(token_probs)} tokens, "
            f"mean_conf={arr.mean():.3f}, std={arr.std():.3f}"
        )
        return result

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return _fallback_result(reason=f"exception: {e}")


def _fallback_result(response: str = "[generation failed]", reason: str = "unknown") -> dict:
    """
    Safe fallback — always has 'token_probs' key so UI never crashes.
    Reason string helps with debugging.
    """
    logger.warning(f"Using calibration fallback (reason={reason})")
    return {
        "response":         response,
        "token_probs":      [],          # empty list, NOT missing key
        "mean_confidence":  0.5,
        "min_confidence":   0.5,
        "confidence_std":   0.0,
        "n_tokens":         0,
        "success":          False,
        "fallback_reason":  reason,
    }


def compute_calibration_score(token_probs: list) -> float:
    """
    Convert per-token probabilities → uncertainty score in [0, 1].
    
    0.0 = model is very confident
    1.0 = model is very uncertain
    
    If token_probs is empty (fallback case), returns 0.5 (neutral).
    """
    if not token_probs:
        return 0.5  # neutral — no information

    arr = np.array(token_probs, dtype=np.float64)
    
    # Clip to valid range (softmax should guarantee this, but be safe)
    arr = np.clip(arr, 1e-10, 1.0)
    
    mean_component     = 1.0 - float(arr.mean())
    variance_component = min(float(arr.std()) / 0.30, 1.0)
    
    score = 0.70 * mean_component + 0.30 * variance_component
    return float(np.clip(score, 0.0, 1.0))


def compute_ece(confidences: list, correct: list, n_bins: int = 10) -> float:
    """Expected Calibration Error. Lower = better calibrated."""
    if not confidences or not correct:
        return 0.0
    
    conf  = np.array(confidences)
    corr  = np.array(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
    
    return float(ece)
