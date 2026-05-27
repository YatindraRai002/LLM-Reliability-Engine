import numpy as np
import yaml
import os
import asyncio
import torch
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "config.yaml")

def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

CONFIG = load_config()

from core.calibration import PlattCalibrator
calibrator_path = os.path.join(CURRENT_DIR, "..", "calibrator.pkl")
calibrator = PlattCalibrator()
if os.path.exists(calibrator_path):
    calibrator.load(calibrator_path)
else:
    logger.warning(f"Calibrator not found at {calibrator_path}, will use raw scores.")

@dataclass
class HallucinationResult:
    score: float
    label: str
    calibration_score: float
    uncertainty_score: float
    cross_check_score: float
    weights_used: dict
    explanation: str

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "calibration_score": self.calibration_score,
            "uncertainty_score": self.uncertainty_score,
            "cross_check_score": self.cross_check_score,
            "weights_used": self.weights_used,
            "explanation": self.explanation,
        }

def aggregate_scores(
    calibration_score: float,
    uncertainty_score: float,
    cross_check_score: float,
    verdict: str = "neutral",
    weights: dict = None
) -> HallucinationResult:

    def sanitize(val):
        try:
            return float(np.clip(val, 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.5

    c_score = sanitize(calibration_score)
    u_score = sanitize(uncertainty_score)
    x_score = sanitize(cross_check_score)

    w = weights or CONFIG.get("detection", {}).get("weights", {"calibration": 0.25, "semantic_uncertainty": 0.3, "cross_check": 0.45})
    w1 = w.get("calibration", 0.25)
    w2 = w.get("semantic_uncertainty", 0.3)
    w3 = w.get("cross_check", 0.45)

    total_w = w1 + w2 + w3
    w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w

    # Risk-Averse Fusion: Combine weighted average with max-signal penalty
    base_avg = (w1 * c_score) + (w2 * u_score) + (w3 * x_score)
    max_signal = max(c_score, u_score, x_score)
    final_score = max(base_avg, max_signal * 0.7)

    # NLI Hard-Override: High risk if contradiction is found
    if verdict == "contradict":
        final_score = max(final_score, 0.85)

    final_score = float(np.clip(final_score, 0.0, 1.0))

    # Apply Platt Calibration
    if calibrator is not None:
        final_score = calibrator.transform(final_score)
        final_score = float(np.clip(final_score, 0.0, 1.0))

    thresholds = CONFIG.get("detection", {}).get("thresholds", {"low": 0.25, "medium": 0.6})

    if final_score < thresholds.get("low", 0.25):
        label = "low"
        explanation = "Model shows consistent, confident responses across all signals."
    elif final_score < thresholds.get("medium", 0.6):
        label = "medium"
        dominant = max([("calibration", c_score), ("uncertainty", u_score), ("cross-check", x_score)], key=lambda x: x[1])
        explanation = f"Moderate hallucination risk. Primary signal: {dominant[0]} ({dominant[1]:.2f})"
    else:
        label = "high"
        signals = []
        if c_score > 0.6: signals.append(f"low token confidence ({c_score:.2f})")
        if u_score > 0.6: signals.append(f"high response variance ({u_score:.2f})")
        if x_score > 0.6: signals.append(f"model disagreement ({x_score:.2f})")
        if verdict == "contradict": signals.insert(0, "NLI contradiction detected")

        if x_score > 0.8:
            explanation = f"Critical hallucination detected: Severe disagreement between models. {', '.join(signals)}"
        else:
            explanation = f"Likely hallucination. Triggered by: {', '.join(signals) or 'all signals'}."

    return HallucinationResult(
        score=round(final_score, 3),
        label=label,
        calibration_score=c_score,
        uncertainty_score=u_score,
        cross_check_score=x_score,
        weights_used={"w1": w1, "w2": w2, "w3": w3},
        explanation=explanation
    )

def detect_model_collapse(text: str) -> bool:
    """Detects if the model has entered a repetition loop or produced gibberish."""
    if not text or len(text) < 10:
        return False

    # Check for repetitive phrases (3 or more times)
    words = text.split()
    if len(words) < 5:
        return False

    for n in range(2, 5): # Check n-grams from 2 to 4 words
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        counts = {}
        for ng in ngrams:
            counts[ng] = counts.get(ng, 0) + 1
            if counts[ng] >= 3:
                return True
    return False

async def run_correction_loop_async(prompt: str, draft_response: str) -> dict:
    """Self-correction loop using adversarial prompting."""
    from models.model_loader import get_open_model

    # Create a local lock for this specific pipeline run since global locks
    # cause "different event loop" errors in Streamlit.
    gpu_lock = asyncio.Lock()

    tokenizer, model = get_open_model()

    # Adversarial Critique phase
    critique_prompt = (
        f"<|system|>\nYou are a critical fact-checker. Your goal is to determine if the Draft Answer is factually correct.\n\n"
        f"Approach:\n1. Extract the primary claims from the Draft Answer.\n2. Evaluate each claim independently.\n"
        f"3. If a claim is unsupported or incorrect, provide a specific counter-fact.\n4. If the answer is correct, state 'The answer is accurate'.\n</s>\n"
        f"<|user|>\nQuestion: {prompt}\nDraft Answer: {draft_response}</s>\n<|assistant|>\n"
    )

    async with gpu_lock:
        inputs = tokenizer(critique_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            critique_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True, # Enable KV Cache
                top_k=50,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            )[0][inputs["input_ids"].shape[1]:]
        critique_text = tokenizer.decode(critique_ids, skip_special_tokens=True)

    # Synthesis phase
    final_prompt = (
        f"<|system|>\nYou are a synthesis expert. Compare the original draft and the critique to produce the most accurate response. "
        f"If the critique identified a factual error, correct it. If the critique is vague, prioritize the most factual information. "
        f"If you are unsure, state that you do not know.\n</s>\n"
        f"<|user|>\nQuestion: {prompt}\nDraft: {draft_response}\nCritique: {critique_text}</s>\n<|assistant|>\n"
    )

    async with gpu_lock:
        inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            final_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True, # Enable KV Cache
                top_k=50,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            )[0][inputs["input_ids"].shape[1]:]
        final_response = tokenizer.decode(final_ids, skip_special_tokens=True)

    return {
        "draft": draft_response,
        "critique": critique_text,
        "final": final_response
    }

async def async_run_full_pipeline(prompt: str, weights: dict = None) -> dict:
    from core.semantic_uncertainty import run_semantic_uncertainty_pipeline_async
    from core.cross_check import run_cross_check_async
    from core.calibration import compute_calibration_from_batch
    from core.sanitizer import sanitize_prompt
    from core.cache import get_cached, set_cached

    safe_prompt = sanitize_prompt(prompt)
    if not safe_prompt:
        raise ValueError("Invalid query after sanitization")

    cached = get_cached(safe_prompt, weights)
    if cached:
        # If it was loaded from JSON, result is a dict, need to convert back to HallucinationResult
        # But we can also just let the caller handle dicts, or convert it here.
        # Streamlit components expect result as HallucinationResult object in current_result["result"].
        # So we reconstruct it.
        if isinstance(cached.get("result"), dict):
            cached["result"] = HallucinationResult(**cached["result"])
        return cached

    try:
        # 1. Parallel Signal Gathering
        sem_task = asyncio.create_task(run_semantic_uncertainty_pipeline_async(safe_prompt))

        sem_result = await sem_task
        draft_response = sem_result["responses"][0] if sem_result["responses"] else ""

        # Fail-Safe 1: Detect Model Collapse (Repetition/Gibberish)
        if detect_model_collapse(draft_response):
            return {
                "prompt": prompt,
                "result": aggregate_scores(1.0, 1.0, 1.0, verdict="contradict"), # Force High Risk
                "calibration_detail": {"response": "Model failure detected", "mean_confidence": 0.0},
                "uncertainty_detail": sem_result,
                "cross_check_detail": {"cross_check_uncertainty": 1.0, "verdict": "collapse"},
                "correction_detail": {"draft": draft_response, "critique": "Catastrophic repetition detected.", "final": "I'm sorry, the model encountered a generation error and cannot provide a factual answer."}
            }

        cc_result = await run_cross_check_async(safe_prompt, draft_response)

        # 2. Derive Calibration from the batch
        cal_score = compute_calibration_from_batch(sem_result["responses"])

        # Fail-Safe 2: Calibration Floor
        if cal_score > 0.9:
            cal_score = 1.0

        # 3. Aggregate
        result = aggregate_scores(
            cal_score,
            sem_result["uncertainty_score"],
            cc_result["cross_check_uncertainty"],
            verdict=cc_result.get("verdict", "neutral"),
            weights=weights
        )

        # 4. Conditional Self-Correction Loop (Latency Optimization)
        # If the risk is low, we skip correction entirely to save GPU time.
        if result.label == "low":
            correction_detail = {
                "draft": draft_response,
                "critique": "No correction needed. Signal confidence is high.",
                "final": draft_response
            }
        else:
            correction_detail = await run_correction_loop_async(safe_prompt, draft_response)

        # 5. Generate Explanations
        try:
            from core.explainer import generate_explanation
            explanation_detail = generate_explanation(
                response=correction_detail["final"],
                token_probs=sem_result.get("token_probs", []),
                local_response=draft_response,
                groq_response=cc_result.get("groq_response", ""),
                cal=cal_score,
                unc=sem_result["uncertainty_score"],
                cc=cc_result["cross_check_uncertainty"],
                weights=result.weights_used,
            )
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            explanation_detail = None

        final_dict = {
            "prompt": prompt,
            "result": result,
            "calibration_detail": {"response": correction_detail["final"], "mean_confidence": cal_score},
            "uncertainty_detail": sem_result,
            "cross_check_detail": cc_result,
            "correction_detail": correction_detail,
            "explanation_detail": explanation_detail,
        }
        
        # Need to convert HallucinationResult to dict for caching
        cache_dict = final_dict.copy()
        cache_dict["result"] = cache_dict["result"].to_dict()
        set_cached(safe_prompt, weights, cache_dict)

        return final_dict
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM occurred during pipeline execution")
        return {
            "prompt": prompt,
            "result": aggregate_scores(1.0, 1.0, 1.0, verdict="error"),
            "error": "GPU Out of Memory. Please try a shorter prompt.",
            "calibration_detail": {}, "uncertainty_detail": {}, "cross_check_detail": {}, "correction_detail": {}
        }
    except Exception as e:
        logger.exception(f"Unexpected pipeline error: {e}")
        return {
            "prompt": prompt,
            "result": aggregate_scores(0.5, 0.5, 0.5, verdict="error"),
            "error": f"Internal Pipeline Error: {str(e)}",
            "calibration_detail": {}, "uncertainty_detail": {}, "cross_check_detail": {}, "correction_detail": {}
        }

def run_full_pipeline(prompt: str, weights: dict = None) -> dict:
    """Synchronous wrapper for Streamlit compatibility."""
    # We create a new event loop for each request to avoid "different event loop" errors
    # especially since Streamlit's execution environment can be tricky with asyncio.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_run_full_pipeline(prompt, weights=weights))
    finally:
        loop.close()
