"""
core/aggregator.py
Weighted score fusion. When Groq is unavailable, switches to
2-signal mode (calibration + uncertainty only) instead of
giving a misleading cross-check score.
"""

import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


@dataclass
class HallucinationResult:
    score:              float
    label:              str
    explanation:        str
    calibration_score:  float
    uncertainty_score:  float
    cross_check_score:  float
    weights_used:       dict
    thresholds_used:    dict
    n_samples_used:     int
    groq_available:     bool    # NEW — tells UI whether cross-check ran
    mode:               str     # "full" or "2-signal"

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_scores(
    calibration_score:  float,
    uncertainty_score:  float,
    cross_check_result: dict,           # full cc dict, not just a float
    weights:            Optional[dict] = None,
    n_samples:          int = None,
) -> HallucinationResult:
    """
    Fuse signals into final risk score.
    
    When groq_available=False in cross_check_result:
      - Redistributes cross-check weight to cal + uncertainty
      - Does NOT apply hard NLI override
      - Sets mode="2-signal" for UI display
    """
    cal = float(np.clip(calibration_score,  0.0, 1.0))
    unc = float(np.clip(uncertainty_score,  0.0, 1.0))
    
    groq_available = cross_check_result.get("groq_available", True)
    cc_raw         = cross_check_result.get("cross_check_uncertainty", 0.5)
    verdict        = cross_check_result.get("verdict", "neutral")
    cc             = float(np.clip(cc_raw, 0.0, 1.0))
    
    # Load base weights from config (spec: calibration=0.20, semantic_uncertainty=0.50, cross_check=0.30)
    w = dict(weights or CONFIG["detection"]["weights"])
    w1 = w.get("calibration",          0.20)  # spec default
    w2 = w.get("semantic_uncertainty",  0.50)  # spec default — was wrongly 0.30
    w3 = w.get("cross_check",           0.30)  # spec default — was wrongly 0.45
    
    mode = "full"
    
    if not groq_available:
        # 2-signal mode: redistribute cross-check weight
        # Split it 40/60 to uncertainty/calibration (uncertainty is more reliable)
        w1 = w1 + w3 * 0.40
        w2 = w2 + w3 * 0.60
        w3 = 0.0
        cc = 0.5   # neutral
        mode = "2-signal"
        logger.info("Aggregator: running in 2-signal mode (Groq unavailable)")
    
    # Normalize weights
    total = w1 + w2 + w3
    if total > 0:
        w1, w2, w3 = w1/total, w2/total, w3/total
    
    # Base weighted score
    score = w1 * cal + w2 * unc + w3 * cc
    
    # Hard NLI override — ONLY when Groq actually ran AND returned contradiction
    # NEVER fire this when Groq failed — that was the source of false 1.0 scores
    if groq_available and verdict == "contradict":
        pre_override = score
        score = max(score, 0.85)
        logger.info(
            f"Hard NLI override fired: {pre_override:.3f} → {score:.3f} "
            f"(verdict=contradict, agreement={cross_check_result.get('symmetric_agreement',0):.3f})"
        )
    
    score = float(np.clip(score, 0.0, 1.0))
    
    # Risk label (using detection/thresholds from config.yaml)
    thr = CONFIG["detection"]["thresholds"]
    if score < thr["low"]:
        label = "low"
    elif score < thr["medium"]:
        label = "medium"
    else:
        label = "high"
    
    n_s = n_samples or CONFIG["sampling"]["n_samples"]
    
    return HallucinationResult(
        score             = round(score, 3),
        label             = label,
        explanation       = _explain(label, cal, unc, cc, n_s, groq_available, verdict),
        calibration_score = round(cal, 3),
        uncertainty_score = round(unc, 3),
        cross_check_score = round(cc,  3),
        weights_used      = {
            "calibration":          round(w1, 3),
            "semantic_uncertainty": round(w2, 3),  # unified — was 'uncertainty'
            "cross_check":          round(w3, 3),
        },
        thresholds_used   = thr,
        n_samples_used    = n_s,
        groq_available    = groq_available,
        mode              = mode,
    )


def _explain(label, cal, unc, cc, n, groq_available, verdict) -> str:
    mode_note = "" if groq_available else " (Groq unavailable — 2-signal mode)"
    
    if label == "low":
        return (
            f"Response appears reliable{mode_note}. "
            f"Token confidence is {'high' if cal<0.3 else 'moderate'}, "
            f"responses are {'consistent' if unc<0.3 else 'somewhat varied'} across {n} samples."
        )
    if label == "medium":
        dominant = max(
            [("calibration", cal), ("uncertainty", unc), ("cross-check", cc)],
            key=lambda x: x[1]
        )
        return (
            f"Moderate hallucination risk{mode_note}. "
            f"Strongest signal: {dominant[0]} ({dominant[1]:.2f}). "
            "Verify key claims independently."
        )
    # high
    triggers = []
    if cal > 0.55:  triggers.append(f"low token confidence ({cal:.2f})")
    if unc > 0.55:  triggers.append(f"high response variance ({unc:.2f})")
    if cc  > 0.55 and groq_available:
        triggers.append(f"model disagreement ({cc:.2f})")
    if groq_available and verdict == "contradict":
        triggers.append("NLI contradiction override")
    
    trigger_str = "; ".join(triggers) or "multiple signals"
    return (
        f"High hallucination risk{mode_note}. "
        f"Triggered by: {trigger_str}. "
        "Do not rely on this response without verification."
    )


def run_full_pipeline(
    prompt: str,
    use_local_for_uncertainty: bool = False,
    weights: Optional[dict] = None,
    explain: bool = True,
) -> dict:
    # NOTE: `explain=False` sets explanation_detail={} in the returned dict.
    # Downstream consumers (UI, tests, analytics) must use .get() to access
    # explanation fields — never assume the key contains a populated dict.
    from core.calibration          import get_generation_with_scores, compute_calibration_score
    from core.semantic_uncertainty import run_semantic_uncertainty_pipeline
    from core.cross_check          import run_cross_check

    t0 = time.time()
    timings = {}
    logger.info(f"=== Pipeline: '{prompt[:60]}' ===")

    # Step 1: Calibration
    t = time.time()
    logger.info("Step 1/3: Calibration...")
    try:
        cal_detail = get_generation_with_scores(prompt)
        cal_score  = compute_calibration_score(cal_detail["token_probs"])
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        from core.calibration import _fallback_result
        cal_detail = _fallback_result(reason=str(e))
        cal_score  = 0.5
    timings["calibration"] = round(time.time()-t, 2)
    logger.info(f"  cal={cal_score:.3f} ({timings['calibration']}s)")

    # Run Step 2 (Semantic Uncertainty) and Step 3 (Cross-check) in parallel
    from concurrent.futures import ThreadPoolExecutor
    t23 = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_sem = executor.submit(run_semantic_uncertainty_pipeline, prompt, use_local_for_uncertainty)
        future_cc = executor.submit(run_cross_check, prompt, cal_detail.get("response", ""))
        
        # Step 2: Semantic uncertainty processing
        logger.info("Step 2/3: Semantic uncertainty...")
        try:
            sem_detail = future_sem.result()
            sem_score = sem_detail["uncertainty_score"]
        except Exception as e:
            logger.error(f"Semantic uncertainty failed: {e}")
            sem_detail = {
                "uncertainty_score":        0.5,
                "n_semantic_clusters":      1,
                "normalized_entropy":       0.5,
                "responses":                [],
                "embeddings_2d":            [],
                "cluster_labels":           [],
                "mean_pairwise_similarity": 1.0,
            }
            sem_score = 0.5
        timings["semantic_uncertainty"] = round(time.time()-t23, 2)
        logger.info(f"  unc={sem_score:.3f} ({timings['semantic_uncertainty']}s)")

        # Step 3: Cross-check processing
        logger.info("Step 3/3: Cross-check...")
        try:
            cc_detail = future_cc.result()
            cc_score  = cc_detail["cross_check_uncertainty"]
        except Exception as e:
            logger.error(f"Cross-check failed: {e}")
            cc_detail = {
                "local_response":          cal_detail.get("response",""),
                "groq_response":           None,
                "groq_available":          False,
                "error":                   str(e),
                "verdict":                 "unavailable",
                "cross_check_uncertainty": 0.5,
                "symmetric_agreement":     0.0,
                "ab_detail": {}, "ba_detail": {},
            }
            cc_score = 0.5
        timings["cross_check"] = round(time.time()-t23, 2)
        logger.info(f"  cc={cc_score:.3f} ({timings['cross_check']}s)")

    # Step 4: Aggregate
    result = aggregate_scores(
        cal_score, sem_score, cc_detail,
        weights=weights,
        n_samples=len(sem_detail.get("responses", [])),
    )
    timings["total"] = round(time.time()-t0, 2)
    logger.info(f"=== Done: {result.score:.3f} ({result.label}) mode={result.mode} ===")

    # ── Step 5: Explanation (Phase B) ──────────────────────────────────────
    # Gated by the `explain` parameter — set False in evaluation harness or
    # when the UI "Show explanation" checkbox is unchecked to avoid NLI overhead.
    # explanation_detail is always present in the return dict; it may be {} when
    # explain=False. All callers must use result_dict.get("explanation_detail", {}).
    if explain:
        from core.explainer import generate_explanation
        from dataclasses import asdict

        explanation_obj = generate_explanation(
            response=cal_detail.get("response", ""),
            token_probs=cal_detail.get("token_probs", []),
            local_response=cal_detail.get("response", ""),
            groq_response=cc_detail.get("groq_response") or None,
            cal=cal_score,
            unc=sem_score,
            cc=cc_score,
            weights={
                "calibration":          result.weights_used["calibration"],
                "semantic_uncertainty": result.weights_used["semantic_uncertainty"],
                "cross_check":          result.weights_used["cross_check"],
            },
        )
        explanation_detail = explanation_obj.to_dict()
    else:
        logger.info("run_full_pipeline: explain=False — skipping explanation engine")
        explanation_detail = {}  # downstream: always use .get() on this

    return {
        "prompt":             prompt,
        "result":             result,
        "calibration_detail": cal_detail,
        "calibration_score":  cal_score,
        "uncertainty_detail": sem_detail,
        "uncertainty_score":  sem_score,
        "cross_check_detail": cc_detail,
        "cross_check_score":  cc_score,
        "timings":            timings,
        "explanation_detail": explanation_detail,
    }
