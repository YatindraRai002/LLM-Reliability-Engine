"""
Explanation Engine for the LLM Lie Detector — Phase B.

Provides three explanation sub-modules on top of the existing scoring pipeline:
  1. Token attribution       — flags spans where prob < mean × threshold
  2. Sentence-level NLI      — finds sentences that contradict the Groq oracle
  3. Signal SHAP (ablation)  — measures each signal's % contribution to the final score

This module is intentionally import-safe: no model is loaded at import time.
All model access is lazy (inside function bodies).

No circular imports: this file imports FROM core.cross_check, never the reverse.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as _f:
    _CONFIG = yaml.safe_load(_f)

_EXPL_CFG   = _CONFIG.get("explanation", {})
_TOKEN_CFG  = _EXPL_CFG.get("token_attribution", {})
_NLI_CFG    = _EXPL_CFG.get("sentence_nli", {})
_SHAP_CFG   = _EXPL_CFG.get("signal_shap", {})
_REC_CFG    = _EXPL_CFG.get("recommendations", {})


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class FlaggedSpan:
    """A token flagged for low confidence (prob < mean × threshold).

    Fields follow the Phase B spec exactly:
      token       — the decoded token string
      position    — zero-based token index in the generated sequence
      probability — the model's softmax probability for this token
      reason      — always "low_confidence" (future: extensible)
    """
    token: str
    position: int
    probability: float
    reason: str = "low_confidence"


@dataclass
class ContradictingSentence:
    """A sentence from the local response that contradicts the oracle response.

    Fields follow the Phase B spec:
      text             — the sentence text
      nli_score        — DeBERTa contradiction probability (0–1)
      label            — NLI label, e.g. "CONTRADICTION"
      entailment_score — DeBERTa entailment probability (kept for diagnostics)
    """
    text: str
    nli_score: float
    label: str = "CONTRADICTION"
    entailment_score: float = 0.0


@dataclass
class ExplanationResult:
    """Bundle of all Phase B explanation outputs."""
    flagged_spans: List[FlaggedSpan] = field(default_factory=list)
    contradicting_sentences: List[ContradictingSentence] = field(default_factory=list)
    signal_pct: Dict[str, float] = field(default_factory=dict)
    highlighted_html: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable representation. Safe to pass to json.dumps()."""
        return {
            "flagged_spans": [asdict(s) for s in self.flagged_spans],
            "contradicting_sentences": [asdict(s) for s in self.contradicting_sentences],
            "signal_pct": self.signal_pct,
            "highlighted_html": self.highlighted_html,
            "recommendations": self.recommendations,
        }


# ── Color Palette ─────────────────────────────────────────────────────────────

# Red: P < mean × 0.50 (critical — flagged span)
COLOR_RED_BG    = "#FCEBEB"
COLOR_RED_TEXT  = "#A32D2D"

# Orange/Yellow: P between mean × 0.50 and mean × 0.75 (mild uncertainty)
COLOR_ORANGE_BG   = "#FAEEDA"
COLOR_ORANGE_TEXT = "#854F0B"

# Green: P >= mean × 0.75 and above overall mean (confident)
COLOR_GREEN_BG   = "#EAF3DE"
COLOR_GREEN_TEXT = "#27500A"

# Neutral: between mild and confident thresholds
COLOR_NEUTRAL_BG   = "#F5F5F5"
COLOR_NEUTRAL_TEXT = "#555555"


# ── 1. Token Attribution ──────────────────────────────────────────────────────

def get_flagged_spans(
    response: str,
    token_probs: List[float],
    tokenizer=None,
) -> List[FlaggedSpan]:
    """Extract tokens with prob < mean × low_confidence_threshold as FlaggedSpans.

    Args:
        response:    The model's text response (used for tokenisation if no tokenizer).
        token_probs: Per-token softmax probabilities from calibration.py.
        tokenizer:   Optional HuggingFace tokenizer for accurate token alignment.

    Returns:
        List of FlaggedSpan, one per flagged token, in sequence order.
        Returns empty list if token_probs is empty or None.
    """
    if not token_probs:
        logger.warning("get_flagged_spans: token_probs is empty — returning no spans")
        return []

    threshold_mult = _TOKEN_CFG.get("low_confidence_threshold", 0.5)

    tokens = _tokenize(response, tokenizer)
    n = min(len(tokens), len(token_probs))
    probs = token_probs[:n]
    tokens = tokens[:n]

    mean_prob = float(np.mean(probs)) if probs else 0.5
    cutoff = mean_prob * threshold_mult

    spans: List[FlaggedSpan] = []
    for i, (tok, prob) in enumerate(zip(tokens, probs)):
        if prob < cutoff:
            spans.append(FlaggedSpan(
                token=tok,
                position=i,
                probability=round(prob, 4),
                reason="low_confidence",
            ))
    return spans


def render_highlighted_response(
    response: str,
    token_probs: List[float],
    tokenizer=None,
) -> str:
    """Generate HTML with each token color-coded by confidence relative to mean.

    Color bands (relative to mean probability):
      Red    — prob < mean × low_confidence_threshold   (flagged, critical)
      Yellow — prob between mean × low_confidence_threshold and mean × mild_threshold
      Green  — prob >= mean (above-average confidence)
      Neutral — between mild_threshold and mean

    Args:
        response:    The model's text response.
        token_probs: Per-token softmax probabilities from calibration.py.
        tokenizer:   Optional HuggingFace tokenizer for accurate token alignment.

    Returns:
        HTML string with <span> elements colored by confidence.
        Falls back to plain <p> on empty token_probs.
    """
    if not token_probs:
        return f"<p>{_escape_html(response)}</p>"

    low_mult  = _TOKEN_CFG.get("low_confidence_threshold", 0.5)
    mild_mult = _TOKEN_CFG.get("mild_threshold", 0.75)

    tokens = _tokenize(response, tokenizer)
    n = min(len(tokens), len(token_probs))
    tokens = tokens[:n]
    probs  = token_probs[:n]

    mean_prob = float(np.mean(probs)) if probs else 0.5

    html_parts = []
    for token, prob in zip(tokens, probs):
        bg, fg = _get_color_for_prob(prob, mean_prob, low_mult, mild_mult)
        escaped = _escape_html(token)
        html_parts.append(
            f'<span style="background-color:{bg}; color:{fg}; '
            f'padding:1px 3px; border-radius:3px; margin:1px; '
            f'display:inline-block; font-family:monospace; font-size:0.9em;" '
            f'title="P={prob:.3f}">{escaped}</span>'
        )

    return f'<div style="line-height:1.8; padding:8px;">{" ".join(html_parts)}</div>'


# ── 2. Sentence-Level NLI Scoring ────────────────────────────────────────────

def find_contradicting_sentences(
    local_response: str,
    groq_response: Optional[str],
) -> List[ContradictingSentence]:
    """Split the local response into sentences and run NLI against the Groq oracle.

    Uses the DeBERTa NLI model already loaded by cross_check.py (via
    ``nli_score_sync`` — no second model load).

    Args:
        local_response: TinyLlama's generated response text.
        groq_response:  Groq oracle response text. If None (degraded mode),
                        returns [] immediately and logs a warning.

    Returns:
        List of ContradictingSentence for sentences where contradiction
        probability exceeds the configured threshold. Empty list on any error.
    """
    if not local_response:
        return []
    if not groq_response:
        logger.warning(
            "find_contradicting_sentences: groq_response is None "
            "(cross-check ran in degraded mode) — skipping sentence NLI"
        )
        return []

    if not _NLI_CFG.get("enabled", True):
        return []

    contradiction_threshold = _NLI_CFG.get("contradiction_threshold", 0.50)
    contradiction_label     = _NLI_CFG.get("contradiction_label", "CONTRADICTION")

    sentences = _split_sentences(local_response)
    if not sentences:
        return []

    try:
        from core.cross_check import nli_score_sync  # reuses already-loaded DeBERTa
    except ImportError:
        logger.error("find_contradicting_sentences: could not import nli_score_sync")
        return []

    contradictions: List[ContradictingSentence] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # skip very short fragments
            continue
        try:
            scores = nli_score_sync(sentence, groq_response)
            contradiction = scores.get("contradiction", 0.0)
            entailment    = scores.get("entailment",    0.0)

            if contradiction > contradiction_threshold:
                contradictions.append(ContradictingSentence(
                    text=sentence,
                    nli_score=round(contradiction, 4),
                    label=contradiction_label,
                    entailment_score=round(entailment, 4),
                ))
        except Exception as exc:
            logger.warning(
                f"NLI scoring failed for sentence '{sentence[:50]}…': {exc}"
            )

    return contradictions


# ── 3. Signal SHAP (Ablation-Based) ──────────────────────────────────────────

def compute_signal_shap(
    cal: float,
    unc: float,
    cc: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Lightweight ablation — each signal's marginal % contribution to final score.

    Algorithm (intentionally NOT full Shapley values — ablation approximation):
      For each signal i:
        score_without_i = re-run weighted sum with weight_i set to 0,
                          remaining weights renormalised.
        contribution_i  = |final_score - score_without_i|
      Normalise contributions to sum to 100%.

    This is pure arithmetic on existing scores — no extra model calls.

    Args:
        cal:     Calibration score [0, 1].
        unc:     Semantic uncertainty score [0, 1].
        cc:      Cross-check score [0, 1].
        weights: {"calibration": w1, "semantic_uncertainty": w2, "cross_check": w3}.
                 Defaults to values from config.yaml detection.weights.

    Returns:
        {"calibration": float, "semantic_uncertainty": float, "cross_check": float}
        Values are percentages that sum to 100.0 (±0.1 rounding).
    """
    if not _SHAP_CFG.get("enabled", True):
        return {"calibration": 33.3, "semantic_uncertainty": 33.3, "cross_check": 33.4}

    default_w = _CONFIG.get("detection", {}).get("weights", {})
    w = weights or {
        "calibration":        default_w.get("calibration",        0.20),
        "semantic_uncertainty": default_w.get("semantic_uncertainty", 0.50),
        "cross_check":        default_w.get("cross_check",        0.30),
    }
    w1 = w.get("calibration",        0.20)
    w2 = w.get("semantic_uncertainty", 0.50)
    w3 = w.get("cross_check",        0.30)

    # Full weighted score
    full_score = w1 * cal + w2 * unc + w3 * cc

    # Ablate each signal: set its weight to 0, renormalise others, recompute
    def _score_without(skip: str) -> float:
        sw = {k: (0.0 if k == skip else v) for k, v in w.items()}
        total = sum(sw.values())
        if total == 0:
            return 0.5
        sw = {k: v / total for k, v in sw.items()}
        return (sw["calibration"] * cal
                + sw["semantic_uncertainty"] * unc
                + sw["cross_check"] * cc)

    delta_cal = abs(full_score - _score_without("calibration"))
    delta_unc = abs(full_score - _score_without("semantic_uncertainty"))
    delta_cc  = abs(full_score - _score_without("cross_check"))

    total_delta = delta_cal + delta_unc + delta_cc

    if total_delta == 0:
        # All signals at same level — equal split
        return {"calibration": 33.3, "semantic_uncertainty": 33.3, "cross_check": 33.4}

    return {
        "calibration":          round(100 * delta_cal / total_delta, 1),
        "semantic_uncertainty": round(100 * delta_unc / total_delta, 1),
        "cross_check":          round(100 * delta_cc  / total_delta, 1),
    }


# ── Full Explanation Generator ────────────────────────────────────────────────

def generate_explanation(
    response: str,
    token_probs: List[float],
    local_response: str,
    groq_response: Optional[str],
    cal: float,
    unc: float,
    cc: float,
    weights: Optional[Dict[str, float]] = None,
    tokenizer=None,
) -> ExplanationResult:
    """Generate a complete ExplanationResult by running all three sub-modules.

    Gated by ``explanation.enabled`` in config.yaml. Individual sub-modules
    are further gated by their own ``enabled`` flags.

    Args:
        response:      The model's text response (for token highlighting).
        token_probs:   Per-token softmax probabilities from calibration.py.
        local_response: TinyLlama's response (same as ``response`` in normal flow).
        groq_response: Groq oracle response; None when cross-check ran in degraded mode.
        cal:           Calibration score [0, 1].
        unc:           Semantic uncertainty score [0, 1].
        cc:            Cross-check score [0, 1].
        weights:       Optional weight dict to override config defaults.
        tokenizer:     Optional HuggingFace tokenizer for accurate token alignment.

    Returns:
        ExplanationResult with all sub-module outputs populated.
        Sub-modules that are disabled or fail return empty lists / empty dict.
    """
    if not _EXPL_CFG.get("enabled", True):
        logger.info("generate_explanation: explanation disabled in config — returning empty result")
        return ExplanationResult()

    # 1. Token attribution
    if _TOKEN_CFG.get("enabled", True):
        highlighted_html = render_highlighted_response(response, token_probs, tokenizer)
        flagged = get_flagged_spans(response, token_probs, tokenizer)
    else:
        highlighted_html = ""
        flagged = []

    # 2. Sentence-level NLI
    if _NLI_CFG.get("enabled", True):
        try:
            contradictions = find_contradicting_sentences(local_response, groq_response)
        except Exception as exc:
            logger.error(f"generate_explanation: sentence NLI failed: {exc}")
            contradictions = []
    else:
        contradictions = []

    # 3. Signal SHAP
    signal_pct = compute_signal_shap(cal, unc, cc, weights)

    # 4. Recommendations
    recommendations = _generate_recommendations(
        flagged, contradictions, signal_pct, groq_response
    )

    return ExplanationResult(
        flagged_spans=flagged,
        contradicting_sentences=contradictions,
        signal_pct=signal_pct,
        highlighted_html=highlighted_html,
        recommendations=recommendations,
    )


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _tokenize(response: str, tokenizer) -> List[str]:
    """Split response into tokens using tokenizer if available, else whitespace."""
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(response, add_special_tokens=False)
            return [tokenizer.decode([tid]) for tid in token_ids]
        except Exception:
            pass
    return response.split()


def _get_color_for_prob(
    prob: float,
    mean_prob: float,
    low_mult: float,
    mild_mult: float,
) -> tuple:
    """Return (bg_color, text_color) based on prob relative to mean.

    Bands:
      prob < mean * low_mult              → red   (critical / flagged)
      mean * low_mult <= prob < mean * mild_mult → yellow (mild)
      mean * mild_mult <= prob < mean     → neutral
      prob >= mean                        → green  (confident)
    """
    low_cutoff  = mean_prob * low_mult
    mild_cutoff = mean_prob * mild_mult

    if prob < low_cutoff:
        return COLOR_RED_BG, COLOR_RED_TEXT
    elif prob < mild_cutoff:
        return COLOR_ORANGE_BG, COLOR_ORANGE_TEXT
    elif prob < mean_prob:
        return COLOR_NEUTRAL_BG, COLOR_NEUTRAL_TEXT
    else:
        return COLOR_GREEN_BG, COLOR_GREEN_TEXT


def _escape_html(text: str) -> str:
    """Minimal HTML escaping to prevent XSS in token rendering."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using nltk.sent_tokenize (preferred) with regex fallback.

    Bootstraps the 'punkt_tab' / 'punkt' NLTK data on first call if missing.
    Falls back to a simple regex split if NLTK is unavailable.
    """
    try:
        import nltk  # lazy import — do not load at module level

        # Ensure punkt tokenizer data is available (silent on subsequent calls)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                nltk.download("punkt", quiet=True)

        sentences = nltk.sent_tokenize(text.strip())
        return [s for s in sentences if s.strip()]
    except Exception as exc:
        logger.warning(f"_split_sentences: nltk unavailable ({exc}), using regex fallback")
        import re
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in parts if s.strip()]


def _generate_recommendations(
    flagged: List[FlaggedSpan],
    contradictions: List[ContradictingSentence],
    signal_pct: Dict[str, float],
    groq_response: Optional[str],
) -> List[str]:
    """Generate plain-English recommendation strings from explanation outputs.

    Thresholds read from config.yaml explanation.recommendations.*
    Spec-required recommendation strings are used verbatim.

    Args:
        flagged:        Flagged token spans from token attribution.
        contradictions: Contradicting sentences from sentence NLI.
        signal_pct:     Signal contribution percentages from SHAP ablation.
        groq_response:  None when Groq ran in degraded mode — noted in recommendations.

    Returns:
        List of recommendation strings. Always ends with the primary source reminder.
    """
    cal_threshold = _REC_CFG.get("calibration_dominant_threshold", 0.40)
    sem_threshold = _REC_CFG.get("semantic_dominant_threshold",     0.50)

    recs: List[str] = []

    cal_pct = signal_pct.get("calibration",          0.0)
    sem_pct = signal_pct.get("semantic_uncertainty",  0.0)

    # Calibration dominant
    if cal_pct > (cal_threshold * 100):
        recs.append(
            "Model showed low token confidence — it may be guessing on this topic."
        )

    # Sentence-level contradictions
    n_contra = len(contradictions)
    if n_contra > 0:
        recs.append(
            f"The response contradicts a larger oracle model on {n_contra} sentence(s)."
        )
    elif groq_response is None:
        # Groq degraded — note the omission
        recs.append(
            "Cross-model sentence comparison was skipped (Groq oracle unavailable)."
        )

    # Semantic uncertainty dominant
    if sem_pct > (sem_threshold * 100):
        recs.append(
            "Responses to this question were semantically inconsistent across multiple samples."
        )

    # Always append
    recs.append("Consider verifying with a primary source.")

    return recs
