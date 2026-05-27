"""
Explanation Engine for the LLM Lie Detector.
Provides three explanation capabilities:
1. Token-level confidence highlighting (HTML rendering)
2. Sentence-level contradiction detection (NLI-based)
3. Signal contribution analysis (SHAP-style ablation)
"""
import re
import logging
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class FlaggedSpan:
    """A token or span that was flagged for low confidence."""
    text: str
    confidence: float
    severity: str  # "critical", "warning", "ok"


@dataclass
class ContradictingSentence:
    """A sentence flagged as contradicting the reference model."""
    sentence: str
    contradiction_score: float
    entailment_score: float


@dataclass
class ExplanationResult:
    """Bundle of all explanation outputs."""
    flagged_spans: list[FlaggedSpan] = field(default_factory=list)
    contradicting_sentences: list[ContradictingSentence] = field(default_factory=list)
    signal_pct: dict = field(default_factory=dict)  # {"calibration": 0.4, ...}
    highlighted_html: str = ""
    recommendations: list[str] = field(default_factory=list)


# ── Color Palette ────────────────────────────────────────────────────

# Red: P < 0.30 (critical uncertainty)
COLOR_RED_BG = "#FCEBEB"
COLOR_RED_TEXT = "#A32D2D"

# Orange: 0.30 <= P < 0.50 (moderate uncertainty)
COLOR_ORANGE_BG = "#FAEEDA"
COLOR_ORANGE_TEXT = "#854F0B"

# Green: P >= 0.70 (confident)
COLOR_GREEN_BG = "#EAF3DE"
COLOR_GREEN_TEXT = "#27500A"

# Neutral: 0.50 <= P < 0.70 (borderline)
COLOR_NEUTRAL_BG = "#F5F5F5"
COLOR_NEUTRAL_TEXT = "#555555"


# ── 1. Token-Level Confidence Highlighting ───────────────────────────

def render_highlighted_response(
    response: str,
    token_probs: list[float],
    tokenizer=None,
) -> str:
    """
    Generate HTML with each token color-coded by confidence.

    Args:
        response: The model's text response.
        token_probs: List of per-token probabilities from logit extraction.
        tokenizer: Optional tokenizer to re-tokenize and align tokens to probs.

    Returns:
        HTML string with <span> elements colored by confidence.
    """
    if not token_probs:
        return f"<p>{_escape_html(response)}</p>"

    # If we have a tokenizer, re-tokenize to align token text with probs
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(response, add_special_tokens=False)
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
        except Exception:
            tokens = response.split()
    else:
        tokens = response.split()

    # Align: trim to min of tokens and probs
    n = min(len(tokens), len(token_probs))
    tokens = tokens[:n]
    probs = token_probs[:n]

    html_parts = []
    for token, prob in zip(tokens, probs):
        bg, fg, severity = _get_color_for_prob(prob)
        escaped = _escape_html(token)
        html_parts.append(
            f'<span style="background-color:{bg}; color:{fg}; '
            f'padding:1px 3px; border-radius:3px; margin:1px; '
            f'display:inline-block; font-family:monospace; font-size:0.9em;" '
            f'title="P={prob:.3f}">{escaped}</span>'
        )

    return f'<div style="line-height:1.8; padding:8px;">{" ".join(html_parts)}</div>'


def get_flagged_spans(
    response: str,
    token_probs: list[float],
    tokenizer=None,
    threshold: float = 0.5,
) -> list[FlaggedSpan]:
    """
    Extract tokens with confidence below threshold as flagged spans.
    """
    if not token_probs:
        return []

    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(response, add_special_tokens=False)
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
        except Exception:
            tokens = response.split()
    else:
        tokens = response.split()

    n = min(len(tokens), len(token_probs))
    spans = []
    for i in range(n):
        prob = token_probs[i]
        if prob < threshold:
            _, _, severity = _get_color_for_prob(prob)
            spans.append(FlaggedSpan(
                text=tokens[i],
                confidence=prob,
                severity=severity,
            ))
    return spans


# ── 2. Sentence-Level Contradiction Detection ───────────────────────

def find_contradicting_sentences(
    local_response: str,
    groq_response: str,
    contradiction_threshold: float = 0.50,
) -> list[ContradictingSentence]:
    """
    Split the local response into sentences, run NLI scoring on each
    sentence against the Groq response, and return sentences where
    contradiction > threshold.

    Uses the synchronous NLI wrapper from cross_check.py.
    """
    if not local_response or not groq_response:
        return []

    sentences = _split_sentences(local_response)
    if not sentences:
        return []

    try:
        from core.cross_check import nli_score_sync
    except ImportError:
        logger.error("Could not import nli_score_sync from core.cross_check")
        return []

    contradictions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short fragments
            continue

        try:
            scores = nli_score_sync(sentence, groq_response)
            contradiction = scores.get("contradiction", 0)
            entailment = scores.get("entailment", 0)

            if contradiction > contradiction_threshold:
                contradictions.append(ContradictingSentence(
                    sentence=sentence,
                    contradiction_score=round(contradiction, 3),
                    entailment_score=round(entailment, 3),
                ))
        except Exception as e:
            logger.warning(f"NLI scoring failed for sentence: {sentence[:50]}... — {e}")

    return contradictions


# ── 3. Signal Contribution Analysis (SHAP-style) ────────────────────

def compute_signal_shap(
    cal: float,
    unc: float,
    cc: float,
    weights: dict = None,
) -> dict:
    """
    Signal ablation: sets each signal to a 0.5 baseline one at a time,
    measures delta from full score, and returns percentage contribution
    of each signal.

    Args:
        cal: Calibration score [0, 1]
        unc: Semantic uncertainty score [0, 1]
        cc: Cross-check score [0, 1]
        weights: Dict with keys "calibration", "semantic_uncertainty", "cross_check"

    Returns:
        Dict with percentage contribution of each signal.
        e.g. {"calibration": 25.3, "semantic_uncertainty": 30.1, "cross_check": 44.6}
    """
    w = weights or {"calibration": 0.25, "semantic_uncertainty": 0.30, "cross_check": 0.45}
    w1 = w.get("calibration", 0.25)
    w2 = w.get("semantic_uncertainty", 0.30)
    w3 = w.get("cross_check", 0.45)

    # Full score (weighted average)
    full_score = w1 * cal + w2 * unc + w3 * cc

    # Baseline value: 0.5 (neutral)
    baseline = 0.5

    # Ablated scores — replace each signal with baseline
    ablated_cal = w1 * baseline + w2 * unc + w3 * cc
    ablated_unc = w1 * cal + w2 * baseline + w3 * cc
    ablated_cc = w1 * cal + w2 * unc + w3 * baseline

    # Delta = how much the score changes when this signal is neutralized
    delta_cal = abs(full_score - ablated_cal)
    delta_unc = abs(full_score - ablated_unc)
    delta_cc = abs(full_score - ablated_cc)

    total_delta = delta_cal + delta_unc + delta_cc

    if total_delta == 0:
        # All signals are at baseline
        return {"calibration": 33.3, "semantic_uncertainty": 33.3, "cross_check": 33.4}

    return {
        "calibration": round(100 * delta_cal / total_delta, 1),
        "semantic_uncertainty": round(100 * delta_unc / total_delta, 1),
        "cross_check": round(100 * delta_cc / total_delta, 1),
    }


# ── Full Explanation Generator ───────────────────────────────────────

def generate_explanation(
    response: str,
    token_probs: list[float],
    local_response: str,
    groq_response: str,
    cal: float,
    unc: float,
    cc: float,
    weights: dict = None,
    tokenizer=None,
) -> ExplanationResult:
    """
    Generate a complete ExplanationResult by running all three
    explanation strategies.
    """
    # 1. Token highlighting
    highlighted_html = render_highlighted_response(response, token_probs, tokenizer)
    flagged = get_flagged_spans(response, token_probs, tokenizer)

    # 2. Contradiction detection
    contradictions = find_contradicting_sentences(local_response, groq_response)

    # 3. Signal SHAP
    signal_pct = compute_signal_shap(cal, unc, cc, weights)

    # 4. Generate recommendations
    recommendations = _generate_recommendations(
        flagged, contradictions, signal_pct, cal, unc, cc
    )

    return ExplanationResult(
        flagged_spans=flagged,
        contradicting_sentences=contradictions,
        signal_pct=signal_pct,
        highlighted_html=highlighted_html,
        recommendations=recommendations,
    )


# ── Internal Helpers ─────────────────────────────────────────────────

def _get_color_for_prob(prob: float) -> tuple[str, str, str]:
    """Return (bg_color, text_color, severity) for a probability value."""
    if prob < 0.30:
        return COLOR_RED_BG, COLOR_RED_TEXT, "critical"
    elif prob < 0.50:
        return COLOR_ORANGE_BG, COLOR_ORANGE_TEXT, "warning"
    elif prob >= 0.70:
        return COLOR_GREEN_BG, COLOR_GREEN_TEXT, "ok"
    else:
        return COLOR_NEUTRAL_BG, COLOR_NEUTRAL_TEXT, "borderline"


def _escape_html(text: str) -> str:
    """Minimal HTML escaping."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on ., !, ? followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _generate_recommendations(
    flagged: list[FlaggedSpan],
    contradictions: list[ContradictingSentence],
    signal_pct: dict,
    cal: float,
    unc: float,
    cc: float,
) -> list[str]:
    """Generate actionable recommendations based on explanation results."""
    recs = []

    critical_spans = [f for f in flagged if f.severity == "critical"]
    if len(critical_spans) > 3:
        recs.append(
            f"⚠️ {len(critical_spans)} tokens have critically low confidence "
            f"(P < 0.30). The model is very unsure about these parts of the response."
        )

    if contradictions:
        recs.append(
            f"🔴 {len(contradictions)} sentence(s) contradict the cross-check model. "
            f"These claims should be verified against authoritative sources."
        )

    # Dominant signal recommendation
    dominant = max(signal_pct, key=signal_pct.get)
    if signal_pct[dominant] > 50:
        signal_names = {
            "calibration": "token-level confidence",
            "semantic_uncertainty": "response consistency",
            "cross_check": "cross-model agreement",
        }
        recs.append(
            f"📊 The dominant uncertainty signal is {signal_names.get(dominant, dominant)} "
            f"({signal_pct[dominant]:.0f}% contribution)."
        )

    if cal > 0.7:
        recs.append("🔻 Token confidence is very low — the model is likely guessing.")
    if unc > 0.7:
        recs.append("🔻 High semantic uncertainty — the model gives different answers each time.")
    if cc > 0.7:
        recs.append("🔻 Strong cross-model disagreement — the two models give conflicting answers.")

    if not recs:
        recs.append("✅ All signals indicate consistent, confident generation.")

    return recs
