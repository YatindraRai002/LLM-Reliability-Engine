"""
Tests for the Phase B Explanation Engine (core/explainer.py).

Covers:
  - Token attribution: flagging, HTML rendering, HTML escaping
  - Signal SHAP: percentage correctness, dominance, edge cases
  - Sentence NLI: contradiction detection, graceful degradation
  - ExplanationResult: structure, to_dict() JSON serialisability
  - Helper utilities: color mapping, sentence splitting, HTML escaping
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from core.explainer import (
    render_highlighted_response,
    get_flagged_spans,
    compute_signal_shap,
    find_contradicting_sentences,
    generate_explanation,
    _get_color_for_prob,
    _split_sentences,
    _escape_html,
    ExplanationResult,
    FlaggedSpan,
    ContradictingSentence,
)


# ── Token Attribution Tests ───────────────────────────────────────────────────

class TestTokenHighlighting:
    def test_empty_probs_returns_plain(self):
        """Empty token probs should return plain HTML paragraph."""
        html = render_highlighted_response("Hello world", [])
        assert "<p>" in html
        assert "Hello world" in html

    def test_renders_spans_for_tokens(self):
        """Each token should get a colored span."""
        html = render_highlighted_response("The cat sat", [0.9, 0.2, 0.6])
        assert "<span" in html
        assert "P=" in html  # title attribute

    def test_html_escaping(self):
        """HTML special characters should be escaped in rendered output."""
        html = render_highlighted_response("<script>alert(1)</script>", [0.5])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestFlaggedSpans:
    """Phase B spec: FlaggedSpan(token, position, probability, reason)."""

    def test_token_attribution_flags_low_prob_tokens(self):
        """Tokens with prob < mean × 0.5 should be flagged.

        mean([0.8, 0.2, 0.1]) = 0.367
        cutoff = 0.367 × 0.5 = 0.183
        → 0.1 < 0.183, so only token at index 2 is flagged.
        """
        spans = get_flagged_spans("Hello world test", [0.8, 0.2, 0.1])
        assert len(spans) == 1
        assert spans[0].token == "test"
        assert spans[0].position == 2
        assert abs(spans[0].probability - 0.1) < 0.001
        assert spans[0].reason == "low_confidence"

    def test_flags_multiple_low_confidence_tokens(self):
        """Multiple tokens below threshold should all be flagged."""
        # mean([0.9, 0.05, 0.05]) = 0.333, cutoff = 0.167
        # 0.05 < 0.167, so indices 1 and 2 are flagged
        spans = get_flagged_spans("a b c", [0.9, 0.05, 0.05])
        assert len(spans) == 2
        assert spans[0].position == 1
        assert spans[1].position == 2

    def test_no_flags_for_confident_tokens(self):
        """All tokens at similar high probability should not be flagged."""
        # mean([0.9, 0.85]) = 0.875, cutoff = 0.4375
        # both probs are above cutoff
        spans = get_flagged_spans("Hello world", [0.9, 0.85])
        assert len(spans) == 0

    def test_empty_token_probs_returns_empty(self):
        """Empty token_probs should return empty list, not raise."""
        spans = get_flagged_spans("some response", [])
        assert spans == []

    def test_reason_is_always_low_confidence(self):
        """reason field must always be 'low_confidence'."""
        # mean = 0.5, cutoff = 0.25; first token (0.1) is flagged
        spans = get_flagged_spans("a b", [0.1, 0.9])
        assert len(spans) >= 1
        for span in spans:
            assert span.reason == "low_confidence"

    def test_field_names_match_spec(self):
        """FlaggedSpan must use Phase B spec field names."""
        # mean([0.1, 0.9]) = 0.5, cutoff = 0.25
        spans = get_flagged_spans("a b", [0.1, 0.9])
        assert len(spans) >= 1
        span = spans[0]
        assert hasattr(span, "token")
        assert hasattr(span, "position")
        assert hasattr(span, "probability")
        assert hasattr(span, "reason")


# ── Signal SHAP Tests ─────────────────────────────────────────────────────────

class TestSignalShap:
    def test_signal_shap_sums_to_100(self):
        """Signal contribution percentages must sum to exactly 100 (± rounding)."""
        pct = compute_signal_shap(0.8, 0.3, 0.7)
        total = sum(pct.values())
        assert abs(total - 100.0) < 0.5, f"Expected ~100, got {total}"

    def test_dominant_signal_identified(self):
        """A very high cross-check score should have the highest contribution."""
        pct = compute_signal_shap(0.1, 0.1, 0.9)
        assert pct["cross_check"] > pct["calibration"]
        assert pct["cross_check"] > pct["semantic_uncertainty"]

    def test_equal_signals_give_equal_contribution(self):
        """When all signals are at baseline (0.5), contribution should be ~33/33/34."""
        pct = compute_signal_shap(0.5, 0.5, 0.5)
        assert abs(pct["calibration"] - 33.3) < 1.0
        assert abs(pct["semantic_uncertainty"] - 33.3) < 1.0
        assert abs(pct["cross_check"] - 33.4) < 1.0

    def test_custom_weights_respected(self):
        """Custom weights should be used and result should still sum to 100."""
        pct = compute_signal_shap(
            0.9, 0.1, 0.5,
            weights={"calibration": 0.5, "semantic_uncertainty": 0.3, "cross_check": 0.2},
        )
        total = sum(pct.values())
        assert abs(total - 100.0) < 0.5

    def test_all_zero_weights_returns_equal_split(self):
        """When no signal contributes (all at same value), return equal split."""
        pct = compute_signal_shap(0.5, 0.5, 0.5)
        assert len(pct) == 3


# ── Sentence NLI Tests ────────────────────────────────────────────────────────

class TestSentenceNLI:
    def test_sentence_nli_catches_contradiction(self):
        """Mock NLI model returning high contradiction — sentence should be returned."""
        mock_scores = {
            "contradiction": 0.85,
            "entailment":    0.05,
            "neutral":       0.10,
            "verdict":       "contradiction",
            "agreement":     -0.80,
        }
        with patch("core.explainer.find_contradicting_sentences") as mock_fn:
            mock_fn.return_value = [
                ContradictingSentence(
                    text="The earth is flat.",
                    nli_score=0.85,
                    label="CONTRADICTION",
                    entailment_score=0.05,
                )
            ]
            result = mock_fn("The earth is flat.", "The earth is a sphere.")
        assert len(result) == 1
        assert result[0].text == "The earth is flat."
        assert result[0].nli_score > 0.5
        assert result[0].label == "CONTRADICTION"

    def test_graceful_degradation_no_oracle(self):
        """Passing None as oracle response must not raise and must return empty list."""
        result = find_contradicting_sentences("Some local response.", None)
        assert result == [], (
            "Expected empty list when groq_response is None, "
            f"got {result}"
        )

    def test_empty_local_response_returns_empty(self):
        """Empty local response must return empty list without error."""
        result = find_contradicting_sentences("", "Groq response text.")
        assert result == []

    def test_contradicting_sentence_field_names(self):
        """ContradictingSentence must use Phase B spec field names."""
        cs = ContradictingSentence(
            text="Some claim.",
            nli_score=0.9,
            label="CONTRADICTION",
            entailment_score=0.05,
        )
        assert hasattr(cs, "text")
        assert hasattr(cs, "nli_score")
        assert hasattr(cs, "label")
        assert hasattr(cs, "entailment_score")


# ── ExplanationResult Tests ───────────────────────────────────────────────────

class TestExplanationResult:
    def test_default_fields(self):
        """Default ExplanationResult should have empty fields."""
        result = ExplanationResult()
        assert result.flagged_spans == []
        assert result.contradicting_sentences == []
        assert result.signal_pct == {}
        assert result.highlighted_html == ""
        assert result.recommendations == []

    def test_with_data(self):
        """ExplanationResult should be constructable with Phase B spec field names."""
        result = ExplanationResult(
            flagged_spans=[FlaggedSpan(token="test", position=0, probability=0.2)],
            signal_pct={"calibration": 40.0, "semantic_uncertainty": 30.0, "cross_check": 30.0},
            highlighted_html="<span>test</span>",
            recommendations=["Check this fact."],
        )
        assert len(result.flagged_spans) == 1
        assert result.flagged_spans[0].token == "test"
        assert result.flagged_spans[0].position == 0
        assert result.flagged_spans[0].probability == 0.2
        assert result.signal_pct["calibration"] == 40.0

    def test_explanation_result_to_dict_is_json_serializable(self):
        """to_dict() output must be serialisable by json.dumps() without error."""
        result = ExplanationResult(
            flagged_spans=[
                FlaggedSpan(token="guessing", position=3, probability=0.12),
                FlaggedSpan(token="maybe", position=7, probability=0.18, reason="low_confidence"),
            ],
            contradicting_sentences=[
                ContradictingSentence(
                    text="The moon is made of cheese.",
                    nli_score=0.91,
                    label="CONTRADICTION",
                    entailment_score=0.02,
                )
            ],
            signal_pct={
                "calibration": 25.0,
                "semantic_uncertainty": 45.0,
                "cross_check": 30.0,
            },
            highlighted_html="<div>highlighted</div>",
            recommendations=[
                "Model showed low token confidence.",
                "Consider verifying with a primary source.",
            ],
        )
        d = result.to_dict()

        # Must be JSON-serialisable
        try:
            serialised = json.dumps(d)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"to_dict() result is not JSON-serialisable: {exc}")

        # Round-trip check: keys must survive serialisation
        parsed = json.loads(serialised)
        assert "flagged_spans" in parsed
        assert "contradicting_sentences" in parsed
        assert "signal_pct" in parsed
        assert "highlighted_html" in parsed
        assert "recommendations" in parsed

        # Spot-check nested field names match Phase B spec
        assert parsed["flagged_spans"][0]["token"] == "guessing"
        assert parsed["flagged_spans"][0]["position"] == 3
        assert parsed["flagged_spans"][0]["probability"] == 0.12
        assert parsed["contradicting_sentences"][0]["text"] == "The moon is made of cheese."
        assert parsed["contradicting_sentences"][0]["nli_score"] == 0.91
        assert parsed["contradicting_sentences"][0]["label"] == "CONTRADICTION"


# ── Helper Utility Tests ──────────────────────────────────────────────────────

class TestHelpers:
    def test_color_for_critical_prob(self):
        # prob=0.1, mean=0.5, low_mult=0.5 → cutoff=0.25 → red
        bg, fg = _get_color_for_prob(0.1, 0.5, 0.5, 0.75)
        assert "FC" in bg.upper()  # COLOR_RED_BG starts with #FC

    def test_color_for_mild_prob(self):
        # prob=0.3, mean=0.5, cutoffs: low=0.25, mild=0.375 → orange
        bg, fg = _get_color_for_prob(0.3, 0.5, 0.5, 0.75)
        assert "FA" in bg.upper()  # COLOR_ORANGE_BG starts with #FA

    def test_color_for_confident_prob(self):
        # prob=0.8, mean=0.5 → above mean → green
        bg, fg = _get_color_for_prob(0.8, 0.5, 0.5, 0.75)
        assert "EA" in bg.upper()  # COLOR_GREEN_BG starts with #EA

    def test_color_for_neutral_prob(self):
        # prob=0.45, mean=0.5, mild_cutoff=0.375 → between mild and mean → neutral
        bg, fg = _get_color_for_prob(0.45, 0.5, 0.5, 0.75)
        assert "F5" in bg.upper()  # COLOR_NEUTRAL_BG starts with #F5

    def test_split_sentences_basic(self):
        text = "Hello world. This is a test! How are you? Fine."
        sentences = _split_sentences(text)
        assert len(sentences) == 4

    def test_split_sentences_preserves_punctuation(self):
        text = "First sentence. Second sentence."
        sentences = _split_sentences(text)
        assert sentences[0].endswith(".")
        assert sentences[1].endswith(".")

    def test_escape_html(self):
        assert _escape_html("<b>test</b>") == "&lt;b&gt;test&lt;/b&gt;"
        assert _escape_html('"quotes"') == "&quot;quotes&quot;"
        assert _escape_html("a & b") == "a &amp; b"
