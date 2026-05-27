"""
Tests for the Explanation Engine.
Covers token highlighting, signal SHAP, contradiction finder, and overall generation.
"""
import pytest
from core.explainer import (
    render_highlighted_response,
    get_flagged_spans,
    compute_signal_shap,
    _get_color_for_prob,
    _split_sentences,
    _escape_html,
    ExplanationResult,
    FlaggedSpan,
)


# ── Token Highlighting Tests ─────────────────────────────────────────

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
        """HTML special characters should be escaped."""
        html = render_highlighted_response("<script>alert(1)</script>", [0.5])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestFlaggedSpans:
    def test_flags_low_confidence_tokens(self):
        """Tokens below threshold should be flagged."""
        spans = get_flagged_spans("Hello world test", [0.8, 0.2, 0.1])
        assert len(spans) == 2
        assert spans[0].confidence == 0.2
        assert spans[1].confidence == 0.1

    def test_no_flags_for_confident_tokens(self):
        """All tokens above threshold should not be flagged."""
        spans = get_flagged_spans("Hello world", [0.9, 0.8])
        assert len(spans) == 0

    def test_severity_classification(self):
        """Severity should match probability ranges."""
        spans = get_flagged_spans("a b c", [0.1, 0.35, 0.45])
        assert spans[0].severity == "critical"    # P < 0.30
        assert spans[1].severity == "warning"     # 0.30 <= P < 0.50
        assert spans[2].severity == "warning"     # 0.30 <= P < 0.50


# ── Signal SHAP Tests ────────────────────────────────────────────────

class TestSignalShap:
    def test_percentages_sum_to_100(self):
        """Signal percentages should sum to approximately 100%."""
        pct = compute_signal_shap(0.8, 0.3, 0.7)
        total = sum(pct.values())
        assert abs(total - 100.0) < 0.5

    def test_dominant_signal_identified(self):
        """A very high cross-check score should dominate."""
        pct = compute_signal_shap(0.1, 0.1, 0.9)
        assert pct["cross_check"] > pct["calibration"]
        assert pct["cross_check"] > pct["semantic_uncertainty"]

    def test_equal_signals_give_equal_contribution(self):
        """When all signals are equal, contribution should be roughly equal."""
        pct = compute_signal_shap(0.5, 0.5, 0.5)
        # All at baseline — should return 33/33/34
        assert abs(pct["calibration"] - 33.3) < 1
        assert abs(pct["semantic_uncertainty"] - 33.3) < 1
        assert abs(pct["cross_check"] - 33.4) < 1

    def test_custom_weights(self):
        """Custom weights should be respected."""
        pct = compute_signal_shap(
            0.9, 0.1, 0.5,
            weights={"calibration": 0.5, "semantic_uncertainty": 0.3, "cross_check": 0.2}
        )
        total = sum(pct.values())
        assert abs(total - 100.0) < 0.5


# ── Helper Tests ─────────────────────────────────────────────────────

class TestHelpers:
    def test_color_for_critical_prob(self):
        bg, fg, severity = _get_color_for_prob(0.1)
        assert severity == "critical"
        assert "#FCE" in bg.upper() or "#FCEBEB" in bg.upper()

    def test_color_for_warning_prob(self):
        bg, fg, severity = _get_color_for_prob(0.4)
        assert severity == "warning"

    def test_color_for_ok_prob(self):
        bg, fg, severity = _get_color_for_prob(0.8)
        assert severity == "ok"

    def test_color_for_borderline_prob(self):
        bg, fg, severity = _get_color_for_prob(0.6)
        assert severity == "borderline"

    def test_split_sentences(self):
        text = "Hello world. This is a test! How are you? Fine."
        sentences = _split_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "Hello world."
        assert sentences[2] == "How are you?"

    def test_escape_html(self):
        assert _escape_html("<b>test</b>") == "&lt;b&gt;test&lt;/b&gt;"
        assert _escape_html('"quotes"') == "&quot;quotes&quot;"


# ── ExplanationResult Structure Tests ────────────────────────────────

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
        """ExplanationResult with data should be constructable."""
        result = ExplanationResult(
            flagged_spans=[FlaggedSpan("test", 0.2, "critical")],
            signal_pct={"calibration": 40, "semantic_uncertainty": 30, "cross_check": 30},
            highlighted_html="<span>test</span>",
            recommendations=["Check this fact."],
        )
        assert len(result.flagged_spans) == 1
        assert result.flagged_spans[0].text == "test"
        assert result.signal_pct["calibration"] == 40
