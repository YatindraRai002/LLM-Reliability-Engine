"""
Tests for the evaluation harness modules.
Covers AUROC computation, per-category grouping, checkpoint resume logic,
and the eval report generator.
"""
import pytest
import json
import os
import tempfile
import numpy as np
from evaluation.truthfulqa_eval import (
    load_checkpoint,
    append_checkpoint,
    print_metrics,
    _print_basic_stats,
)
from evaluation.tune_weights import tune_weights
from evaluation.eval_report import generate_report


class TestCheckpointResume:
    def test_load_empty_checkpoint(self, tmp_path):
        """Loading a non-existent checkpoint returns empty list."""
        path = str(tmp_path / "nonexistent.jsonl")
        results = load_checkpoint(path)
        assert results == []

    def test_append_and_load_checkpoint(self, tmp_path):
        """Appending results and loading them back works correctly."""
        path = str(tmp_path / "checkpoint.jsonl")
        result1 = {"question": "Q1", "score": 0.5}
        result2 = {"question": "Q2", "score": 0.8}

        append_checkpoint(path, result1)
        append_checkpoint(path, result2)

        loaded = load_checkpoint(path)
        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1"
        assert loaded[1]["question"] == "Q2"

    def test_load_corrupted_checkpoint(self, tmp_path):
        """Corrupted lines in checkpoint are silently skipped."""
        path = str(tmp_path / "corrupt.jsonl")
        with open(path, "w") as f:
            f.write('{"question": "Q1", "score": 0.5}\n')
            f.write('INVALID JSON LINE\n')
            f.write('{"question": "Q2", "score": 0.8}\n')

        loaded = load_checkpoint(path)
        assert len(loaded) == 2


class TestMetrics:
    def test_print_metrics_with_valid_data(self, capsys):
        """Metrics print without errors when data has both classes."""
        results = [
            {"hallucination_score": 0.9, "correctness": False, "risk_label": "high",
             "category": "Health", "elapsed_seconds": 5.0},
            {"hallucination_score": 0.1, "correctness": True, "risk_label": "low",
             "category": "Health", "elapsed_seconds": 3.0},
            {"hallucination_score": 0.8, "correctness": False, "risk_label": "high",
             "category": "Law", "elapsed_seconds": 4.0},
            {"hallucination_score": 0.2, "correctness": True, "risk_label": "low",
             "category": "Law", "elapsed_seconds": 2.5},
        ]
        print_metrics(results)
        captured = capsys.readouterr()
        assert "AUROC" in captured.out

    def test_print_metrics_single_class(self, capsys):
        """Metrics gracefully handle single-class labels."""
        results = [
            {"hallucination_score": 0.5, "correctness": True, "risk_label": "medium",
             "category": "Health", "elapsed_seconds": 3.0},
            {"hallucination_score": 0.3, "correctness": True, "risk_label": "low",
             "category": "Health", "elapsed_seconds": 2.0},
        ]
        print_metrics(results)
        captured = capsys.readouterr()
        assert "one class" in captured.out.lower() or "EVALUATION" in captured.out

    def test_print_metrics_empty(self, capsys):
        """Metrics handle empty results gracefully."""
        print_metrics([])

    def test_basic_stats_fallback(self, capsys):
        """Basic stats work when AUROC cannot be computed."""
        results = [
            {"risk_label": "high", "elapsed_seconds": 5.0},
            {"risk_label": "low", "elapsed_seconds": 3.0},
        ]
        _print_basic_stats(results)
        captured = capsys.readouterr()
        assert "High Risk Flagged" in captured.out


class TestEvalReport:
    def test_generate_report_with_data(self, tmp_path):
        """Report generates correctly from valid evaluation results."""
        results = [
            {"hallucination_score": 0.9, "correctness": False, "risk_label": "high",
             "category": "Conspiracies", "elapsed_seconds": 5.0},
            {"hallucination_score": 0.1, "correctness": True, "risk_label": "low",
             "category": "Health", "elapsed_seconds": 3.0},
            {"hallucination_score": 0.7, "correctness": False, "risk_label": "high",
             "category": "Conspiracies", "elapsed_seconds": 4.0},
            {"hallucination_score": 0.2, "correctness": True, "risk_label": "low",
             "category": "Health", "elapsed_seconds": 2.5},
        ]

        input_path = str(tmp_path / "results.json")
        output_path = str(tmp_path / "report.json")

        with open(input_path, "w") as f:
            json.dump(results, f)

        report = generate_report(input_path, output_path)

        assert report["metadata"]["total_samples"] == 4
        assert report["metadata"]["labeled_samples"] == 4
        assert report["overall_metrics"]["auroc"] is not None
        assert 0 <= report["overall_metrics"]["auroc"] <= 1
        assert report["risk_distribution"]["high"] == 2
        assert report["risk_distribution"]["low"] == 2
        assert "Conspiracies" in report["per_category"]
        assert "Health" in report["per_category"]
        assert report["latency"]["p50"] > 0

    def test_generate_report_missing_file(self, tmp_path):
        """Report handles missing input file gracefully."""
        report = generate_report(
            str(tmp_path / "nonexistent.json"),
            str(tmp_path / "report.json"),
        )
        assert report == {}


class TestTuneWeights:
    def test_tune_with_valid_data(self, tmp_path):
        """Weight tuner finds valid weights from labeled results."""
        results = [
            {"calibration": 0.8, "uncertainty": 0.7, "cross_check": 0.9,
             "correctness": False, "hallucination_score": 0.8},
            {"calibration": 0.1, "uncertainty": 0.2, "cross_check": 0.1,
             "correctness": True, "hallucination_score": 0.1},
            {"calibration": 0.7, "uncertainty": 0.8, "cross_check": 0.85,
             "correctness": False, "hallucination_score": 0.75},
            {"calibration": 0.2, "uncertainty": 0.1, "cross_check": 0.15,
             "correctness": True, "hallucination_score": 0.15},
            {"calibration": 0.6, "uncertainty": 0.5, "cross_check": 0.7,
             "correctness": False, "hallucination_score": 0.6},
        ]

        input_path = str(tmp_path / "results.json")
        with open(input_path, "w") as f:
            json.dump(results, f)

        weights = tune_weights(input_path, write_back=False)
        assert weights is not None
        assert "calibration" in weights
        assert "semantic_uncertainty" in weights
        assert "cross_check" in weights
        total = weights["calibration"] + weights["semantic_uncertainty"] + weights["cross_check"]
        assert abs(total - 1.0) < 0.01

    def test_tune_missing_file(self, tmp_path):
        """Tuner handles missing results file gracefully."""
        weights = tune_weights(str(tmp_path / "nope.json"), write_back=False)
        assert weights is None

    def test_tune_insufficient_labels(self, tmp_path):
        """Tuner returns None when too few labeled samples."""
        results = [
            {"calibration": 0.5, "uncertainty": 0.5, "cross_check": 0.5,
             "correctness": True, "hallucination_score": 0.5},
        ]
        input_path = str(tmp_path / "results.json")
        with open(input_path, "w") as f:
            json.dump(results, f)

        weights = tune_weights(input_path, write_back=False)
        assert weights is None
