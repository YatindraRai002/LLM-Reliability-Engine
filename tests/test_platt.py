"""
Tests for the Platt scaling calibration logic.
"""
import pytest
import os
import numpy as np
from core.calibration import PlattCalibrator, compute_ece

class TestPlattCalibrator:
    def test_unfitted_passthrough(self):
        """Unfitted calibrator should pass through the raw score."""
        calibrator = PlattCalibrator()
        assert calibrator.transform(0.75) == 0.75

    def test_fit_and_transform(self):
        """Fitted calibrator should transform scores using logistic regression."""
        calibrator = PlattCalibrator()
        # Mock data: low scores = mostly 0, high scores = mostly 1
        scores = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
        labels = [0, 0, 1, 1, 1, 1]
        
        calibrator.fit(scores, labels)
        assert calibrator._fitted
        
        t_low = calibrator.transform(0.1)
        t_high = calibrator.transform(0.9)
        
        # High score should yield higher probability of class 1
        assert t_high > t_low
        assert 0 <= t_low <= 1
        assert 0 <= t_high <= 1

    def test_save_and_load(self, tmp_path):
        """Calibrator state should be saveable and loadable."""
        calibrator = PlattCalibrator()
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        calibrator.fit(scores, labels)
        
        path = str(tmp_path / "calibrator.pkl")
        calibrator.save(path)
        
        # Create new calibrator and load
        loaded_calibrator = PlattCalibrator()
        loaded_calibrator.load(path)
        
        assert loaded_calibrator._fitted
        assert loaded_calibrator.transform(0.5) == calibrator.transform(0.5)

    def test_load_missing_file_graceful(self, tmp_path):
        """Loading a missing file should fail gracefully and leave it unfitted."""
        calibrator = PlattCalibrator()
        calibrator.load(str(tmp_path / "missing.pkl"))
        assert not calibrator._fitted
        assert calibrator.transform(0.5) == 0.5

class TestECE:
    def test_compute_ece(self):
        """ECE should correctly measure difference between conf and acc."""
        conf = [0.9, 0.9, 0.9]
        acc = [1.0, 1.0, 0.0] # empirical acc = 2/3 = 0.666...
        
        # Bin 0.9 falls in [0.8, 0.9) or [0.9, 1.0] depending on boundaries
        # For simplicity, mean conf = 0.9, mean acc = 0.666
        # ECE = |0.9 - 0.666| = 0.233
        ece = compute_ece(conf, acc, n_bins=1)
        assert abs(ece - (0.9 - 2/3)) < 1e-4

    def test_ece_empty(self):
        assert compute_ece([], []) == 0.0
