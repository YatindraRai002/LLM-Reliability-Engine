import pytest
import numpy as np
from core.calibration import compute_calibration_score, compute_ece

def test_high_confidence_gives_low_uncertainty():
    token_probs = [0.95] * 20
    score = compute_calibration_score(token_probs)
    assert score < 0.2, f"Expected low uncertainty, got {score}"

def test_low_confidence_gives_high_uncertainty():
    token_probs = [0.1] * 20
    score = compute_calibration_score(token_probs)
    assert score >= 0.6, f"Expected high uncertainty, got {score}"

def test_output_is_bounded():
    import random
    for _ in range(50):
        probs = [random.random() for _ in range(15)]
        score = compute_calibration_score(probs)
        assert 0.0 <= score <= 1.0

def test_ece_perfect_calibration():
    confs = np.linspace(0.1, 0.9, 100)
    accs = confs
    ece = compute_ece(confs.tolist(), accs.tolist())
    assert ece < 0.01
