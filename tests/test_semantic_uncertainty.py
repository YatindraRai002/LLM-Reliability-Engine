import pytest
import numpy as np
from core.semantic_uncertainty import compute_semantic_uncertainty

def test_identical_responses_low_uncertainty():
    # All responses are identical -> 1 cluster -> low uncertainty
    responses = ["The capital of France is Paris."] * 10
    result = compute_semantic_uncertainty(responses)
    assert result["n_semantic_clusters"] == 1
    assert result["uncertainty_score"] < 0.1
    # Use approx for floating point comparison
    assert result["normalized_entropy"] == pytest.approx(0.0, abs=1e-9)

def test_diverse_responses_high_uncertainty():
    # Completely different responses -> many clusters -> high uncertainty
    responses = [
        "The capital of France is Paris.",
        "I think it is London.",
        "Maybe Berlin.",
        "It is definitely Tokyo.",
        "I am not sure, perhaps Madrid.",
        "Rome is the answer.",
        "Washington DC.",
        "Beijing.",
        "Ottawa.",
        "Canberra."
    ]
    result = compute_semantic_uncertainty(responses)
    assert result["n_semantic_clusters"] > 1
    assert result["uncertainty_score"] > 0.5

def test_empty_responses():
    responses = [""] * 10
    result = compute_semantic_uncertainty(responses)
    assert result["uncertainty_score"] == 1.0
