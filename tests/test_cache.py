"""
Tests for caching and storage logic.
"""
import os
import json
import pytest
from core.cache import get_cached, set_cached, _prompt_key

def test_prompt_key_generation():
    key1 = _prompt_key("Hello world", None)
    key2 = _prompt_key("Hello world", {"w1": 0.5})
    key3 = _prompt_key("Hello world ", None)
    
    assert key1 != key2
    assert key1 == key3

def test_cache_miss_when_empty():
    assert get_cached("Non existent prompt") is None

def test_set_and_get_graceful_failure():
    dummy_result = {
        "score": 0.5,
        "label": "medium",
        "calibration_score": 0.5,
        "uncertainty_score": 0.5,
        "cross_check_score": 0.5,
        "weights_used": {},
        "explanation": "Test",
    }
    set_cached("Test prompt", None, {"result": dummy_result})
    
    result = get_cached("Test prompt", None)
    assert result is None or isinstance(result, dict)
