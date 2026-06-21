import pytest
import os
import json
import numpy as np
from core.meta_classifier import MetaClassifier

def test_meta_classifier_fallback():
    clf = MetaClassifier()
    assert clf.is_fitted is False
    # Fallback should run and produce a score in [0, 1]
    score = clf.predict_proba(0.8, 0.7, 0.9, mode="full")
    assert 0.0 <= score <= 1.0
    
    score_2s = clf.predict_proba(0.8, 0.7, mode="2-signal")
    assert 0.0 <= score_2s <= 1.0

def test_meta_classifier_fit_and_save(tmp_path):
    clf = MetaClassifier()
    cal = [0.1, 0.2, 0.8, 0.9, 0.85]
    unc = [0.15, 0.25, 0.75, 0.85, 0.8]
    cc = [0.2, 0.1, 0.9, 0.8, 0.85]
    labels = [0, 0, 1, 1, 1]
    
    stats = clf.fit(cal, unc, cc, labels)
    assert stats["success"] is True
    assert clf.is_fitted is True
    
    save_path = str(tmp_path / "meta_model.json")
    clf.save(save_path)
    
    clf2 = MetaClassifier()
    clf2.load(save_path)
    assert clf2.is_fitted is True
    assert clf2.coef_full == clf.coef_full
    assert clf2.intercept_full == clf.intercept_full
    
    # Predict and verify output
    prob = clf2.predict_proba(0.8, 0.75, 0.85, mode="full")
    assert 0.5 < prob <= 1.0  # Should predict high risk for high input signals
    
    prob_low = clf2.predict_proba(0.1, 0.1, 0.1, mode="full")
    assert 0.0 <= prob_low < 0.5  # Should predict low risk for low input signals
