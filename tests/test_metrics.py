"""
Tests for Prometheus metrics integration in the FastAPI backend.
"""
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_metrics_endpoint():
    """Verify that the /metrics endpoint returns 200 and custom metrics definitions."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "api_request_duration_seconds" in response.text
    assert "hallucination_score" in response.text
    assert "cache_hits_total" in response.text
    assert "risk_labels_total" in response.text
    assert "api_active_requests" in response.text

def test_metrics_increment_on_request(monkeypatch):
    """Verify that calling the /analyze endpoint increments request counters and records details."""
    mock_result = {
        "prompt": "Test query",
        "result": {
            "score": 0.15,
            "label": "low",
            "explanation": "Mocked response",
            "calibration_score": 0.1,
            "uncertainty_score": 0.1,
            "cross_check_score": 0.2,
            "weights_used": {},
            "thresholds_used": {},
            "n_samples_used": 1,
            "groq_available": False,
            "mode": "2-signal"
        }
    }
    
    monkeypatch.setattr("backend.api.run_full_pipeline", lambda *args, **kwargs: mock_result)
    monkeypatch.setattr("backend.api.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.api.set_cached", lambda *args, **kwargs: None)

    response = client.post("/analyze", json={"prompt": "test query for metrics", "explain": False})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    import time
    for _ in range(30):
        result_resp = client.get(f"/result/{job_id}")
        if result_resp.json().get("status") != "pending":
            break
        time.sleep(0.1)

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    
    assert 'api_requests_total{endpoint="/analyze",method="POST",status="202"}' in metrics_response.text
    assert 'cache_hits_total{status="miss"}' in metrics_response.text
    assert 'risk_labels_total{label="low"}' in metrics_response.text
