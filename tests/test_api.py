"""
Tests for the FastAPI REST Layer.
"""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "gpu_status" in data
    assert "models_loaded" in data

def test_analyze_endpoint_creates_job():
    # Submit a job
    response = client.post(
        "/analyze",
        json={"prompt": "Test prompt", "weights": {"calibration": 0.5, "semantic_uncertainty": 0.3, "cross_check": 0.2}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ["pending", "completed", "failed"]

def test_get_result_not_found():
    response = client.get("/result/non_existent_job_id")
    assert response.status_code == 404
