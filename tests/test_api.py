import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_analyze_returns_job_id():
    r = client.post("/analyze", json={"prompt": "What is a banana?"})
    assert r.status_code == 202
    assert "job_id" in r.json()

def test_result_unknown_job_404s():
    r = client.get("/result/nonexistent-id")
    assert r.status_code == 404
