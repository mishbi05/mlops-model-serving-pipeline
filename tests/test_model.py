from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_predict_positive():
    res = client.post("/predict", json={"text": "I love deploying automated cloud infrastructure!"})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "POSITIVE"
    assert "latency_ms" in data

def test_predict_empty_validation():
    res = client.post("/predict", json={"text": "   "})
    assert res.status_code == 400

def test_metrics_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert b"ml_predictions_total" in res.content
