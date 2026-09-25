from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.model import model

app = FastAPI(
    title="MLOps Model Serving API",
    description="Production-grade AI inference service with Prometheus observability",
    version="1.0.0"
)

# Prometheus Metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total prediction requests', ['sentiment'])
INFERENCE_LATENCY = Histogram('ml_inference_latency_seconds', 'Time spent processing model inference')

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="The deployment pipeline runs fast and reliably!")

@app.get("/healthz")
def health_check():
    return {"status": "healthy", "service": "model-serving"}

@app.get("/ready")
def readiness_check():
    return {"status": "ready"}

@app.post("/predict")
def predict(payload: PredictRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")


cat << 'EOF' > tests/test_model.py
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
