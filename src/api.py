
"""FastAPI prediction service for the educational heart disease model."""

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_disease_api")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

model = None
started_at = time.time()
metrics_lock = threading.Lock()

metrics = {
    "requests_total": 0,
    "responses_2xx": 0,
    "responses_4xx": 0,
    "responses_5xx": 0,
    "predictions_total": 0,
    "predictions_absence": 0,
    "predictions_presence": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    # Load only trusted model files from this project.
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational demonstration of a machine-learning classifier. "
        "It is not a medical diagnostic tool."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


class HeartDiseaseInput(BaseModel):
    """Input features for one prediction."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(gt=0, le=300)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=20)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_absence: float
    probability_presence: float
    disclaimer: str


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Log request metadata and latency without input feature values."""
    start = time.perf_counter()

    with metrics_lock:
        metrics["requests_total"] += 1

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        with metrics_lock:
            metrics["responses_5xx"] += 1
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000

    with metrics_lock:
        if response.status_code < 400:
            metrics["responses_2xx"] += 1
        elif response.status_code < 500:
            metrics["responses_4xx"] += 1
        else:
            metrics["responses_5xx"] += 1

    logger.info(
        "request method=%s path=%s status=%d duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health")
def health_check() -> dict:
    """Report whether the prediction model is loaded."""
    is_loaded = model is not None
    return {
        "status": "ok" if is_loaded else "model_not_loaded",
        "model_loaded": is_loaded,
    }


@app.get("/metrics")
def get_metrics() -> dict:
    """Return in-memory API and prediction counters."""
    with metrics_lock:
        snapshot = dict(metrics)

    snapshot["uptime_seconds"] = round(time.time() - started_at, 2)
    return snapshot


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HeartDiseaseInput) -> PredictionResponse:
    """Predict the model's binary class for supplied input features."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    row = pd.DataFrame(
        [input_data.model_dump()],
        columns=FEATURE_NAMES,
    )

    try:
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
    except Exception as exc:
        logger.exception("model_prediction_failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    with metrics_lock:
        metrics["predictions_total"] += 1
        if prediction == 1:
            metrics["predictions_presence"] += 1
        else:
            metrics["predictions_absence"] += 1

    logger.info("prediction_completed class=%s", prediction)

    return PredictionResponse(
        prediction=prediction,
        prediction_label="Presence" if prediction == 1 else "Absence",
        probability_absence=float(probabilities[0]),
        probability_presence=float(probabilities[1]),
        disclaimer=(
            "Educational demonstration only; not for clinical "
            "diagnosis or medical decision-making."
        ),
    )
