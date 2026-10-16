
"""FastAPI prediction service for the educational heart disease model."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational demonstration of a machine-learning classifier. "
        "It is not a medical diagnostic tool."
    ),
    version="1.0.0",
)

# Keep feature names and order consistent with src/preprocess.py.
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

model = None


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


@app.on_event("startup")
def load_model() -> None:
    """Load the locally trained model when the service starts."""
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    # Only load this pickle-based model from your trusted project files.
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check() -> dict:
    """Report whether the prediction model is loaded."""
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


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
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        prediction_label=(
            "Presence" if prediction == 1 else "Absence"
        ),
        probability_absence=float(probabilities[0]),
        probability_presence=float(probabilities[1]),
        disclaimer=(
            "Educational demonstration only; not for clinical "
            "diagnosis or medical decision-making."
        ),
    )
