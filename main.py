from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pandas as pd

from Model.predict import (
    reduced_rf_model,
    screening_threshold,
    MODEL_VERSION
)

app = FastAPI(title="Diabetes Screening API")


# -----------------------------
# Request Schema
# -----------------------------
class DiabetesScreeningInput(BaseModel):
    age: Annotated[float, Field(..., gt=0, lt=120)]
    gender: Annotated[Literal["Male", "Female", "Other"]]
    height: Annotated[float, Field(..., gt=0.5, lt=2.5)]
    weight: Annotated[float, Field(..., gt=20, lt=300)]
    smoking_history: Annotated[
        Literal["never", "former", "current", "ever", "not current"]
    ]
    hypertension: Annotated[Literal["Yes", "No"]]
    heart_disease: Annotated[Literal["Yes", "No"]]

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / (self.height ** 2), 2)

    @computed_field
    @property
    def hypertension_bin(self) -> int:
        return 1 if self.hypertension == "Yes" else 0

    @computed_field
    @property
    def heart_disease_bin(self) -> int:
        return 1 if self.heart_disease == "Yes" else 0


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Diabetes Screening API is running"}


@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": reduced_rf_model is not None,
        "threshold_loaded": screening_threshold is not None,
        "model_version": MODEL_VERSION
    }


@app.post("/screen-diabetes")
def screen_patient(data: DiabetesScreeningInput):

    input_df = pd.DataFrame([{
        "age": data.age,
        "gender": data.gender,
        "smoking_history": data.smoking_history,
        "bmi": data.bmi,
        "hypertension": data.hypertension_bin,
        "heart_disease": data.heart_disease_bin
    }])

    prob = reduced_rf_model.predict_proba(input_df)[:, 1][0]

    screening_result = (
        "High Risk" if prob >= screening_threshold else "Low Risk"
    )

    return {
        "diabetes_risk_probability": round(float(prob), 3),
        "screening_result": screening_result,
        "screening_threshold": round(float(screening_threshold), 3),
        "model_version": MODEL_VERSION
    }
