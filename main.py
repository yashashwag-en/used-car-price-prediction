
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Used Car Price Prediction API",
    description="Predict used car selling prices using Machine Learning",
    version="1.0.0",
)


class CarDetails(BaseModel):
    Year: int = Field(ge=1980, le=2035)
    Present_Price: float = Field(gt=0)
    Kms_Driven: int = Field(ge=0)
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int = Field(ge=0, le=5)


@app.get("/")
def home():
    return {
        "message": "Used Car Price Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_price(car: CarDetails):
    data = pd.DataFrame([car.model_dump()])

    prediction = float(model.predict(data)[0])

    return {
        "predicted_selling_price": round(prediction, 2),
        "currency": "dataset price units"
    }
