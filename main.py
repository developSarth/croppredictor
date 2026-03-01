from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Crop Recommendation API")

model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def home():
    return {"message": "Crop Recommendation API Running 🚀"}

@app.post("/predict")
def predict(data: CropInput):
   
    features = np.array([[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    probs = model.predict_proba(features)[0]
    top3_indices = probs.argsort()[-3:][::-1]
    top3_crops = label_encoder.inverse_transform(top3_indices)

    return {
        "top_3_crops": top3_crops.tolist()
    }