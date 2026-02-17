from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("industrial_failure_model.pkl")

app = FastAPI(title="Industrial Predictive Maintenance API")

class MachineInput(BaseModel):
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running."}

@app.post("/predict")
def predict_failure(data: MachineInput):

    input_df = pd.DataFrame([{
        "Air temperature [K]": data.Air_temperature_K,
        "Process temperature [K]": data.Process_temperature_K,
        "Rotational speed [rpm]": data.Rotational_speed_rpm,
        "Torque [Nm]": data.Torque_Nm,
        "Tool wear [min]": data.Tool_wear_min
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "machine_failure_prediction": int(prediction),
        "failure_probability": float(probability)
    }
