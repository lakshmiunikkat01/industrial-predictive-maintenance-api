# Industrial Predictive Maintenance API

Production-ready Machine Learning system for predicting industrial machine failure using structured sensor data.  
Built with Scikit-learn, FastAPI, and Docker for real-time inference.

---

## Project Overview

This project implements an end-to-end machine learning pipeline to predict machine failures using sensor inputs such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The system includes:

- ML training pipeline with feature preprocessing
- Hyperparameter tuning using GridSearchCV
- ROC-AUC optimized model selection
- Model serialization using joblib
- REST API using FastAPI
- Docker containerization for production deployment

---

## Machine Learning Details

- Model: RandomForestClassifier  
- Cross-validation: 5-fold  
- Hyperparameter tuning: GridSearchCV  
- Evaluation Metric: ROC-AUC  
- Final ROC-AUC: ~0.89  

The model is trained using selected sensor features to avoid data leakage from derived failure categories.
## Architecture
Sensor Data -> Preprocessing Pipeline -> RandomForest Model -> Serialized Model (.pkl)
↓
FastAPI Inference Layer
↓
Docker Container

## Project Structure
industrial-predictive-maintenance-api/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
│
└── src/
└── train.py
## Run with Docker

### Build Image

docker build -t engine-api .


### Run Container

docker run -p 8000:8000 engine-api


### Open Swagger UI

Visit:

http://localhost:8000/docs


---

## Example API Request

```json
{
  "Air_temperature_K": 300,
  "Process_temperature_K": 310,
  "Rotational_speed_rpm": 1500,
  "Torque_Nm": 40,
  "Tool_wear_min": 10
}
Example Response:

{
  "machine_failure_prediction": 0,
  "failure_probability": 0.00056
}

