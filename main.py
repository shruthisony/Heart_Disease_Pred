# Import required libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load models
with open("heart_disease.pkl","rb") as file:
    model = pickle.load(file)

with open("scalar.pkl","rb") as file:
    scalar = pickle.load(file)

# FastAPI Instance
app = FastAPI()

# Request Data
class DiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int 
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Defining the function to predict the output
@app.post("/predict")
def predict_disease(heart: DiseaseInput):
    try:
        #making data as an numpy array
        input_array = np.array([[heart.age,heart.sex,heart.cp,heart.trestbps,heart.chol,heart.fbs,heart.restecg,
                                 heart.thalach,heart.exang,heart.oldpeak,heart.slope,heart.ca,heart.thal]])
        # Apply the same scaler used during training
        input_scaled = scalar.transform(input_array)

        # Predicting the results
        prediction = model.predict(input_scaled)

        if int(prediction[0]) == 1:
            prediction = "Suffering with heart disease"

        else:
            prediction = "No heart disease"

        return {"Result":prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")