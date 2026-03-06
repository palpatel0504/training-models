from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Adult Salary Prediction API", version="1.0.0")

# Load artifacts
model = pickle.load(open("adult.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Adult Income Prediction API Running"}

@app.get("/predict")
def predict(
    age: int,
    workclass: str,
    education_num: int,
    marital_status: str,
    occupation: str,
    relationship: str,
    race: str,
    sex: str,
    hours_per_week: int
):
    
    # Base numeric features
    input_dict = {
        "age": age,
        "education.num": education_num,
        "hours.per.week": hours_per_week
    }

    input_df = pd.DataFrame([input_dict])

    # Initialize all columns with 0
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Set categorical dummy columns
    cat_inputs = {
        f"workclass_{workclass}": 1,
        f"marital.status_{marital_status}": 1,
        f"occupation_{occupation}": 1,
        f"relationship_{relationship}": 1,
        f"race_{race}": 1,
        f"sex_{sex}": 1
    }

    for key in cat_inputs:
        if key in input_df.columns:
            input_df[key] = 1

    # Reorder exactly like training
    input_df = input_df[columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    result = ">50K" if prediction == 1 else "<=50K"

    return {"income_prediction": result}