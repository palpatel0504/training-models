from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Iris Flower Classification API", version="1.0.0")

# Load artifacts
model = pickle.load(open("iris_model.pkl", "rb"))
scaler = pickle.load(open("iris_scaler.pkl", "rb"))
columns = pickle.load(open("iris_columns.pkl", "rb"))

# Target labels
flower_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.get("/")
def home():
    return {"message": "Iris Classification API Running 🌸"}


@app.get("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):
    # Create input DataFrame
    input_data = pd.DataFrame([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]], columns=columns)

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    return {
        "predicted_class": int(prediction),
        "flower_name": flower_names[prediction]
    }