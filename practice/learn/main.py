from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="Broiler Farming API", version="1.0.0")

model = pickle.load(open("titanic_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.get("/")
def home():
    return{"message":"xyz"}


@app.get("/predict")
def predict(
    pclass:int,
    age:int,
    sibsp:int,
    parch:int,
    fare:float,
    sex:str,
    embarked:str
):
         # ✅ Encode categorical inputs (must be here)
    sex_male = 1 if sex.lower() == "male" else 0
    embarked_q = 1 if embarked.upper() == "Q" else 0
    embarked_s = 1 if embarked.upper() == "S" else 0
         # Arrange in same training order

    data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s]])

    # Scale
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data)[0]

    return {"survived_prediction": int(prediction)}

