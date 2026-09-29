import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

MODEL_PATH = os.path.join("model", "model.pkl")
ENCODER_PATH = os.path.join("model", "encoder.pkl")
LB_PATH = os.path.join("model", "lb.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

with open(LB_PATH, "rb") as f:
    lb = pickle.load(f)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = 39
    workclass: str = "State-gov"
    fnlgt: int = 77516
    education: str = "Bachelors"

    education_num: int = Field(
        default=13,
        alias="education-num"
    )

    marital_status: str = Field(
        default="Never-married",
        alias="marital-status"
    )

    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"

    capital_gain: int = Field(
        default=2174,
        alias="capital-gain"
    )

    capital_loss: int = Field(
        default=0,
        alias="capital-loss"
    )

    hours_per_week: int = Field(
        default=40,
        alias="hours-per-week"
    )

    native_country: str = Field(
        default="United-States",
        alias="native-country"
    )


@app.get("/")
def welcome():
    return {
        "message": "Welcome to Census Income Prediction API"
    }


@app.post("/predict")
def predict(data: CensusData):
    input_data = pd.DataFrame(
        [{
            "age": data.age,
            "workclass": data.workclass,
            "fnlgt": data.fnlgt,
            "education": data.education,
            "education-num": data.education_num,
            "marital-status": data.marital_status,
            "occupation": data.occupation,
            "relationship": data.relationship,
            "race": data.race,
            "sex": data.sex,
            "capital-gain": data.capital_gain,
            "capital-loss": data.capital_loss,
            "hours-per-week": data.hours_per_week,
            "native-country": data.native_country
        }]
    )

    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = inference(model, X)[0]

    result = ">50K" if prediction == 1 else "<=50K"

    return {
        "prediction": result
    }
