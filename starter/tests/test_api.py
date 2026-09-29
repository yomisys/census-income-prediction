# flake8: noqa

import sys
import os

sys.path.append(os.path.abspath("."))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    """Test that the root endpoint returns the HTML landing page."""
    response = client.get("/")

    assert response.status_code == 200
    assert "Census Income Predictor" in response.text


def test_post_prediction_lte_50k():
    response = client.post(
        "/predict",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_prediction_gt_50k():
    response = client.post(
        "/predict",
        json={
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 209642,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 15024,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States"
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
    