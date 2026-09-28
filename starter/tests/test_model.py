import sys
import os
import numpy as np

sys.path.append(os.path.abspath("."))

from starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)


def test_train_model():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)

    assert model is not None


def test_compute_model_metrics():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_inference():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)

    preds = inference(model, X)

    assert len(preds) == len(y)
