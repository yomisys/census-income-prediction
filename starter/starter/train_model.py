import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "census_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

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


def main():
    data = pd.read_csv(DATA_PATH)

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["salary"],
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(
        y_test,
        predictions,
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-beta: {fbeta:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as file:
        pickle.dump(model, file)

    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as file:
        pickle.dump(encoder, file)

    with open(os.path.join(MODEL_DIR, "lb.pkl"), "wb") as file:
        pickle.dump(lb, file)

    print(f"Artifacts saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
