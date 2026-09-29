import pickle
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics


def compute_slice_performance(
    data,
    feature,
    category,
    model,
    encoder,
    lb,
    categorical_features,
):
    """
    Compute precision, recall, and F1 score for a given
    feature slice.
    """

    slice_df = data[data[feature] == category]

    X, y, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(
        y,
        preds
    )

    return precision, recall, fbeta


data = pd.read_csv("data/census_clean.csv")

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("model/lb.pkl", "rb") as f:
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

with open("slice_output.txt", "w") as out:

    for feature in cat_features:

        out.write(f"\nFeature: {feature}\n")

        for cls in data[feature].unique():

            precision, recall, fbeta = compute_slice_performance(
                data=data,
                feature=feature,
                category=cls,
                model=model,
                encoder=encoder,
                lb=lb,
                categorical_features=cat_features,
            )

            out.write(
                f"{cls}: "
                f"Precision={precision:.3f}, "
                f"Recall={recall:.3f}, "
                f"F1={fbeta:.3f}\n"
            )

print("slice_output.txt created")
