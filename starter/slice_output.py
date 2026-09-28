import pickle
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics


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

            slice_df = data[data[feature] == cls]

            X, y, _, _ = process_data(
                slice_df,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = inference(model, X)

            precision, recall, fbeta = compute_model_metrics(
                y,
                preds
            )

            out.write(
                f"{cls}: "
                f"Precision={precision:.3f}, "
                f"Recall={recall:.3f}, "
                f"F1={fbeta:.3f}\n"
            )

print("slice_output.txt created")
