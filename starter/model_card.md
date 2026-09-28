# Census Income Prediction Model Card

## Model Details

This model is a Random Forest Classifier built using Scikit-Learn.

## Intended Use

This model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from the UCI Census Income dataset.

## Training Data

The model was trained using the UCI Adult Census Income dataset.

## Evaluation Metrics

Precision: 0.7353

Recall: 0.6378

F1 Score: 0.6831

## Ethical Considerations

The dataset contains demographic variables such as race, sex, education, and occupation. Predictions may reflect historical societal biases present in the training data.

## Caveats and Recommendations

The model should not be used as the sole basis for employment, lending, legal, insurance, or other high-impact decisions.

Human review should always accompany model predictions.
