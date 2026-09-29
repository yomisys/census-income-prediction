# Census Income Prediction Model Card

## Model Details

This model is a Random Forest Classifier built using Scikit-Learn.

## Intended Use

This model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from the UCI Census Income dataset.

## Training Data

The model was trained using the UCI Adult Census Income dataset.

## Evaluation Data
 
The model was evaluated using a held-out 20% test partition derived from the UCI Adult Census Income dataset. The training and evaluation datasets were created using an 80/20 train-test split with stratification on the salary label to preserve the class distribution across both partitions.
 
The preprocessing artifacts fitted on the training data, including the OneHotEncoder and LabelBinarizer, were reused to transform the evaluation data. No preprocessing parameters were fitted on the evaluation partition, ensuring that evaluation remained independent of the training process.
 
The reported Precision, Recall, and F1 scores were calculated using predictions generated on this held-out test dataset after model training was completed.
Show more lines

## Evaluation Metrics

Precision: 0.7353

Recall: 0.6378

F1 Score: 0.6831

## Ethical Considerations

The dataset contains demographic variables such as race, sex, education, and occupation. Predictions may reflect historical societal biases present in the training data.

## Caveats and Recommendations

The model should not be used as the sole basis for employment, lending, legal, insurance, or other high-impact decisions.

Human review should always accompany model predictions.
