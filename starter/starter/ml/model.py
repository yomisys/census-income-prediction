from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Train and return a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Calculate precision, recall, and F-beta score.
    """
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    return precision, recall, fbeta


def inference(model, X):
    """
    Generate predictions from a trained model.
    """
    return model.predict(X)
