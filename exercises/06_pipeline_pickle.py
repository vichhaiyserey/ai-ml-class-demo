from pathlib import Path

import pandas as pd
import joblib
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def create_regression_pipeline():
    """Create a scikit-learn regression pipeline with preprocessing + model."""
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression()),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {"MSE": mean_squared_error(y_test, y_pred), "R2": r2_score(y_test, y_pred)}


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def test_pickle_roundtrip():
    """Train, save, load and re-evaluate a model on the same test data."""
    model, X_test, y_test = create_regression_pipeline()

    metrics_before = evaluate_regression(model, X_test, y_test)
    print(f"Metrics before save: {metrics_before}")

    filepath = MODELS_DIR / "best_model.joblib"
    save_model(model, filepath)
    loaded_model = load_model(filepath)

    metrics_after = evaluate_regression(loaded_model, X_test, y_test)
    print(f"Metrics after load: {metrics_after}")
    assert metrics_before.keys() == metrics_after.keys()
    return metrics_before, metrics_after


def churn_pipeline_with_pickle():
    """Full workflow on sample data: train -> save -> load -> evaluate."""
    data = pd.read_csv(DATA_DIR / "sample_data.csv")
    X = data[["feature1", "feature2"]]
    y = data["target"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline.fit(X_train, y_train)
    print(f"Churn pipeline accuracy: {accuracy_score(y_test, pipeline.predict(X_test)):.4f}")

    path = MODELS_DIR / "churn_pipeline.joblib"
    save_model(pipeline, path)
    loaded_pipeline = load_model(path)

    print(f"Loaded pipeline accuracy: {accuracy_score(y_test, loaded_pipeline.predict(X_test)):.4f}")


if __name__ == "__main__":
    test_pickle_roundtrip()
    print("\n--- Churn Pipeline ---")
    churn_pipeline_with_pickle()
