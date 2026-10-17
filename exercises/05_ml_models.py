from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


def regression_demo():
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Regression - MSE: {mse:.4f}, R2: {r2:.4f}")


def classification_demo():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Classification - Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


def iris_classification():
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print(f"Iris KNN - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))


def churn_prediction():
    """Practical: churn prediction with sample data"""
    data = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'sample_data.csv')
    X = data[['feature1', 'feature2']]
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Churn Prediction - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    regression_demo()
    classification_demo()
    iris_classification()
    churn_prediction()