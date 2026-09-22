import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


FEATURES = [
    "pH",
    "Turbidity",
    "Dissolved_Oxygen",
    "Hardness",
    "Bacterial_Count",
    "TDS",
    "Conductivity",
    "Temperature"
]


def prepare_features(df):
    available = [
        column
        for column in FEATURES
        if column in df.columns
    ]

    X = df[available].copy()

    return X, available


def train_models(df):
    if "Potability" not in df.columns:
        raise ValueError(
            "Potability column is required."
        )

    X, feature_names = prepare_features(df)

    y = df["Potability"]

    if y.nunique() < 2:
        raise ValueError(
            "Potability must contain both classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        results[name] = {
            "accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions
            )
        }

        trained_models[name] = pipeline

    return {
        "models": trained_models,
        "results": results,
        "features": feature_names
    }


def save_bundle(bundle, path="models/water_quality_model.joblib"):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    joblib.dump(bundle, path)


def load_bundle(path="models/water_quality_model.joblib"):
    return joblib.load(path)
