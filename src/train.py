"""Train, evaluate, and register credit risk models with MLflow tracking."""

from __future__ import annotations

import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from src.data_processing import (
    CATEGORICAL_FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
    build_model_pipeline,
    coerce_categoricals,
)

warnings.filterwarnings("ignore")

TARGET_COL = "is_high_risk"
FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
REGISTERED_MODEL_NAME = "CreditRiskClassifier"
# Stock scikit-learn/numpy internals that MLflow's skops serialiser does not trust by
# default. Listing them keeps the safer skops format instead of falling back to pickle.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
]
DEFAULT_DATA_PATH = os.path.join("data", "processed", "model_ready_data.csv")
DEFAULT_MODEL_PATH = os.path.join("models", "best_model.pkl")


def load_processed_data(file_path: str) -> pd.DataFrame:
    """Load the model-ready data from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Processed data file not found at {file_path}. "
            "Run `python -m src.data_processing` first."
        )
    return pd.read_csv(file_path)


def load_data(file_path: str = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load preprocessed data and separate raw features (X) from the target (y)."""
    data = load_processed_data(file_path)

    if TARGET_COL not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. Available: {list(data.columns)}"
        )

    missing = [col for col in FEATURE_COLS if col not in data.columns]
    if missing:
        raise ValueError(f"Processed data is missing expected features: {missing}")

    X = coerce_categoricals(data[FEATURE_COLS])
    y = data[TARGET_COL].astype(int)

    print(f"Loaded {X.shape[0]} rows and {X.shape[1]} raw features.")
    print(f"High-risk rate: {y.mean():.2%}")
    return X, y


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    """Calculate classification metrics required for model comparison."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def get_model_search_space() -> dict:
    """Candidate estimators and their hyperparameter grids.

    Grids are namespaced with ``classifier__`` because each estimator is tuned
    inside the preprocessing pipeline, which prevents scaling statistics from
    leaking across cross-validation folds.
    """
    return {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l1", "l2"],
                "classifier__solver": ["liblinear"],
            },
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            },
        },
        "GradientBoostingClassifier": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        },
    }


def train_models(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    tracking_uri: str | None = None,
) -> dict:
    """Train candidate models, log every run to MLflow, and register the best one."""
    X, y = load_data(data_path)

    if y.nunique() < 2:
        raise ValueError("Target variable must have at least 2 classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split into {X_train.shape[0]} train and {X_test.shape[0]} test rows.")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("CreditRiskModeling")

    best: dict | None = None

    for model_name, config in get_model_search_space().items():
        with mlflow.start_run(run_name=model_name):
            print(f"\n--- Training {model_name} ---")

            grid_search = GridSearchCV(
                build_model_pipeline(config["model"]),
                config["params"],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            model = grid_search.best_estimator_
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_proba)

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            )
            mlflow.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
            mlflow.sklearn.log_model(
                model,
                name=model_name,
                input_example=X_train.head(1),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

            print(f"Best params: {grid_search.best_params_}")
            print(
                f"ROC AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )
            print(f"Confusion matrix:\n{np.array(metrics['confusion_matrix'])}")

            if best is None or metrics["roc_auc"] > best["metrics"]["roc_auc"]:
                best = {"name": model_name, "estimator": model, "metrics": metrics}

    if best is None:
        raise RuntimeError("No models were trained successfully.")

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(best["estimator"], model_path)

    with mlflow.start_run(run_name="Register_Best_Model"):
        mlflow.sklearn.log_model(
            best["estimator"],
            name="best_credit_risk_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(1),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        mlflow.log_metrics(
            {k: v for k, v in best["metrics"].items() if k != "confusion_matrix"}
        )
        mlflow.set_tag("best_model", best["name"])

    print(
        f"\nBest model '{best['name']}' (ROC AUC {best['metrics']['roc_auc']:.4f}) "
        f"saved to {model_path} and registered as '{REGISTERED_MODEL_NAME}'."
    )
    return best


if __name__ == "__main__":
    train_models()
