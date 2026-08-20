"""Model training and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


def temporal_split(frame: pd.DataFrame, validation_year: int, test_year: int):
    """Use past years for fitting and future years for honest validation/test evaluation."""
    train = frame[frame.collision_year < validation_year].copy()
    valid = frame[frame.collision_year == validation_year].copy()
    test = frame[frame.collision_year == test_year].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Temporal split is empty; check the years available in the dataset.")
    return train, valid, test


def make_pipeline(features: pd.DataFrame, random_state: int, settings: dict, model_kind: str = "hist_gradient_boosting") -> Pipeline:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # Ordinal encoding keeps the matrix compact enough for the full five-year extract.
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical),
        ],
        sparse_threshold=0,
    )
    if model_kind == "dummy":
        estimator = DummyClassifier(strategy="prior", random_state=random_state)
    elif model_kind == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=random_state)
    elif model_kind == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=settings["learning_rate"], max_leaf_nodes=settings["max_leaf_nodes"],
            l2_regularization=settings["l2_regularization"], max_iter=settings["max_iter"],
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    # Dense ordinal encoding keeps the full five-year extract tractable.
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", estimator),
    ])


def select_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a validation threshold that maximises F1 for the minority KSI class."""
    candidates = np.linspace(0.05, 0.75, 141)
    scores = [f1_score(y, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "prevalence": float(y.mean()),
        "classification_report": classification_report(y, predictions, output_dict=True, zero_division=0),
    }


def save_model_outputs(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, out_dir: str | Path, threshold: float = 0.5) -> dict:
    """Persist the trained pipeline, test metrics, and permutation importance for slides."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, X_test, y_test, threshold)
    importance = permutation_importance(model, X_test, y_test, n_repeats=5, scoring="average_precision", random_state=42, n_jobs=-1)
    ranking = pd.DataFrame({"feature": X_test.columns, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    joblib.dump(model, out_dir / "severity_pipeline.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ranking.to_csv(out_dir / "permutation_importance.csv", index=False)
    return metrics
