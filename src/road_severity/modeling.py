'''
Build, tune, evaluate, and persist collision-severity classifiers.

The module supplies leakage-aware preprocessing pipelines, expanding-year
LightGBM tuning, validation-only threshold selection, held-out metrics, and
serialisation helpers used by the Stage 3 scripts.
'''

from __future__ import annotations

import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import ParameterSampler


CODED_CATEGORICAL_FEATURES = {
    'police_force', 'day_of_week', 'local_authority_district',
    'local_authority_ons_district', 'local_authority_highway',
    'local_authority_highway_current', 'first_road_class', 'first_road_number',
    'road_type', 'junction_control', 'second_road_class', 'second_road_number',
    'light_conditions', 'weather_conditions', 'road_surface_conditions',
    'special_conditions_at_site', 'urban_or_rural_area', 'trunk_road_flag',
    'junction_detail_unified', 'pedestrian_crossing_unified',
    'carriageway_hazards_unified',
}


def feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    '''Separate continuous measurements from coded or textual categories.'''
    categorical = [
        column for column in features
        if column in CODED_CATEGORICAL_FEATURES
        or not pd.api.types.is_numeric_dtype(features[column])
    ]
    numeric = [column for column in features if column not in categorical]
    return numeric, categorical


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    '''Preserve a DataFrame while making missing values valid for CatBoost.'''

    def __init__(self, categorical_columns: list[str]):
        self.categorical_columns = categorical_columns

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.numeric_columns_ = [
            column for column in self.feature_names_in_
            if column not in self.categorical_columns
        ]
        self.numeric_medians_ = {
            column: pd.to_numeric(X[column], errors='coerce').median()
            for column in self.numeric_columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X[self.feature_names_in_].copy()
        for column in self.categorical_columns:
            transformed[column] = (
                transformed[column].astype('string').fillna('__MISSING__').astype(str)
            )
        for column in self.numeric_columns_:
            transformed[column] = pd.to_numeric(
                transformed[column], errors='coerce'
            ).fillna(self.numeric_medians_[column])
        return transformed


def temporal_split(frame: pd.DataFrame, validation_year: int, test_year: int):
    '''
    Use past years for fitting and future years for honest validation/test evaluation.
    '''
    train = frame[frame.collision_year < validation_year].copy()
    valid = frame[frame.collision_year == validation_year].copy()
    test = frame[frame.collision_year == test_year].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Temporal split is empty; check the years available in the dataset.")
    return train, valid, test


def make_pipeline(features: pd.DataFrame, random_state: int, settings: dict, model_kind: str = "lightgbm") -> Pipeline:
    '''
    Create a fitted-ready preprocessing and classifier pipeline for one model kind.
    '''
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    numeric, categorical = feature_groups(features)
    if model_kind == 'catboost':
        estimator = CatBoostClassifier(
            loss_function='Logloss',
            iterations=settings.get('catboost_iterations', settings.get('max_iter', 250)),
            depth=settings.get('catboost_depth', 7),
            learning_rate=settings.get('catboost_learning_rate', 0.08),
            l2_leaf_reg=settings.get('l2_regularization', 3.0),
            auto_class_weights='Balanced',
            cat_features=categorical,
            random_seed=random_state,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        return Pipeline([
            ('preprocess', CatBoostPreprocessor(categorical)),
            ('model', estimator),
        ])
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
    if model_kind == 'extra_trees':
        estimator = ExtraTreesClassifier(
            n_estimators=settings.get('extra_trees_estimators', 300),
            max_depth=settings.get('extra_trees_max_depth', 18),
            min_samples_leaf=settings.get('extra_trees_min_samples_leaf', 5),
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([
            ('preprocess', preprocessing),
            ('model', estimator),
        ])
    if model_kind == "dummy":
        estimator = DummyClassifier(strategy="prior", random_state=random_state)
    elif model_kind == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=random_state)
    elif model_kind == "lightgbm":
        estimator = LGBMClassifier(
            objective="binary", n_estimators=settings["max_iter"],
            learning_rate=settings["learning_rate"], num_leaves=settings["num_leaves"],
            reg_lambda=settings["l2_regularization"],
            colsample_bytree=settings.get("feature_fraction", 1.0),
            subsample=settings.get("bagging_fraction", 1.0),
            subsample_freq=settings.get("bagging_freq", 0), class_weight="balanced",
            min_child_samples=settings.get("min_child_samples", 20),
            max_depth=settings.get("max_depth", -1),
            reg_alpha=settings.get("l1_regularization", 0.0),
            random_state=random_state, n_jobs=-1, verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    # Dense ordinal encoding keeps the full five-year extract tractable.
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", estimator),
    ])


def tune_lightgbm(
    features: pd.DataFrame,
    target: pd.Series,
    years: pd.Series,
    base_settings: dict,
    parameter_space: dict,
    n_iter: int,
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    '''
    Tune once with expanding-year validation and return persisted-ready results.
    '''
    unique_years = sorted(pd.Series(years).dropna().unique().tolist())
    if len(unique_years) < 2:
        raise ValueError("LightGBM tuning requires at least two training years.")
    folds = [
        (years < validation_year, years == validation_year, int(validation_year))
        for validation_year in unique_years[1:]
    ]
    rows = []
    candidates = list(ParameterSampler(parameter_space, n_iter=n_iter, random_state=random_state))
    for candidate_id, candidate in enumerate(candidates, start=1):
        settings = {**base_settings, **candidate}
        fold_scores = []
        row = {"candidate": candidate_id, **candidate}
        for train_mask, valid_mask, validation_year in folds:
            model = make_pipeline(features.loc[train_mask], random_state, settings, "lightgbm")
            model.fit(features.loc[train_mask], target.loc[train_mask])
            probabilities = model.predict_proba(features.loc[valid_mask])[:, 1]
            score = float(average_precision_score(target.loc[valid_mask], probabilities))
            row[f"ap_{validation_year}"] = score
            fold_scores.append(score)
        row["mean_average_precision"] = float(np.mean(fold_scores))
        row["std_average_precision"] = float(np.std(fold_scores))
        rows.append(row)
    results = pd.DataFrame(rows).sort_values(
        ["mean_average_precision", "std_average_precision"], ascending=[False, True]
    ).reset_index(drop=True)
    best_keys = parameter_space.keys()
    best_settings = {**base_settings, **{key: results.loc[0, key] for key in best_keys}}
    for key, value in best_settings.items():
        if isinstance(value, np.generic):
            best_settings[key] = value.item()
    return best_settings, results


def select_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    '''
    Choose a validation threshold that maximises F1 for the minority KSI class.
    '''
    candidates = np.linspace(0.05, 0.75, 141)
    scores = [f1_score(y, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    '''
    Evaluate probability ranking, calibration, and thresholded KSI decisions.
    '''
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "prevalence": float(y.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "ksi_precision": float(precision_score(y, predictions, zero_division=0)),
        "ksi_recall": float(recall_score(y, predictions, zero_division=0)),
        "ksi_f1": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(y, predictions, output_dict=True, zero_division=0),
    }


def save_model_outputs(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, out_dir: str | Path, threshold: float = 0.5) -> dict:
    '''
    Persist the trained pipeline, test metrics, and permutation importance for slides.
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, X_test, y_test, threshold)
    importance = permutation_importance(model, X_test, y_test, n_repeats=5, scoring="average_precision", random_state=42, n_jobs=-1)
    ranking = pd.DataFrame({"feature": X_test.columns, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    joblib.dump(model, out_dir / "severity_pipeline.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ranking.to_csv(out_dir / "permutation_importance.csv", index=False)
    return metrics
