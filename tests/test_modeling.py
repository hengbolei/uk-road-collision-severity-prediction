'''
Verify temporal isolation, LightGBM evaluation, and one-off tuning behaviour.
'''

import pandas as pd

from road_severity.modeling import evaluate, make_pipeline, temporal_split, tune_lightgbm


SETTINGS = {"max_iter": 5, "learning_rate": 0.1, "num_leaves": 7, "l2_regularization": 1.0}


def test_temporal_split_keeps_future_years_out_of_training():
    '''
    Ensure validation and test years never enter the historical training split.
    '''
    frame = pd.DataFrame({"collision_year": [2022, 2023, 2024, 2025]})
    train, validation, test = temporal_split(frame, 2024, 2025)
    assert train["collision_year"].tolist() == [2022, 2023]
    assert validation["collision_year"].tolist() == [2024]
    assert test["collision_year"].tolist() == [2025]


def test_lightgbm_pipeline_and_evaluation_include_decision_metrics():
    '''
    Ensure LightGBM pipelines fit and expose ranking and decision metrics.
    '''
    X = pd.DataFrame({"speed_limit": [20, 30, 40, 50, 60, 70] * 4, "road_type": ["urban", "urban", "rural", "rural", "dual", "dual"] * 4})
    y = pd.Series([0, 0, 0, 1, 1, 1] * 4)
    model = make_pipeline(X, random_state=42, settings=SETTINGS, model_kind="lightgbm")
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    assert model.named_steps["model"].__class__.__name__ == "LGBMClassifier"
    assert {"roc_auc", "average_precision", "brier_score", "balanced_accuracy", "ksi_f1"}.issubset(metrics)
    assert sum(metrics["confusion_matrix"].values()) == len(y)


def test_one_off_tuning_uses_expanding_year_folds_and_returns_best_settings():
    '''
    Ensure tuning scores expanding-year folds and returns persisted-ready settings.
    '''
    X = pd.DataFrame({
        "speed_limit": [20, 30, 40, 50, 60, 70] * 6,
        "road_type": ["urban", "urban", "rural", "rural", "dual", "dual"] * 6,
    })
    y = pd.Series([0, 0, 0, 1, 1, 1] * 6)
    years = pd.Series([2021] * 12 + [2022] * 12 + [2023] * 12)
    best, results = tune_lightgbm(
        X, y, years, SETTINGS,
        {"num_leaves": [3, 7], "learning_rate": [0.05, 0.1]},
        n_iter=2, random_state=42,
    )
    assert len(results) == 2
    assert {"ap_2022", "ap_2023", "mean_average_precision"}.issubset(results.columns)
    assert best["num_leaves"] in {3, 7}
