'''Verify the CatBoost and ExtraTrees candidate pipelines.'''

import numpy as np
import pandas as pd

from road_severity.modeling import feature_groups, make_pipeline


SETTINGS = {
    'max_iter': 5,
    'learning_rate': 0.1,
    'num_leaves': 7,
    'l2_regularization': 1.0,
    'catboost_iterations': 8,
    'catboost_depth': 3,
    'extra_trees_estimators': 12,
    'extra_trees_max_depth': 5,
    'extra_trees_min_samples_leaf': 1,
}


def _training_data():
    X = pd.DataFrame({
        'speed_limit': [20, 30, 40, 50, 60, 70] * 5,
        'road_type': [1, 1, 2, 2, 3, 3] * 5,
        'weather_label': ['fine', 'rain', 'fine', None, 'fog', 'rain'] * 5,
    })
    y = pd.Series([0, 0, 0, 1, 1, 1] * 5)
    return X, y


def test_coded_numeric_fields_are_treated_as_categories():
    X, _ = _training_data()
    numeric, categorical = feature_groups(X)
    assert 'speed_limit' in numeric
    assert {'road_type', 'weather_label'}.issubset(categorical)


def test_added_tree_models_fit_and_return_probabilities():
    X, y = _training_data()
    for kind in ['extra_trees', 'catboost']:
        model = make_pipeline(X, random_state=42, settings=SETTINGS, model_kind=kind)
        model.fit(X, y)
        probabilities = model.predict_proba(X)[:, 1]
        assert probabilities.shape == (len(X),)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
