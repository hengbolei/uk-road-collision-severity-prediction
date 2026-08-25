'''Verify temporal drift helpers.'''

import numpy as np
import pandas as pd
import pytest
from road_severity.drift_analysis import (
    FEATURE_DRIFT_LIST, drift_level, feature_drift_scores, population_stability_index,
)


def test_psi_identical_numeric_is_zero():
    series = pd.Series(np.arange(1, 11).repeat(20), dtype=float)
    assert population_stability_index(series, series, categorical=False) == 0.0


def test_psi_identical_categorical_is_zero():
    series = pd.Series(['a', 'b', 'c'] * 30)
    assert population_stability_index(series, series) == 0.0


def test_psi_shifted_numeric_exceeds_significant_threshold():
    reference = pd.Series(np.random.default_rng(42).normal(5, 1, 2000))
    observed = pd.Series(np.random.default_rng(0).normal(15, 1, 2000))
    assert population_stability_index(reference, observed, categorical=False) > 0.25


def test_psi_absent_category_is_finite_and_positive():
    reference = pd.Series(['a'] * 50 + ['b'] * 40 + ['c'] * 10)
    observed = pd.Series(['a'] * 50 + ['b'] * 40)
    psi = population_stability_index(reference, observed)
    assert np.isfinite(psi)
    assert psi > 0


def test_psi_constant_reference_is_zero():
    reference = pd.Series([7, 7, 7, 7])
    observed = pd.Series([7, 7, 8, 8])
    assert population_stability_index(reference, observed) == 0.0


def test_psi_drops_missing_values():
    reference = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    observed = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0])
    assert np.isfinite(population_stability_index(reference, observed))


def test_drift_level_boundaries():
    assert drift_level(0.05) == 'stable'
    assert drift_level(0.10) == 'moderate'
    assert drift_level(0.24) == 'moderate'
    assert drift_level(0.25) == 'significant'
    assert drift_level(0.50) == 'significant'


def test_feature_drift_scores_schema_and_values():
    frame = pd.DataFrame({
        'collision_year': [2023, 2023, 2023, 2025, 2025, 2025],
        'speed_limit': [30, 30, 60, 30, 60, 60],
        'light_conditions': [1, 1, 4, 1, 4, 4],
    })
    result = feature_drift_scores(frame, [2023], [2025])
    assert list(result.columns) == ['feature', 'psi', 'drift_level']
    assert set(result['feature']) == {'speed_limit', 'light_conditions'}
    assert result['psi'].notna().all()
    assert result['drift_level'].isin(['stable', 'moderate', 'significant']).all()


def test_feature_drift_identical_years_is_zero():
    frame = pd.DataFrame({
        'collision_year': [2023, 2023, 2023, 2025, 2025, 2025],
        'speed_limit': [30, 30, 60, 30, 30, 60],
    })
    result = feature_drift_scores(frame, [2023], [2023])
    assert result['psi'].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_curated_list_excludes_collision_year():
    assert 'collision_year' not in dict(FEATURE_DRIFT_LIST)
    assert len(FEATURE_DRIFT_LIST) >= 10
