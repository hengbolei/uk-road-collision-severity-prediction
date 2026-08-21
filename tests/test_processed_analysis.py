'''
Verify Stage 2 intervals, category handling, and reporting definitions.
'''

import numpy as np
import pandas as pd
import pytest

from road_severity.processed_analysis import build_all_tables, proportion_summary, wilson_interval


def test_wilson_interval_contains_observed_proportion():
    '''
    Ensure Wilson bounds contain the observed proportion for ordinary samples.
    '''
    low, high = wilson_interval([20, 50], [100, 100])
    observed = np.array([0.2, 0.5])
    assert np.all(low < observed)
    assert np.all(observed < high)


def test_proportion_summary_keeps_unknown_and_missing_but_marks_status():
    '''
    Ensure summaries retain unknown and missing groups with explicit statuses.
    '''
    frame = pd.DataFrame({"category": [1, 1, 9, np.nan], "ksi": [1, 0, 1, 0]})
    summary = proportion_summary(frame, "category", {1: "Known", 9: "Unknown"}, {9})
    assert summary["collisions"].sum() == len(frame)
    assert set(summary["status"]) == {"observed", "unknown", "missing"}
    assert "Missing" in set(summary["label"])


def test_annual_table_uses_recorded_and_adjusted_ksi_definitions():
    '''
    Ensure annual reporting distinguishes recorded, adjusted, and injury-based KSI.
    '''
    frame = pd.DataFrame({
        "collision_year": [2024, 2024, 2025, 2025],
        "collision_severity": [1, 3, 2, 3],
        "ksi": [1, 0, 1, 0],
        "collision_injury_based": [0, 1, 1, 1],
        "collision_adjusted_severity_serious": [0.0, 0.2, 0.8, 0.4],
        "hour": [1, 2, 3, 4], "speed_limit": [30] * 4,
        "urban_or_rural_area": [1] * 4, "light_conditions": [1] * 4,
        "road_type": [6] * 4, "weather_conditions": [1] * 4,
        "road_surface_conditions": [1] * 4, "month": [1] * 4,
        "day_of_week": [1] * 4, "junction_detail_unified": [0] * 4,
        "pedestrian_crossing_unified": [0] * 4,
        "carriageway_hazards_unified": [0] * 4,
        "first_road_class": [3] * 4, "trunk_road_flag": [2] * 4,
    })
    annual = build_all_tables(frame)["annual_reporting_summary"].set_index("collision_year")
    assert annual.loc[2024, "ksi_rate"] == 0.5
    assert annual.loc[2024, "adjusted_ksi_rate"] == 0.6
    assert annual.loc[2025, "adjusted_ksi_rate"] == pytest.approx(0.6)
    assert annual.loc[2025, "injury_based_share"] == 1.0
