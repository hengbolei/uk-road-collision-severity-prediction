"""Verify data cleaning, schema enforcement, sampling, and leakage controls."""

import pandas as pd

from road_severity.data import (
    build_features, clean_collisions, coded_missing_summary, make_target,
    resolve_duplicates, stratified_sample, validate_collisions,
    validate_schema_contract,
)


def test_features_remove_outcome_leakage_and_create_time_features():
    """Ensure feature construction removes post-outcome data and derives time fields."""
    frame = pd.DataFrame({"collision_severity": [1, 3], "date": ["01/01/2025", "02/01/2025"], "time": ["08:30", "16:00"], "enhanced_severity_collision": [1, 3], "number_of_casualties": [2, 1], "did_police_officer_attend_scene_of_accident": [1, 2], "speed_limit": [30, 40]})
    features = build_features(frame)
    assert "enhanced_severity_collision" not in features
    assert "number_of_casualties" not in features
    assert "did_police_officer_attend_scene_of_accident" not in features
    assert {"hour", "month", "speed_limit"}.issubset(features.columns)
    assert make_target(frame).tolist() == [1, 0]


def test_clean_collisions_removes_duplicate_ids_and_replaces_sentinels():
    """Ensure exact duplicate handling and coded-missing replacement remain stable."""
    frame = pd.DataFrame({
        "collision_index": ["a", "a", "b"], "collision_severity": [1, 1, 3],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01"]),
        "time": pd.to_datetime(["08:30", "08:30", "16:00"], format="%H:%M"),
        "road_type": [-1, -1, 6],
    })
    deduplicated, _, conflicts = resolve_duplicates(frame)
    assert conflicts.empty
    cleaned = clean_collisions(deduplicated)
    assert len(cleaned) == 2
    assert cleaned["road_type"].isna().sum() == 1
    assert cleaned["ksi"].tolist() == [1, 0]
    assert cleaned["time"].tolist() == ["08:30", "16:00"]


def test_clean_collisions_creates_unified_fields_and_recovers_junction():
    """Ensure harmonised fields recover compatible historic junction information."""
    frame = pd.DataFrame({
        "collision_index": ["a", "b"], "collision_severity": [1, 3],
        "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
        "time": pd.to_datetime(["08:30", "16:00"], format="%H:%M"),
        "junction_detail_historic": [3, 6], "junction_detail": [-1, 16],
        "pedestrian_crossing": [13, -1], "carriageway_hazards": [0, 17],
    })
    cleaned = clean_collisions(frame)
    assert cleaned["junction_detail_unified"].tolist() == [13, 16]
    assert cleaned["junction_detail_unified_source"].tolist() == ["historic_mapped", "official_2024_format"]
    assert cleaned["pedestrian_crossing_unified"].iloc[0] == 13
    assert pd.isna(cleaned["pedestrian_crossing_unified"].iloc[1])
    features = build_features(cleaned)
    assert "junction_detail_unified" in features
    assert "junction_detail" not in features
    assert "junction_detail_historic" not in features


def test_stratified_sample_preserves_year_and_severity_strata():
    """Ensure development sampling preserves year-by-severity composition."""
    frame = pd.DataFrame({
        "collision_year": [2024] * 80 + [2025] * 20,
        "collision_severity": [3] * 60 + [2] * 20 + [3] * 15 + [2] * 5,
    })
    sampled = stratified_sample(frame, 50, random_state=42)
    assert len(sampled) == 50
    assert sampled.groupby(["collision_year", "collision_severity"]).size().to_dict() == {
        (2024, 2): 10, (2024, 3): 30, (2025, 2): 2, (2025, 3): 8,
    }


def test_validation_and_missing_code_meanings_are_reported():
    """Ensure invalid values, duplicate conflicts, and special-code meanings are reported."""
    frame = pd.DataFrame({
        "collision_index": ["a", "a"], "collision_year": [2025, 2024],
        "collision_severity": [1, 8], "date": ["01/01/2025", "bad"],
        "time": ["08:30", "25:00"], "number_of_vehicles": [1, 0],
        "number_of_casualties": [1, 0], "speed_limit": [30, 45],
        "longitude": [-1.0, 20.0], "latitude": [52.0, 52.0],
        "local_authority_district": [-1, -1], "second_road_number": [-1, 1],
    })
    deduplicated, duplicate_summary, conflicts = resolve_duplicates(frame)
    assert duplicate_summary.set_index("measure").loc["conflicting_collision_indices", "value"] == 1
    assert len(conflicts) == 2
    issues = validate_collisions(deduplicated).set_index("check")["failed_rows"]
    assert issues["invalid_collision_severity"] == 1
    assert issues["invalid_date"] == 1
    contract = {"special_codes": {
        "local_authority_district": [{"code": -1, "category": "deprecated", "meaning": "Code deprecated"}],
        "second_road_number": [{"code": -1, "category": "unknown", "meaning": "Unknown"}],
    }}
    meanings = coded_missing_summary(frame, contract).set_index("column")["meaning_category"]
    assert meanings["local_authority_district"] == "deprecated"
    assert meanings["second_road_number"] == "unknown"


def test_contract_and_allowed_codes_produce_error_level_issues():
    """Ensure missing contract fields and unexpected codes produce blocking errors."""
    frame = pd.DataFrame({
        "collision_index": ["a"], "collision_year": [2025], "collision_severity": [1],
        "date": ["01/01/2025"], "time": ["08:30"], "number_of_vehicles": [1],
        "number_of_casualties": [1], "road_type": ["not-a-code"],
    })
    contract = {
        "required_columns": {"core": ["collision_index", "road_type"], "analysis": ["missing_field"]},
        "allowed_codes": {"road_type": [-1, 1, 2, 3, 6, 7, 9, 12]},
    }
    schema_issues = validate_schema_contract(frame, contract).set_index("check")
    assert schema_issues.loc["required_columns_analysis", "failed_rows"] == 1
    code_issues = validate_collisions(frame, contract).set_index("check")
    assert code_issues.loc["unexpected_code_road_type", "severity"] == "error"
    assert code_issues.loc["unexpected_code_road_type", "failed_rows"] == 1
