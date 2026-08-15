import pandas as pd

from road_severity.data import build_features, make_target


def test_features_remove_outcome_leakage_and_create_time_features():
    frame = pd.DataFrame({"collision_severity": [1, 3], "date": ["01/01/2025", "02/01/2025"], "time": ["08:30", "16:00"], "enhanced_severity_collision": [1, 3], "speed_limit": [30, 40]})
    features = build_features(frame)
    assert "enhanced_severity_collision" not in features
    assert {"hour", "month", "speed_limit"}.issubset(features.columns)
    assert make_target(frame).tolist() == [1, 0]
