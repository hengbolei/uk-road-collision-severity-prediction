"""Data loading, validation, and leakage-aware feature construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_LEAKAGE_COLUMNS = {
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
}
IDENTIFIER_COLUMNS = {"collision_index", "collision_ref_no", "lsoa_of_accident_location"}
GEO_COLUMNS = {"location_easting_osgr", "location_northing_osgr", "longitude", "latitude"}


def load_collisions(path: str | Path, max_rows: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Load DfT collision data and parse the temporal fields used by the project."""
    frame = pd.read_csv(path, low_memory=False)
    required = {"collision_year", "collision_severity", "date", "time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
    frame["time"] = pd.to_datetime(frame["time"], format="%H:%M", errors="coerce")
    if max_rows and len(frame) > max_rows:
        # Files are ordered by year. Sampling within each year preserves the temporal split.
        shares = frame["collision_year"].value_counts(normalize=True)
        counts = (shares * max_rows).round().astype(int)
        counts.iloc[0] += max_rows - counts.sum()
        samples = [
            group.sample(n=min(len(group), counts.loc[year]), random_state=random_state)
            for year, group in frame.groupby("collision_year", sort=False)
        ]
        frame = pd.concat(samples, ignore_index=True)
    return frame


def make_target(frame: pd.DataFrame, task: str = "binary_ksi") -> pd.Series:
    """Return a prediction target without exposing post-collision outcome fields."""
    severity = pd.to_numeric(frame["collision_severity"], errors="coerce")
    if task == "binary_ksi":
        # KSI is fatal (1) or serious (2); slight injury (3) is the reference class.
        return severity.isin([1, 2]).astype("int8").rename("ksi")
    if task == "multiclass":
        return severity.rename("collision_severity")
    raise ValueError("task must be 'binary_ksi' or 'multiclass'")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build pre-collision features, excluding IDs and outcome-derived leakage fields."""
    features = frame.drop(columns=["collision_severity", *TARGET_LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS], errors="ignore").copy()
    if "date" in features:
        raw_date = features.pop("date")
        dt = raw_date if pd.api.types.is_datetime64_any_dtype(raw_date) else pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
        features["month"] = dt.dt.month
        features["day_of_month"] = dt.dt.day
        features["day_of_year_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
        features["day_of_year_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    if "time" in features:
        raw_time = features.pop("time")
        tm = raw_time if pd.api.types.is_datetime64_any_dtype(raw_time) else pd.to_datetime(raw_time, format="%H:%M", errors="coerce")
        hours = tm.dt.hour + tm.dt.minute / 60
        features["hour"] = hours
        features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    # Coordinates are retained at coarse precision: useful spatial context without exact location IDs.
    for column in GEO_COLUMNS.intersection(features.columns):
        features[column] = pd.to_numeric(features[column], errors="coerce").round(3)
    return features.replace({-1: np.nan, "-1": np.nan})
