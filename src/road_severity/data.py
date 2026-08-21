"""Load, validate, clean, and transform DfT collision records.

This module enforces the schema and official code rules, resolves duplicates,
harmonises fields across source formats, creates the KSI target, and constructs
pre-collision model features while excluding identifiers and leakage fields.
"""

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
POST_COLLISION_COLUMNS = {
    "number_of_casualties",
    "did_police_officer_attend_scene_of_accident",
}
IDENTIFIER_COLUMNS = {"collision_index", "collision_ref_no", "lsoa_of_accident_location"}
GEO_COLUMNS = {"location_easting_osgr", "location_northing_osgr", "longitude", "latitude"}
SUPERSEDED_BY_UNIFIED_COLUMNS = {
    "junction_detail_historic", "junction_detail",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic", "pedestrian_crossing",
    "special_conditions_at_site", "carriageway_hazards_historic", "carriageway_hazards",
}
# Official 2011-to-2024 conversion published in the DfT open dataset data guide.
JUNCTION_DETAIL_2024_MAP = {0: 0, 1: 0, 2: 0, 3: 13, 5: 0, 6: 16, 7: 17, 8: 18, 9: 19, 99: 99}

# The official guide distinguishes why -1 occurs by field. Values are converted to
# missing in the analysis file, while their meaning and counts are retained in a report.
MINUS_ONE_MEANINGS = {
    "local_authority_district": ("deprecated", "Code deprecated"),
    "local_authority_ons_district": ("not_collected", "Record predates use of this code"),
    "local_authority_highway": ("not_collected", "Record predates use of this code"),
    "first_road_number": ("unknown", "Unknown"),
    "second_road_number": ("unknown", "Unknown"),
}
DEFAULT_MINUS_ONE_MEANING = ("missing_or_out_of_range", "Data missing or out of range")


def read_raw_collisions(path: str | Path) -> pd.DataFrame:
    """Read the source exactly once without transforming values."""
    return pd.read_csv(path, low_memory=False)


def validate_schema_contract(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Check the named field groups required by the three-stage pipeline."""
    rows = []
    for group, columns in contract["required_columns"].items():
        missing = sorted(set(columns).difference(frame.columns))
        rows.append({
            "check": f"required_columns_{group}", "severity": "error",
            "failed_rows": len(missing), "example_row_indices": "",
            "details": ";".join(missing),
        })
    return pd.DataFrame(rows)


def resolve_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove identical rows but retain and report conflicting duplicate IDs."""
    exact_duplicate_mask = frame.duplicated(keep="first")
    without_exact = frame.loc[~exact_duplicate_mask].copy()
    conflict_mask = without_exact["collision_index"].duplicated(keep=False)
    conflicts = without_exact.loc[conflict_mask].sort_values("collision_index").copy()
    summary = pd.DataFrame({
        "measure": ["exact_duplicate_rows_removed", "conflicting_collision_indices", "conflicting_rows"],
        "value": [int(exact_duplicate_mask.sum()), int(conflicts["collision_index"].nunique()), len(conflicts)],
    })
    return without_exact, summary, conflicts


def parse_temporal_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse DfT day-first dates and 24-hour times after raw-data auditing."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
    frame["time"] = pd.to_datetime(frame["time"], format="%H:%M", errors="coerce")
    return frame


def stratified_sample(frame: pd.DataFrame, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    """Sample reproducibly within year and severity so rare fatal records are retained."""
    if max_rows and len(frame) > max_rows:
        strata = ["collision_year", "collision_severity"]
        shares = frame.groupby(strata, dropna=False).size() / len(frame)
        counts = (shares * max_rows).round().astype(int)
        counts.iloc[0] += max_rows - counts.sum()
        samples = [
            group.sample(n=min(len(group), counts.loc[key]), random_state=random_state)
            for key, group in frame.groupby(strata, sort=False, dropna=False)
        ]
        frame = pd.concat(samples, ignore_index=True)
    return frame.copy()


def load_collisions(path: str | Path, max_rows: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Compatibility loader: read, validate, parse, then stratify."""
    return stratified_sample(parse_temporal_fields(read_raw_collisions(path)), max_rows, random_state)


def validate_collisions(frame: pd.DataFrame, contract: dict | None = None) -> pd.DataFrame:
    """Return aggregated validation failures without silently deleting records."""
    date = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
    time = pd.to_datetime(frame["time"], format="%H:%M", errors="coerce")
    numeric = {column: pd.to_numeric(frame[column], errors="coerce") for column in [
        "collision_year", "collision_severity", "number_of_vehicles", "number_of_casualties"
    ]}
    checks = {
        "missing_collision_index": frame["collision_index"].isna() | frame["collision_index"].astype("string").str.strip().eq(""),
        "invalid_collision_severity": ~numeric["collision_severity"].isin([1, 2, 3]),
        "invalid_collision_year": numeric["collision_year"].isna() | ~numeric["collision_year"].between(1979, 2100),
        "invalid_date": date.isna(),
        "invalid_time": time.isna(),
        "date_year_mismatch": date.notna() & numeric["collision_year"].notna() & date.dt.year.ne(numeric["collision_year"]),
        "invalid_number_of_vehicles": numeric["number_of_vehicles"].isna() | numeric["number_of_vehicles"].lt(1),
        "invalid_number_of_casualties": numeric["number_of_casualties"].isna() | numeric["number_of_casualties"].lt(1),
    }
    if "speed_limit" in frame:
        speed = pd.to_numeric(frame["speed_limit"], errors="coerce")
        checks["invalid_speed_limit"] = ~speed.isin([-1, 20, 30, 40, 50, 60, 70, 99])
    if {"longitude", "latitude"}.issubset(frame.columns):
        lon = pd.to_numeric(frame["longitude"], errors="coerce")
        lat = pd.to_numeric(frame["latitude"], errors="coerce")
        checks["coordinates_partially_missing"] = lon.isna() ^ lat.isna()
        checks["coordinates_outside_gb_bounds"] = lon.notna() & lat.notna() & (~lon.between(-9, 3) | ~lat.between(49, 61))
    severity = {
        "coordinates_partially_missing": "warning",
    }
    rows = []
    for check, mask in checks.items():
        examples = frame.index[mask].tolist()[:5]
        rows.append({
            "check": check, "severity": severity.get(check, "error"),
            "failed_rows": int(mask.sum()), "example_row_indices": ";".join(map(str, examples)),
            "details": "",
        })
    if contract:
        for column, allowed in contract.get("allowed_codes", {}).items():
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & (values.isna() | ~values.isin(allowed))
            unexpected = sorted(frame.loc[invalid, column].astype("string").unique().tolist())
            rows.append({
                "check": f"unexpected_code_{column}", "severity": "error",
                "failed_rows": int(invalid.sum()),
                "example_row_indices": ";".join(map(str, frame.index[invalid].tolist()[:5])),
                "details": ";".join(map(str, unexpected[:20])),
            })
    return pd.DataFrame(rows)


def coded_missing_summary(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Count field-specific -1, 9 and 99 meanings defined by the official guide."""
    rows = []
    configured = set()
    for column, entries in contract.get("special_codes", {}).items():
        if column not in frame:
            continue
        for entry in entries:
            code = entry["code"]
            configured.add((column, code))
            values = pd.to_numeric(frame[column], errors="coerce")
            rows.append({
                "column": column, "code": code, "meaning_category": entry["category"],
                "official_meaning": entry["meaning"], "count": int(values.eq(code).sum()),
            })
    # Every remaining -1 is explicitly retained in the audit with the official default meaning.
    for column in frame.columns:
        count = int(frame[column].eq(-1).sum() if pd.api.types.is_numeric_dtype(frame[column]) else frame[column].eq("-1").sum())
        if count and (column, -1) not in configured:
            category, meaning = DEFAULT_MINUS_ONE_MEANING
            rows.append({"column": column, "code": -1, "meaning_category": category, "official_meaning": meaning, "count": count})
    return pd.DataFrame(rows).sort_values(["count", "column"], ascending=[False, True])


def missing_code_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible -1-only summary used by older callers."""
    rows = []
    for column in frame.columns:
        count = int(frame[column].eq(-1).sum() if pd.api.types.is_numeric_dtype(frame[column]) else frame[column].eq("-1").sum())
        if count:
            category, meaning = MINUS_ONE_MEANINGS.get(column, DEFAULT_MINUS_ONE_MEANING)
            rows.append({"column": column, "code": -1, "meaning_category": category, "official_meaning": meaning, "count": count})
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def field_unification_summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Describe how much each explicit unified field recovered from retained sources."""
    pairs = {
        "junction_detail_unified": "junction_detail",
        "pedestrian_crossing_unified": "pedestrian_crossing",
        "carriageway_hazards_unified": "carriageway_hazards",
    }
    rows = []
    for unified, primary in pairs.items():
        primary_values = pd.to_numeric(before[primary], errors="coerce").replace(-1, np.nan)
        unified_values = pd.to_numeric(after[unified], errors="coerce")
        primary_missing = primary_values.isna()
        rows.append({
            "unified_column": unified,
            "primary_column": primary,
            "primary_missing_count": int(primary_missing.sum()),
            "recovered_from_retained_sources": int((primary_missing & unified_values.notna()).sum()),
            "final_missing_count": int(unified_values.isna().sum()),
            "source_column": f"{unified}_source",
            "official_2024_format_count": int(after[f"{unified}_source"].eq("official_2024_format").sum()),
            "historic_mapped_count": int(after[f"{unified}_source"].eq("historic_mapped").sum()),
        })
    return pd.DataFrame(rows)


def make_target(frame: pd.DataFrame, task: str = "binary_ksi") -> pd.Series:
    """Return a prediction target without exposing post-collision outcome fields."""
    severity = pd.to_numeric(frame["collision_severity"], errors="coerce")
    if task == "binary_ksi":
        # KSI is fatal (1) or serious (2); slight injury (3) is the reference class.
        return severity.isin([1, 2]).astype("int8").rename("ksi")
    if task == "multiclass":
        return severity.rename("collision_severity")
    raise ValueError("task must be 'binary_ksi' or 'multiclass'")


def clean_collisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready copy while preserving the original coded fields."""
    cleaned = frame.copy()
    # Prefer the already-converted 2024-format field, but recover an unambiguous
    # junction value from the retained historic field when the converted value is missing.
    empty = pd.Series(np.nan, index=cleaned.index)
    historic_junction = pd.to_numeric(cleaned.get("junction_detail_historic", empty), errors="coerce").map(JUNCTION_DETAIL_2024_MAP)
    current_junction = pd.to_numeric(cleaned.get("junction_detail", empty), errors="coerce").replace(-1, np.nan)
    cleaned["junction_detail_unified"] = current_junction.fillna(historic_junction)
    cleaned["junction_detail_unified_source"] = np.select(
        [current_junction.notna(), current_junction.isna() & historic_junction.notna()],
        ["official_2024_format", "historic_mapped"], default="missing",
    )
    for unified, current in {
        "pedestrian_crossing_unified": "pedestrian_crossing",
        "carriageway_hazards_unified": "carriageway_hazards",
    }.items():
        cleaned[unified] = pd.to_numeric(cleaned.get(current, empty), errors="coerce").replace(-1, np.nan)
        cleaned[f"{unified}_source"] = np.where(cleaned[unified].notna(), "official_2024_format", "missing")
    cleaned = cleaned.replace({-1: np.nan, "-1": np.nan})
    cleaned["ksi"] = make_target(cleaned)
    cleaned["hour"] = cleaned["time"].dt.hour
    cleaned["month"] = cleaned["date"].dt.month
    cleaned["year_month"] = cleaned["date"].dt.to_period("M").astype("string")
    cleaned["time"] = cleaned["time"].dt.strftime("%H:%M")
    return cleaned.reset_index(drop=True)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build pre-collision features, excluding IDs and outcome-derived leakage fields."""
    features = frame.drop(
        columns=[
            "collision_severity", "ksi", "year_month",
            *TARGET_LEAKAGE_COLUMNS, *POST_COLLISION_COLUMNS, *IDENTIFIER_COLUMNS,
            *SUPERSEDED_BY_UNIFIED_COLUMNS,
            "junction_detail_unified_source", "pedestrian_crossing_unified_source",
            "carriageway_hazards_unified_source",
        ],
        errors="ignore",
    ).copy()
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
