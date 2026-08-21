"""Build reusable audit tables for the raw-data analysis stage.

This module profiles raw and processed columns, records missingness and value
ranges, and explains which fields are retained, transformed, or excluded from
modelling because of identifiers, superseded definitions, or leakage risk.
"""

from __future__ import annotations

import pandas as pd

from road_severity.data import (
    IDENTIFIER_COLUMNS, POST_COLLISION_COLUMNS, SUPERSEDED_BY_UNIFIED_COLUMNS,
    TARGET_LEAKAGE_COLUMNS,
)

def raw_quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise raw dtypes, missing values, sentinel values, and cardinality."""
    return pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "missing_count": frame.isna().sum(),
        "missing_rate": frame.isna().mean(),
        "sentinel_minus_one_count": pd.Series({column: frame[column].eq(-1).sum() if pd.api.types.is_numeric_dtype(frame[column]) else frame[column].eq("-1").sum() for column in frame}),
        "unique_count": frame.nunique(dropna=True),
    }).rename_axis("column").reset_index().sort_values(
        ["missing_rate", "sentinel_minus_one_count"], ascending=False
    )


def column_decisions_table(frame: pd.DataFrame, quality: pd.DataFrame, high_missing_threshold: float = 0.95) -> pd.DataFrame:
    """Explain how each processed column should be treated without deleting it."""
    quality_by_column = quality.set_index("column")
    rows = []
    for column in frame.columns:
        missing_rate = float(quality_by_column.loc[column, "missing_rate"])
        unique_count = int(quality_by_column.loc[column, "unique_count"])
        if column in TARGET_LEAKAGE_COLUMNS or column in POST_COLLISION_COLUMNS:
            decision, reason = "retain_not_model", "Outcome or post-collision information"
        elif column in IDENTIFIER_COLUMNS:
            decision, reason = "retain_not_model", "Identifier or fine-grained location code"
        elif column in {"collision_severity", "ksi"}:
            decision, reason = "target", "Prediction target or source of target"
        elif column in SUPERSEDED_BY_UNIFIED_COLUMNS:
            decision, reason = "retain_for_audit", "Source specification field; use explicit unified field downstream"
        elif column.endswith("_unified_source"):
            decision, reason = "retain_for_audit", "Provenance of unified field; do not use as a predictive feature"
        elif missing_rate > high_missing_threshold:
            decision, reason = "retain_not_recommended", f"Missing rate exceeds {high_missing_threshold:.0%}"
        elif unique_count <= 1:
            decision, reason = "retain_not_recommended", "Constant or entirely missing"
        else:
            decision, reason = "retain_candidate", "Available for downstream analysis subject to semantic review"
        rows.append({"column": column, "decision": decision, "reason": reason, "missing_rate": missing_rate, "unique_count": unique_count})
    return pd.DataFrame(rows)
