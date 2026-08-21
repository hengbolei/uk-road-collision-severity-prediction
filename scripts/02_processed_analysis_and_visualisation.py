'''
Run Stage 2 descriptive analysis on the validated processed dataset.

The script verifies Stage 1 metadata, writes all KSI summary tables, rebuilds
explanatory figures, and records a reporting-oriented figure catalogue.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_severity.processed_analysis import build_all_tables, create_all_figures


REQUIRED_PROCESSED_COLUMNS = {
    "collision_year", "collision_severity", "ksi", "date", "hour", "month",
    "collision_injury_based", "collision_adjusted_severity_serious",
    "speed_limit", "urban_or_rural_area", "light_conditions", "road_type",
    "weather_conditions", "road_surface_conditions", "day_of_week",
    "junction_detail_unified", "pedestrian_crossing_unified", "carriageway_hazards_unified",
    "junction_detail_unified_source", "pedestrian_crossing_unified_source",
    "carriageway_hazards_unified_source", "longitude", "latitude",
}


def validate_input(frame: pd.DataFrame, metadata: dict, contract: dict) -> pd.DataFrame:
    '''
    Check that processed data is complete, current, non-empty, and schema-compatible.
    '''
    checks = [
        ("metadata_requires_full_data", metadata.get("sampling_strategy") == "full_data", metadata.get("sampling_strategy")),
        ("metadata_row_count_matches_file", metadata.get("processed_rows") == len(frame), f"metadata={metadata.get('processed_rows')}; actual={len(frame)}"),
        ("metadata_contract_version_matches", metadata.get("data_contract_version") == contract.get("version"), f"metadata={metadata.get('data_contract_version')}; expected={contract.get('version')}"),
        ("required_processed_columns_present", REQUIRED_PROCESSED_COLUMNS.issubset(frame.columns), ";".join(sorted(REQUIRED_PROCESSED_COLUMNS.difference(frame.columns)))),
        ("processed_data_is_not_empty", len(frame) > 0, str(len(frame))),
    ]
    return pd.DataFrame([{"check": name, "passed": passed, "details": details} for name, passed, details in checks])


def main() -> None:
    '''
    Validate Stage 2 inputs and regenerate processed-data tables and figures.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    path = ROOT / config["data"]["processed_path"]
    metadata_path = path.parent / "processing_metadata.json"
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run Part 1 before Part 2; processed data or metadata is missing.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = yaml.safe_load((ROOT / config["data"]["schema_path"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(path, low_memory=False, parse_dates=["date"])

    table_dir = ROOT / "reports/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_input(frame, metadata, contract)
    validation.to_csv(table_dir / "processed_input_validation.csv", index=False)
    if not validation["passed"].all():
        raise RuntimeError("Part 2 input validation failed; figures were not overwritten.")

    tables = build_all_tables(frame)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    figure_dir = ROOT / "reports/figures/processed"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for existing in figure_dir.glob("*.png"):
        existing.unlink()
    catalog = create_all_figures(frame, tables, figure_dir, table_dir)
    catalog.to_csv(table_dir / "figure_catalog.csv", index=False)
    legacy_yearly = table_dir / "yearly_summary.csv"
    if legacy_yearly.exists():
        legacy_yearly.unlink()
    print(f"Wrote processed-data tables and figures for {len(frame):,} rows")


if __name__ == "__main__":
    main()
