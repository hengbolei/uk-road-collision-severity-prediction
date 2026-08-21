"""Run Stage 1 raw-data auditing and produce the analysis-ready collision file.

The script validates the DfT schema and values, resolves duplicates, records
quality and provenance metadata, harmonises fields, and atomically writes the
processed dataset only when error-level checks pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_severity.analysis import column_decisions_table, raw_quality_table
from road_severity.data import (
    clean_collisions, coded_missing_summary, field_unification_summary,
    parse_temporal_fields, read_raw_collisions, resolve_duplicates,
    stratified_sample, validate_collisions, validate_schema_contract,
)


def main() -> None:
    """Parse Stage 1 options, validate raw collisions, and write governed outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_path = args.data or ROOT / config["data"]["raw_path"]
    schema_path = ROOT / config["data"]["schema_path"]
    contract = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    max_rows = args.max_rows if args.max_rows is not None else config["project"]["max_rows"]

    table_dir = ROOT / "reports/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw_collisions(raw_path)
    raw_quality = raw_quality_table(raw)
    raw_quality.to_csv(table_dir / "raw_data_quality.csv", index=False)
    contract_issues = validate_schema_contract(raw, contract)
    if contract_issues["failed_rows"].sum():
        contract_issues.to_csv(table_dir / "validation_issues.csv", index=False)
        raise RuntimeError("Required data contract fields are missing; valid processed outputs were not overwritten.")

    deduplicated, duplicate_summary, duplicate_conflicts = resolve_duplicates(raw)
    duplicate_summary.to_csv(table_dir / "duplicate_summary.csv", index=False)
    duplicate_conflicts.to_csv(table_dir / "duplicate_conflicts.csv", index=False)
    validation = pd.concat([contract_issues, validate_collisions(deduplicated, contract)], ignore_index=True)
    validation = pd.concat([validation, pd.DataFrame([{
        "check": "conflicting_duplicate_collision_index", "severity": "error",
        "failed_rows": int(duplicate_conflicts["collision_index"].nunique()),
        "example_row_indices": ";".join(map(str, duplicate_conflicts.index.tolist()[:5])),
        "details": ";".join(map(str, duplicate_conflicts["collision_index"].drop_duplicates().tolist()[:10])),
    }])], ignore_index=True)
    validation.to_csv(table_dir / "validation_issues.csv", index=False)
    coded_missing_summary(raw, contract).to_csv(table_dir / "coded_missing_summary.csv", index=False)
    error_count = int(validation.loc[validation["severity"].eq("error"), "failed_rows"].sum())
    if error_count:
        raise RuntimeError(f"Data validation found {error_count} error-level failures; valid processed outputs were not overwritten.")

    parsed = parse_temporal_fields(deduplicated)
    sampled = stratified_sample(parsed, None if max_rows == 0 else max_rows, config["project"]["random_state"])
    processed = clean_collisions(sampled)
    field_unification_summary(sampled.reset_index(drop=True), processed).to_csv(
        table_dir / "field_unification_summary.csv", index=False
    )
    output = ROOT / config["data"]["processed_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    processed.to_csv(temporary_output, index=False, compression="gzip", date_format="%Y-%m-%d")
    temporary_output.replace(output)
    exact_duplicates_removed = int(duplicate_summary.loc[duplicate_summary["measure"].eq("exact_duplicate_rows_removed"), "value"].iloc[0])
    pd.DataFrame({
        "measure": ["full_raw_rows", "sampled_rows", "processed_rows", "duplicates_removed", "raw_columns", "processed_columns", "years", "ksi_share"],
        "value": [str(len(raw)), str(len(sampled)), str(len(processed)), str(exact_duplicates_removed), str(len(raw.columns)), str(len(processed.columns)), str(processed["collision_year"].nunique()), f"{processed['ksi'].mean():.12f}"],
    }).to_csv(table_dir / "processing_summary.csv", index=False)
    processed_quality = raw_quality_table(processed)
    processed_quality.to_csv(table_dir / "processed_data_quality.csv", index=False)
    column_decisions_table(processed, processed_quality).to_csv(table_dir / "column_decisions.csv", index=False)

    digest = hashlib.sha256()
    with raw_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "source_file": str(raw_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/") if raw_path.resolve().is_relative_to(ROOT.resolve()) else raw_path.name,
        "source_sha256": digest.hexdigest(),
        "source_size_bytes": raw_path.stat().st_size,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_raw_rows": len(raw),
        "processed_rows": len(processed),
        "max_rows": max_rows,
        "sampling_strategy": "full_data" if max_rows == 0 else "collision_year_x_collision_severity",
        "random_state": config["project"]["random_state"],
        "date_range": [str(processed["date"].min().date()), str(processed["date"].max().date())],
        "official_data_guide": "https://www.gov.uk/government/statistical-data-sets/road-safety-open-data",
        "data_contract": str(schema_path.relative_to(ROOT)).replace("\\", "/"),
        "data_contract_version": contract["version"],
        "data_contract_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    }
    (output.parent / "processing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(processed):,} rows to {output}")


if __name__ == "__main__":
    main()
