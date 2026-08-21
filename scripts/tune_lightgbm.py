'''
Run the explicitly invoked, one-off LightGBM hyperparameter search.

The script searches expanding-year folds within 2021-2023, ranks candidates by
mean Average Precision, and persists both the winning parameters and complete
search table for later Stage 3 runs without automatic retuning.
'''

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_severity.data import build_features, make_target
from road_severity.modeling import temporal_split, tune_lightgbm


def main() -> None:
    '''
    Execute one-off temporal tuning and save reproducible parameters and results.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--iterations", type=int, help="Override the configured number of sampled candidates.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / config["data"]["processed_path"], low_memory=False, parse_dates=["date"])
    train, _, _ = temporal_split(frame, config["model"]["validation_year"], config["model"]["test_year"])
    features = build_features(train)
    target = make_target(train, config["model"]["task"])
    iterations = args.iterations or config["tuning"]["iterations"]
    best_settings, results = tune_lightgbm(
        features=features,
        target=target,
        years=train["collision_year"],
        base_settings=config["model"],
        parameter_space=config["tuning"]["parameter_space"],
        n_iter=iterations,
        random_state=config["project"]["random_state"],
    )
    parameter_keys = config["tuning"]["parameter_space"].keys()
    output = {
        "lightgbm": {key: best_settings[key] for key in parameter_keys},
        "metadata": {
            "selected_by": "mean_average_precision",
            "validation_scheme": "expanding_year",
            "training_years": sorted(int(year) for year in train["collision_year"].unique()),
            "iterations": iterations,
            "random_state": config["project"]["random_state"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_mean_average_precision": float(results.loc[0, "mean_average_precision"]),
        },
    }
    params_path = ROOT / config["model"]["best_params_path"]
    results_path = ROOT / "reports/tables/lightgbm_tuning_results.csv"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    results.to_csv(results_path, index=False)
    print(f"Saved best parameters to {params_path}")
    print(f"Best rolling-validation AP={results.loc[0, 'mean_average_precision']:.4f}")


if __name__ == "__main__":
    main()
