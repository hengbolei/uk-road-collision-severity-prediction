'''Generate temporal drift and robustness figures beyond the core figure story.'''

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from road_severity.drift_analysis import plot_feature_drift, plot_predicted_risk_drift
from road_severity.modeling import temporal_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument('--skip-risk-drift', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    frame = pd.read_csv(ROOT / config['data']['processed_path'],
                        low_memory=False, parse_dates=['date'])
    processed_dir = ROOT / 'reports/figures/processed'
    model_dir = ROOT / 'reports/figures/model'
    table_dir = ROOT / 'reports/tables'
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    validation_year = int(config['model']['validation_year'])
    test_year = int(config['model']['test_year'])
    train_years = sorted(int(year) for year in frame['collision_year'].unique()
                         if year < validation_year)
    scores = plot_feature_drift(
        frame, processed_dir / '24_feature_distribution_drift.png',
        reference_years=tuple(train_years), comparison_years=(test_year,))
    outputs = [(scores, 'feature_drift_scores.csv')]
    if not args.skip_risk_drift:
        best_path = ROOT / config['model']['best_params_path']
        best = yaml.safe_load(best_path.read_text(encoding='utf-8'))['lightgbm']
        settings = {**config['model'], **best}
        train, validation, test = temporal_split(frame, validation_year, test_year)
        summary = plot_predicted_risk_drift(
            train, validation, test, settings, config['project']['random_state'],
            model_dir / 'predicted_risk_drift.png')
        outputs.append((summary, 'predicted_risk_drift_summary.csv'))
    for table, filename in outputs:
        table.to_csv(table_dir / filename, index=False)
    selection = pd.DataFrame([
        ('processed/24_feature_distribution_drift.png', True, 'Exploration / feature design'),
        ('model/predicted_risk_drift.png', True, 'AI algorithm design'),
    ], columns=['figure', 'main_story', 'recommended_stage'])
    selection_path = table_dir / 'extended_figure_selection.csv'
    if selection_path.exists():
        existing = pd.read_csv(selection_path)
        selection = pd.concat([existing, selection], ignore_index=True).drop_duplicates('figure', keep='last')
    selection.to_csv(selection_path, index=False)
    print(f'Generated {len(outputs)} drift figures and tables')


if __name__ == '__main__':
    main()
