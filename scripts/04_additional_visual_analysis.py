'''Generate the extended figure set without modifying notebooks or the figure story.'''

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from road_severity.additional_analysis import (
    plot_association_heatmap, plot_monthly_time_series, plot_parallel_profiles,
    plot_temporal_learning_curve, plot_vehicle_distribution,
)
from road_severity.modeling import temporal_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument('--skip-learning-curve', action='store_true')
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
    outputs = [
        (plot_monthly_time_series(frame, processed_dir / '20_monthly_time_series.png'),
         'monthly_time_series.csv'),
        (plot_association_heatmap(frame, processed_dir / '21_mixed_association_heatmap.png'),
         'mixed_association_matrix.csv'),
        (plot_vehicle_distribution(frame, processed_dir / '22_vehicle_count_violin_box.png'),
         'vehicle_count_by_severity.csv'),
        (plot_parallel_profiles(frame, processed_dir / '23_parallel_road_profiles.png'),
         'parallel_road_profiles.csv'),
    ]
    if not args.skip_learning_curve:
        best_path = ROOT / config['model']['best_params_path']
        best = yaml.safe_load(best_path.read_text(encoding='utf-8'))['lightgbm']
        settings = {**config['model'], **best}
        train, validation, _ = temporal_split(
            frame, config['model']['validation_year'], config['model']['test_year'])
        curve = plot_temporal_learning_curve(
            train, validation, settings, config['project']['random_state'],
            model_dir / 'temporal_learning_curve.png')
        outputs.append((curve, 'temporal_learning_curve.csv'))
    for table, filename in outputs:
        table.to_csv(table_dir / filename, index='matrix' in filename)
    selection = pd.DataFrame([
        ('processed/20_monthly_time_series.png', True, 'Exploration / visual story'),
        ('processed/21_mixed_association_heatmap.png', True, 'Exploration / feature design'),
        ('processed/22_vehicle_count_violin_box.png', False, 'Appendix'),
        ('processed/23_parallel_road_profiles.png', False, 'Appendix'),
        ('model/temporal_learning_curve.png', True, 'AI algorithm design'),
    ], columns=['figure', 'main_story', 'recommended_stage'])
    selection.to_csv(table_dir / 'extended_figure_selection.csv', index=False)
    print(f'Generated {len(outputs)} figures in the processed and model figure directories')


if __name__ == '__main__':
    main()
