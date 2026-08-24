'''Train candidate models, compare on 2024, and evaluate the winner on 2025.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from road_severity.data import build_features, make_target
from road_severity.model_comparison_visualisation import create_model_comparison_figures
from road_severity.model_visualisation import create_model_figures
from road_severity.modeling import evaluate, make_pipeline, select_threshold, temporal_split


def main() -> None:
    '''Fit all candidates under one temporal protocol and persist the winner.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument(
        '--importance-rows', type=int, default=10000,
        help='Test rows used for permutation importance; 0 uses all.',
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    best_params_path = ROOT / config['model']['best_params_path']
    if not best_params_path.exists():
        raise FileNotFoundError(
            f'Best LightGBM parameters not found: {best_params_path}. '
            'Run scripts/tune_lightgbm.py once first.'
        )
    best_params = yaml.safe_load(
        best_params_path.read_text(encoding='utf-8')
    )['lightgbm']
    model_settings = {**config['model'], **best_params}
    frame = pd.read_csv(
        ROOT / config['data']['processed_path'], low_memory=False, parse_dates=['date']
    )
    train, validation, test = temporal_split(
        frame, config['model']['validation_year'], config['model']['test_year']
    )
    X_train, y_train = build_features(train), make_target(train, config['model']['task'])
    X_valid, y_valid = build_features(validation), make_target(validation, config['model']['task'])
    X_test, y_test = build_features(test), make_target(test, config['model']['task'])

    candidates = [
        'dummy', 'logistic_regression', 'extra_trees', 'catboost', 'lightgbm'
    ]
    comparison = []
    fitted = {}
    validation_probabilities = {}
    for kind in candidates:
        print(f'Training {kind}...', flush=True)
        model = make_pipeline(
            X_train, config['project']['random_state'], model_settings, kind
        )
        started = perf_counter()
        model.fit(X_train, y_train)
        elapsed = perf_counter() - started
        probabilities = model.predict_proba(X_valid)[:, 1]
        metrics = evaluate(model, X_valid, y_valid)
        comparison.append({
            'model': kind,
            'roc_auc': metrics['roc_auc'],
            'average_precision': metrics['average_precision'],
            'brier_score': metrics['brier_score'],
            'training_seconds': elapsed,
        })
        fitted[kind] = model
        validation_probabilities[kind] = probabilities

    comparison_frame = pd.DataFrame(comparison).sort_values(
        'average_precision', ascending=False
    ).reset_index(drop=True)
    comparison_frame.insert(0, 'validation_rank', range(1, len(comparison_frame) + 1))
    selected_kind = comparison_frame.iloc[0]['model']
    selected_model = fitted[selected_kind]
    threshold = select_threshold(y_valid, validation_probabilities[selected_kind])
    test_metrics = evaluate(selected_model, X_test, y_test, threshold)

    model_dir = ROOT / 'models/test'
    table_dir = ROOT / 'reports/tables'
    figure_dir = ROOT / 'reports/figures/model'
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(table_dir / 'model_comparison_validation.csv', index=False)
    joblib.dump(selected_model, model_dir / 'severity_pipeline.joblib')
    (model_dir / 'metrics.json').write_text(
        json.dumps({'selected_model': selected_kind, **test_metrics}, indent=2),
        encoding='utf-8',
    )
    create_model_comparison_figures(
        comparison_frame, y_valid, validation_probabilities, figure_dir
    )

    if args.importance_rows and len(X_test) > args.importance_rows:
        sampled = X_test.sample(
            args.importance_rows, random_state=config['project']['random_state']
        )
        sampled_y = y_test.loc[sampled.index]
    else:
        sampled, sampled_y = X_test, y_test
    result = permutation_importance(
        selected_model, sampled, sampled_y, n_repeats=5,
        scoring='average_precision', random_state=config['project']['random_state'],
        n_jobs=1,
    )
    importance = pd.DataFrame({
        'feature': sampled.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False)
    importance.to_csv(model_dir / 'permutation_importance.csv', index=False)
    create_model_figures(
        selected_model, X_test, y_test, threshold, importance, figure_dir
    )
    print(comparison_frame.to_string(index=False))
    test_ap = test_metrics.get('average_precision')
    test_auc = test_metrics.get('roc_auc')
    print(
        f'Selected {selected_kind}; test AP={test_ap:.3f}, '
        f'ROC-AUC={test_auc:.3f}, threshold={threshold:.2f}'
    )


if __name__ == '__main__':
    main()
