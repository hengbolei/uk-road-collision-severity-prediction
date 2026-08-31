'''Build a compact, tracked snapshot from generated analysis artefacts.'''

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _required(paths: list[Path]) -> None:
    missing = [_relative(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Generate the analysis artefacts before building the snapshot: '
            + ', '.join(missing)
        )


def build_snapshot(config_path: Path) -> dict:
    '''Read generated outputs and return a JSON-serialisable result summary.'''
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    processed_path = ROOT / config['data']['processed_path']
    processing_metadata_path = processed_path.parent / 'processing_metadata.json'
    processing_summary_path = ROOT / 'reports/tables/processing_summary.csv'
    comparison_path = ROOT / 'reports/tables/model_comparison_validation.csv'
    metrics_path = ROOT / 'models/test/metrics.json'
    importance_path = ROOT / 'models/test/permutation_importance.csv'
    params_path = ROOT / config['model']['best_params_path']
    required = [
        processing_metadata_path,
        processing_summary_path,
        comparison_path,
        metrics_path,
        importance_path,
        params_path,
    ]
    _required(required)

    processing_metadata = json.loads(
        processing_metadata_path.read_text(encoding='utf-8')
    )
    summary_frame = pd.read_csv(processing_summary_path, dtype=str)
    processing_summary = dict(
        zip(summary_frame['measure'], summary_frame['value'], strict=True)
    )
    comparison = pd.read_csv(comparison_path)
    test_metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    importance = pd.read_csv(importance_path).head(15)
    tuning = yaml.safe_load(params_path.read_text(encoding='utf-8'))

    validation_models = [
        {
            'rank': int(row.validation_rank),
            'model': str(row.model),
            'roc_auc': float(row.roc_auc),
            'average_precision': float(row.average_precision),
            'brier_score': float(row.brier_score),
            'training_seconds': float(row.training_seconds),
        }
        for row in comparison.itertuples(index=False)
    ]
    top_features = [
        {
            'feature': str(row.feature),
            'importance_mean': float(row.importance_mean),
            'importance_std': float(row.importance_std),
        }
        for row in importance.itertuples(index=False)
    ]
    compact_test_metrics = {
        key: value
        for key, value in test_metrics.items()
        if key != 'classification_report'
    }

    return {
        'schema_version': 1,
        'snapshot_generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'project': 'uk-road-collision-severity-prediction',
        'data': {
            'source_file': processing_metadata['source_file'],
            'source_sha256': processing_metadata['source_sha256'],
            'source_size_bytes': processing_metadata['source_size_bytes'],
            'processed_at_utc': processing_metadata['generated_at_utc'],
            'rows': int(processing_summary['processed_rows']),
            'date_range': processing_metadata['date_range'],
            'ksi_share': float(processing_summary['ksi_share']),
            'contract_version': processing_metadata['data_contract_version'],
        },
        'evaluation_protocol': {
            'training_years': tuning['metadata']['training_years'],
            'validation_year': int(config['model']['validation_year']),
            'test_year': int(config['model']['test_year']),
            'target': config['model']['task'],
            'selection_metric': 'average_precision',
            'threshold_selection': 'maximum_ksi_f1_on_validation_year',
            'random_state': int(config['project']['random_state']),
        },
        'tuning': {
            'validation_scheme': tuning['metadata']['validation_scheme'],
            'iterations': int(tuning['metadata']['iterations']),
            'best_mean_average_precision': float(
                tuning['metadata']['best_mean_average_precision']
            ),
            'parameters': tuning['lightgbm'],
        },
        'validation': {'models': validation_models},
        'test': compact_test_metrics,
        'permutation_importance': {
            'scoring': 'average_precision',
            'top_features': top_features,
        },
        'source_artifacts': {
            'processing_metadata': _relative(processing_metadata_path),
            'model_comparison': _relative(comparison_path),
            'test_metrics': _relative(metrics_path),
            'permutation_importance': _relative(importance_path),
            'lightgbm_parameters': _relative(params_path),
        },
    }


def main() -> None:
    '''Write the tracked result snapshot atomically.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument(
        '--output', type=Path, default=ROOT / 'reports/results_snapshot.json'
    )
    args = parser.parse_args()
    snapshot = build_snapshot(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(f'Wrote result snapshot to {args.output}')


if __name__ == '__main__':
    main()
