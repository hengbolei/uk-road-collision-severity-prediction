'''Validate the tracked result snapshot used by documentation and notebooks.'''

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_results_snapshot_has_expected_protocol_and_models():
    snapshot = json.loads(
        (ROOT / 'reports/results_snapshot.json').read_text(encoding='utf-8')
    )

    assert snapshot['schema_version'] == 1
    assert snapshot['data']['rows'] > 0
    assert len(snapshot['data']['source_sha256']) == 64
    assert snapshot['evaluation_protocol']['training_years'] == [2021, 2022, 2023]
    assert snapshot['evaluation_protocol']['validation_year'] == 2024
    assert snapshot['evaluation_protocol']['test_year'] == 2025

    models = snapshot['validation']['models']
    assert [row['model'] for row in models] == [
        'lightgbm',
        'catboost',
        'extra_trees',
        'logistic_regression',
        'dummy',
    ]
    assert [row['rank'] for row in models] == [1, 2, 3, 4, 5]


def test_results_snapshot_metrics_are_well_formed():
    snapshot = json.loads(
        (ROOT / 'reports/results_snapshot.json').read_text(encoding='utf-8')
    )

    for metric in ['roc_auc', 'average_precision', 'brier_score', 'ksi_recall']:
        assert 0 <= snapshot['test'][metric] <= 1
    assert snapshot['test']['selected_model'] == 'lightgbm'
    assert snapshot['permutation_importance']['top_features']


def test_documented_metrics_match_snapshot():
    snapshot = json.loads(
        (ROOT / 'reports/results_snapshot.json').read_text(encoding='utf-8')
    )
    documents = '\n'.join(
        (ROOT / path).read_text(encoding='utf-8')
        for path in [
            'README.md',
            'README.zh-CN.md',
            'reports/figure_story.md',
            'reports/figure_story_zh-CN.md',
        ]
    )

    for model in snapshot['validation']['models']:
        assert f"{model['average_precision']:.4f}" in documents
    for metric in ['roc_auc', 'average_precision', 'brier_score']:
        assert f"{snapshot['test'][metric]:.4f}" in documents
