'''Create validation-only charts for fair model comparison.'''

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve


ACCENT = '#39728C'
GREY = '#B8BDC2'
DARK_GREY = '#555B61'
GRID_GREY = '#E5E7E9'
SOURCE = 'Source: UK DfT road collision data; 2024 validation set (model selection only).'
LABELS = {
    'dummy': 'Dummy baseline',
    'logistic_regression': 'Logistic regression',
    'extra_trees': 'ExtraTrees',
    'catboost': 'CatBoost',
    'lightgbm': 'LightGBM',
}
COLORS = {
    'dummy': GREY,
    'logistic_regression': '#7A8793',
    'extra_trees': '#59A14F',
    'catboost': '#E15759',
    'lightgbm': ACCENT,
}


def _style() -> None:
    sns.set_theme(style='whitegrid', rc={
        'axes.edgecolor': DARK_GREY,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': GRID_GREY,
        'grid.linewidth': 0.7,
        'figure.facecolor': 'white',
    })


def _save(fig: plt.Figure, path: Path) -> None:
    fig.text(0.01, 0.008, SOURCE, ha='left', va='bottom', fontsize=8, color=DARK_GREY)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_model_comparison_figures(
    comparison: pd.DataFrame,
    y_valid: pd.Series,
    probability_map: dict,
    out_dir: str | Path,
) -> None:
    '''Generate ranking, PR-curve, and performance/time comparisons.'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    best = comparison.sort_values('average_precision', ascending=False).iloc[0]['model']

    ordered = comparison.sort_values('average_precision').copy()
    ordered['label'] = ordered['model'].map(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    axes[0].barh(
        ordered['label'], ordered['average_precision'],
        color=[COLORS[name] if name == best else GREY for name in ordered['model']],
    )
    for y_pos, value in enumerate(ordered['average_precision']):
        axes[0].text(value + 0.004, y_pos, f'{value:.3f}', va='center', fontsize=9)
    axes[0].set(title='Ranking quality: higher is better',
                xlabel='Validation average precision', ylabel='')
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))

    calibrated = comparison.sort_values('brier_score', ascending=False).copy()
    calibrated['label'] = calibrated['model'].map(LABELS)
    axes[1].barh(
        calibrated['label'], calibrated['brier_score'],
        color=[COLORS[name] if name == best else GREY for name in calibrated['model']],
    )
    for y_pos, value in enumerate(calibrated['brier_score']):
        axes[1].text(value + 0.002, y_pos, f'{value:.3f}', va='center', fontsize=9)
    axes[1].set(title='Probability error: lower is better',
                xlabel='Validation Brier score', ylabel='')
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(f'{LABELS[best]} provides the strongest validation ranking', fontsize=14)
    _save(fig, out_dir / 'model_validation_comparison.png')

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for model_name in comparison.sort_values('average_precision')['model']:
        precision, recall, _ = precision_recall_curve(y_valid, probability_map[model_name])
        ap = comparison.loc[
            comparison['model'] == model_name, 'average_precision'
        ].iloc[0]
        ax.plot(
            recall, precision, color=COLORS[model_name],
            linewidth=2.8 if model_name == best else 1.7,
            label=f'{LABELS[model_name]} (AP {ap:.3f})',
        )
    prevalence = float(y_valid.mean())
    ax.axhline(prevalence, linestyle='--', color=DARK_GREY, linewidth=1.2,
               label=f'Prevalence ({prevalence:.1%})')
    ax.set(title='Models differ most in precision at practical recall levels',
           xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    _save(fig, out_dir / 'model_precision_recall_comparison.png')

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for row in comparison.itertuples():
        ax.scatter(row.training_seconds, row.average_precision, s=90,
                   color=COLORS[row.model], edgecolor='white', linewidth=0.8, zorder=3)
        ax.annotate(LABELS[row.model], (row.training_seconds, row.average_precision),
                    xytext=(7, 4), textcoords='offset points', fontsize=9)
    ax.set_xscale('log')
    ax.set(title='Validation performance versus model training cost',
           xlabel='Training time (seconds, log scale)',
           ylabel='Validation average precision')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, out_dir / 'model_performance_time_tradeoff.png')
