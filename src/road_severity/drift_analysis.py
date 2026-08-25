'''Measure temporal drift between training and deployment inputs and predicted risk.'''

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, brier_score_loss
from road_severity.data import build_features, make_target
from road_severity.modeling import make_pipeline

ACCENT, ORANGE, GREY, DARK_GREY = '#0072B2', '#E69F00', '#B8C2CC', '#4B5563'
SIGNIFICANT_RED = '#D62728'
SOURCE = 'Source: UK Department for Transport road collision data, 2021-2025.'
sns.set_theme(style='whitegrid', rc={
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#E5E7EB', 'figure.facecolor': 'white',
})

PSI_EPS = 1e-4
PSI_STABLE, PSI_MODERATE = 0.1, 0.25

# Curated model inputs whose distributions are meaningful to compare over time.
# collision_year is deliberately excluded: the model is trained on 2021-2023 and
# faces an out-of-distribution 2025 value, so that feature trivially drifts.
# High-cardinality identifiers (police force, local authorities, road numbers)
# are excluded because their categorical PSI is dominated by recording noise.
FEATURE_DRIFT_LIST = [
    ('number_of_vehicles', 'numeric'),
    ('speed_limit', 'numeric'),
    ('longitude', 'numeric'),
    ('latitude', 'numeric'),
    ('hour', 'numeric'),
    ('urban_or_rural_area', 'categorical'),
    ('light_conditions', 'categorical'),
    ('road_type', 'categorical'),
    ('weather_conditions', 'categorical'),
    ('road_surface_conditions', 'categorical'),
    ('day_of_week', 'categorical'),
    ('month', 'categorical'),
    ('first_road_class', 'categorical'),
    ('junction_detail_unified', 'categorical'),
    ('trunk_road_flag', 'categorical'),
]
FEATURE_KIND = dict(FEATURE_DRIFT_LIST)


def _save(fig: plt.Figure, path: Path, footer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.008, f'{footer}  {SOURCE}', fontsize=8, color=DARK_GREY)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def population_stability_index(reference: pd.Series, observed: pd.Series,
                               n_bins: int = 10,
                               categorical: bool | None = None) -> float:
    '''Return the Population Stability Index between two distributions.

    Numeric series are binned by reference quantiles (with +/-inf sentinels so
    out-of-range observed values count as drift); categorical series are binned
    by value counts with rare categories pooled. Missing values are dropped and
    zero proportions are clipped to a small epsilon to keep the log finite.
    '''
    reference = pd.Series(reference).dropna()
    observed = pd.Series(observed).dropna()
    if reference.empty or observed.empty:
        return 0.0
    if categorical is None:
        categorical = reference.dtype == object or reference.nunique() <= 20
    if reference.nunique() < 2:
        return 0.0
    if categorical:
        counts = reference.value_counts()
        rare = counts[counts < max(1, int(0.01 * len(reference)))].index
        mapping = {category: category for category in counts.index if category not in rare}

        def bucket(values: pd.Series) -> pd.Series:
            return values.map(mapping).fillna('__other__')

        reference_bins = bucket(reference).value_counts()
        observed_bins = bucket(observed).value_counts()
    else:
        _, edges = pd.qcut(reference, q=n_bins, duplicates='drop', retbins=True)
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0
        bins = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
        reference_bins = pd.cut(reference, bins=bins).value_counts()
        observed_bins = pd.cut(observed, bins=bins).value_counts()
    keys = reference_bins.index.union(observed_bins.index)
    expected = reference_bins.reindex(keys, fill_value=0).to_numpy(dtype=float)
    observed_counts = observed_bins.reindex(keys, fill_value=0).to_numpy(dtype=float)
    expected = np.maximum(expected / expected.sum(), PSI_EPS)
    observed_counts = np.maximum(observed_counts / observed_counts.sum(), PSI_EPS)
    return float(((observed_counts - expected) * np.log(observed_counts / expected)).sum())


def drift_level(psi: float) -> str:
    '''Classify a PSI value as stable, moderate, or significant drift.'''
    if psi < PSI_STABLE:
        return 'stable'
    if psi < PSI_MODERATE:
        return 'moderate'
    return 'significant'


def feature_drift_scores(frame: pd.DataFrame, reference_years: list[int],
                         comparison_years: list[int],
                         features: list[str] | None = None) -> pd.DataFrame:
    '''Score per-feature PSI between pooled reference and comparison years.'''
    if features is None:
        features = [column for column, _ in FEATURE_DRIFT_LIST]
    features = [column for column in features if column in frame.columns]
    reference = frame[frame['collision_year'].isin(reference_years)]
    observed = frame[frame['collision_year'].isin(comparison_years)]
    rows = []
    for column in features:
        psi = population_stability_index(
            reference[column], observed[column],
            categorical=(FEATURE_KIND.get(column) == 'categorical'))
        rows.append({'feature': column, 'psi': psi, 'drift_level': drift_level(psi)})
    return (pd.DataFrame(rows)
            .sort_values('psi', ascending=False)
            .reset_index(drop=True))


def plot_feature_drift(frame: pd.DataFrame, path: Path,
                       reference_years: tuple[int, ...] = (2021, 2022, 2023),
                       comparison_years: tuple[int, ...] = (2025,)) -> pd.DataFrame:
    '''Plot PSI for each inspected model input between training and deployment.'''
    scores = feature_drift_scores(frame, list(reference_years), list(comparison_years))
    ref_label = '-'.join(str(year) for year in reference_years)
    cmp_label = '-'.join(str(year) for year in comparison_years)
    colours = {'stable': GREY, 'moderate': ORANGE, 'significant': SIGNIFICANT_RED}
    ordered = scores.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ordered['feature'], ordered['psi'],
            color=[colours[level] for level in ordered['drift_level']], edgecolor='none')
    for y, psi in enumerate(ordered['psi']):
        ax.text(psi, y, f'{psi:.3f}', va='center', ha='left', fontsize=8.5, color=DARK_GREY)
    for level in ['stable', 'moderate', 'significant']:
        ax.plot([], [], color=colours[level], linewidth=4, label=level.capitalize())
    ax.legend(title='Drift severity', frameon=False, loc='lower right')
    ax.set(title='Most model inputs drift only modestly between training and deployment',
           xlabel=f'Population Stability Index ({ref_label} training vs {cmp_label} deployment)',
           ylabel='Model feature')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path,
          f'PSI pools {ref_label} training inputs versus {cmp_label} inputs; '
          'missing values excluded and rare categories pooled.')
    return scores


def plot_predicted_risk_drift(train: pd.DataFrame, validation: pd.DataFrame,
                              test: pd.DataFrame, settings: dict, random_state: int,
                              path: Path) -> pd.DataFrame:
    '''Refit LightGBM on the training years and compare predicted risk over time.'''
    train_x, train_y = build_features(train), make_target(train)
    model = make_pipeline(train_x, random_state, settings, 'lightgbm')
    model.fit(train_x, train_y)
    rows = []
    distributions = {}
    for period, period_frame in [('2024 validation', validation), ('2025 test', test)]:
        x, y = build_features(period_frame), make_target(period_frame)
        probabilities = model.predict_proba(x)[:, 1]
        distributions[period] = probabilities
        rows.append({
            'period': period, 'n': len(x),
            'mean_pred': float(probabilities.mean()), 'std_pred': float(probabilities.std()),
            'prevalence': float(y.mean()),
            'average_precision': float(average_precision_score(y, probabilities)),
            'brier_score': float(brier_score_loss(y, probabilities)),
        })
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    palette = {'2024 validation': ACCENT, '2025 test': ORANGE}
    for period, colour in palette.items():
        row = summary[summary['period'] == period].iloc[0]
        sns.kdeplot(distributions[period], clip=(0, 1), fill=True, alpha=0.3,
                    linewidth=2, color=colour, label=period, ax=ax)
        ax.axvline(row['prevalence'], color=colour, linestyle='--', linewidth=1.5,
                   label=f"{period.split()[0]} observed KSI {row['prevalence']:.1%}")
    annotation = '\n'.join(
        f"{row['period']}: AP {row['average_precision']:.3f} · Brier {row['brier_score']:.3f}"
        for _, row in summary.iterrows())
    ax.text(0.98, 0.98, annotation, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor=DARK_GREY, alpha=0.9))
    ax.set(title='Predicted-risk distributions remain stable as the model is deployed to 2025',
           xlabel='Predicted KSI probability', ylabel='Density')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path,
          'LightGBM refit on 2021-2023; probabilities for 2024 validation and 2025 test. '
          'Dashed lines show observed KSI share; KDE clipped to [0, 1].')
    return summary
