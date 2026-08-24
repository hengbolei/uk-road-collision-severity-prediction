'''Create optional diagnostics outside the core figure story.'''

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score
from road_severity.data import build_features, make_target
from road_severity.modeling import make_pipeline
from road_severity.processed_analysis import LABELS, wilson_interval

ACCENT, ORANGE, GREY, DARK_GREY = '#0072B2', '#E69F00', '#B8C2CC', '#4B5563'
SOURCE = 'Source: UK Department for Transport road collision data, 2021-2025.'
sns.set_theme(style='whitegrid', rc={
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#E5E7EB', 'figure.facecolor': 'white',
})


def _save(fig: plt.Figure, path: Path, footer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.008, f'{footer}  {SOURCE}', fontsize=8, color=DARK_GREY)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_monthly_time_series(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    '''Plot monthly collision volume and KSI share with 12-month smoothers.'''
    monthly = (frame.assign(month=frame['date'].dt.to_period('M').dt.to_timestamp())
        .groupby('month', as_index=False)
        .agg(collisions=('ksi', 'size'), ksi_collisions=('ksi', 'sum'), ksi_rate=('ksi', 'mean'))
        .sort_values('month'))
    monthly['ksi_ci_low'], monthly['ksi_ci_high'] = wilson_interval(monthly['ksi_collisions'], monthly['collisions'])
    monthly['collisions_12m'] = monthly['collisions'].rolling(12, min_periods=6).mean()
    monthly['ksi_rate_12m'] = (monthly['ksi_collisions'].rolling(12, min_periods=6).sum()
                               / monthly['collisions'].rolling(12, min_periods=6).sum())
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(monthly['month'], monthly['collisions'], color=GREY, linewidth=1.1, label='Monthly')
    axes[0].plot(monthly['month'], monthly['collisions_12m'], color=ACCENT, linewidth=2.4, label='12-month mean')
    axes[0].set(title='Monthly collision volume reveals seasonality behind the annual totals', ylabel='Reported collisions')
    axes[0].legend(frameon=False, ncol=2, loc='upper right')
    axes[1].fill_between(monthly['month'], monthly['ksi_ci_low'], monthly['ksi_ci_high'], color=GREY, alpha=0.28)
    axes[1].plot(monthly['month'], monthly['ksi_rate'], color=GREY, linewidth=1.0)
    axes[1].plot(monthly['month'], monthly['ksi_rate_12m'], color=ORANGE, linewidth=2.4, label='12-month pooled KSI share')
    axes[1].set(xlabel='Month', ylabel='KSI share')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, loc='upper right')
    _save(fig, path, 'Monthly observations; shading is the 95% Wilson CI. Rates are not exposure-normalised.')
    return monthly


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    '''Return bias-corrected Cramer's V for two categorical variables.'''
    table = pd.crosstab(left, right)
    n = table.to_numpy().sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(1, keepdims=True) @ observed.sum(0, keepdims=True) / n
    chi2 = np.divide((observed - expected) ** 2, expected, out=np.zeros_like(expected), where=expected > 0).sum()
    rows, cols = table.shape
    phi2 = max(0.0, chi2 / n - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
    corrected_rows = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    corrected_cols = cols - ((cols - 1) ** 2) / max(n - 1, 1)
    denominator = min(corrected_cols - 1, corrected_rows - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    '''Return correlation ratio eta for a categorical and a numeric variable.'''
    valid = categories.notna() & values.notna()
    categories, values = categories[valid], values[valid].astype(float)
    if values.empty or values.var() == 0:
        return 0.0
    grand_mean = values.mean()
    numerator = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in values.groupby(categories))
    denominator = ((values - grand_mean) ** 2).sum()
    return float(np.sqrt(numerator / denominator)) if denominator else 0.0


def mixed_association_matrix(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    '''Build a symmetric matrix without treating category codes as quantities.'''
    columns = [column for column in [*numeric, *categorical] if column in frame]
    result = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            if left in numeric and right in numeric:
                value = abs(frame[left].corr(frame[right], method='spearman'))
            elif left not in numeric and right not in numeric:
                value = cramers_v(frame[left], frame[right])
            else:
                category, number = (right, left) if left in numeric else (left, right)
                value = correlation_ratio(frame[category], frame[number])
            result.loc[left, right] = result.loc[right, left] = 0.0 if pd.isna(value) else value
    return result


def plot_association_heatmap(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    '''Plot selected pre-collision mixed-type associations.'''
    numeric = ['number_of_vehicles', 'speed_limit']
    categorical = ['ksi', 'urban_or_rural_area', 'road_type', 'light_conditions',
                   'weather_conditions', 'road_surface_conditions',
                   'junction_detail_unified']
    columns = [column for column in [*numeric, *categorical] if column in frame]
    sample = frame[columns].sample(min(len(frame), 100_000), random_state=42)
    matrix = mixed_association_matrix(sample, numeric, categorical)
    display_names = {
        'number_of_vehicles': 'Number of vehicles',
        'speed_limit': 'Speed limit',
        'hour': 'Hour',
        'ksi': 'KSI outcome',
        'urban_or_rural_area': 'Urban / rural area',
        'road_type': 'Road type',
        'first_road_class': 'Road class',
        'light_conditions': 'Light conditions',
        'weather_conditions': 'Weather conditions',
        'road_surface_conditions': 'Road surface',
        'day_of_week': 'Day of week',
        'month': 'Month',
        'junction_detail_unified': 'Junction detail',
        'trunk_road_flag': 'Trunk road',
    }
    labels = [display_names.get(column, column.replace('_', ' ').title()) for column in matrix]
    fig, ax = plt.subplots(figsize=(12.5, 10))
    sns.heatmap(matrix, mask=np.triu(np.ones_like(matrix, dtype=bool), k=1),
                cmap=sns.light_palette(ACCENT, as_cmap=True), vmin=0, vmax=1,
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={'label': 'Association strength'},
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Road context contains clusters of related, not interchangeable, features')
    ax.set_xticklabels(labels, rotation=42, ha='right', rotation_mode='anchor', fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    _save(fig, path, 'Absolute Spearman rho (numeric), Cramers V (categorical), and eta (mixed); association is not causation.')
    return matrix


def plot_vehicle_distribution(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    '''Compare involved-vehicle distributions across severity classes.'''
    plot_data = frame[['collision_severity', 'number_of_vehicles']].dropna().copy()
    upper = float(plot_data['number_of_vehicles'].quantile(0.99))
    plot_data = plot_data[plot_data['number_of_vehicles'] <= upper]
    plot_data['severity'] = plot_data['collision_severity'].map(LABELS['severity'])
    order = ['Slight', 'Serious', 'Fatal']
    fig, ax = plt.subplots(figsize=(9, 5.4))
    sns.violinplot(data=plot_data, x='severity', y='number_of_vehicles', order=order,
                   inner=None, cut=0, density_norm='width', color='#C9DCEB', linewidth=0.8, ax=ax)
    sns.boxplot(data=plot_data, x='severity', y='number_of_vehicles', order=order,
                width=0.18, showfliers=False, color='white', boxprops={'zorder': 3}, ax=ax)
    medians = plot_data.groupby('severity', observed=True)['number_of_vehicles'].median().reindex(order)
    for index, value in enumerate(medians):
        ax.text(index, value + 0.14, f'median {value:.0f}', ha='center', color=ACCENT, fontsize=9)
    ax.set(title='Vehicle-count distributions remain concentrated across all severity classes',
           xlabel='Recorded collision severity', ylabel='Number of vehicles')
    _save(fig, path, f'Distribution shown through the 99th percentile ({upper:g} vehicles); boxes show IQR and median.')
    return plot_data.groupby('severity', observed=True)['number_of_vehicles'].describe().reset_index()


def plot_parallel_profiles(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    '''Plot standardised profiles for common urban/rural road-type contexts.'''
    data = frame[
        frame['urban_or_rural_area'].isin([1, 2])
        & frame['road_type'].isin(LABELS['road_type'])
        & frame['first_road_class'].isin(LABELS['first_road_class'])
    ].copy()
    data['dark'] = data['light_conditions'].isin([4, 5, 6]).astype(float)
    data['adverse_weather'] = data['weather_conditions'].isin([2, 3, 4, 5, 6, 7]).astype(float)
    data['at_junction'] = (~data['junction_detail_unified'].isin([0, 99])
                           & data['junction_detail_unified'].notna()).astype(float)
    data['collision_type'] = np.where(data['ksi'].eq(1), 'KSI', 'Slight')
    data['day_type'] = np.where(data['day_of_week'].isin([1, 7]), 'Weekend', 'Weekday')
    profiles = data.groupby(
        ['urban_or_rural_area', 'road_type', 'first_road_class',
         'day_type', 'collision_type'], as_index=False
    ).agg(
        collisions=('ksi', 'size'),
        median_speed=('speed_limit', 'median'), dark_share=('dark', 'mean'),
        adverse_weather_share=('adverse_weather', 'mean'),
        junction_share=('at_junction', 'mean'),
        mean_vehicles=('number_of_vehicles', 'mean'))
    paired_minimum = profiles.groupby(
        ['urban_or_rural_area', 'road_type', 'first_road_class', 'day_type']
    )['collisions'].transform('min')
    profiles = profiles[paired_minimum >= 40].copy().reset_index(drop=True)
    profiles['profile'] = (profiles['urban_or_rural_area'].map({1: 'Urban', 2: 'Rural'})
                           + ' - ' + profiles['road_type'].map(LABELS['road_type'])
                           + ' - ' + profiles['first_road_class'].map(LABELS['first_road_class'])
                           + ' - ' + profiles['day_type'])
    metrics = ['dark_share', 'adverse_weather_share', 'median_speed',
               'junction_share', 'mean_vehicles']
    axis_labels = ['Darkness', 'Adverse weather', 'Median speed',
                   'At junction', 'Mean vehicles', 'Collision type']
    scaled = profiles[metrics].copy()
    for column in metrics:
        span = scaled[column].max() - scaled[column].min()
        scaled[column] = (scaled[column] - scaled[column].min()) / span if span else 0.5
    scaled['collision_type'] = profiles['collision_type'].map({'Slight': 0.0, 'KSI': 1.0})
    colors = {'Slight': ACCENT, 'KSI': '#D62728'}
    fig, ax = plt.subplots(figsize=(14, 7.4))
    x = np.arange(len(axis_labels))
    draw_order = profiles.sort_values(
        'collision_type',
        key=lambda values: values.map({'Slight': 0, 'KSI': 1}),
    ).index
    for index in draw_order:
        row = scaled.loc[index]
        collision_type = profiles.loc[index, 'collision_type']
        values = [*row[metrics], row['collision_type']]
        is_ksi = collision_type == 'KSI'
        ax.plot(x, values, marker='o', markersize=2.4, linewidth=1.5,
                alpha=1.0 if is_ksi else 0.28,
                color=colors[collision_type], zorder=3 if is_ksi else 2)
    for position in x:
        ax.axvline(position, color='#CBD5E1', linewidth=0.9, zorder=0)
    for collision_type, color in colors.items():
        ax.plot([], [], color=color, linewidth=2.5, label=collision_type)
    ax.text(x[-1] + 0.05, 0, 'Slight', va='center', color=ACCENT, fontsize=9)
    ax.text(x[-1] + 0.05, 1, 'KSI', va='center', color='#D62728', fontsize=9)
    ax.set(xticks=x, xticklabels=axis_labels, xlim=(-0.1, len(axis_labels) - 0.55),
           ylim=(-0.05, 1.05), ylabel='Within-metric scale (low to high)',
           title='Expanded road and time contexts reveal varied Slight and KSI profiles')
    ax.legend(title='Collision type', frameon=False, loc='upper left', ncol=2)
    ax.grid(axis='x', visible=False)
    _save(fig, path, 'Each context has at least 40 Slight and 40 KSI collisions; numeric axes are independently scaled.')
    return profiles


def plot_temporal_learning_curve(train: pd.DataFrame, validation: pd.DataFrame,
                                 settings: dict, random_state: int,
                                 path: Path) -> pd.DataFrame:
    '''Fit chronological training fractions and compare train versus future-year AP.'''
    ordered = train.sort_values('date')
    validation_x, validation_y = build_features(validation), make_target(validation)
    rows = []
    for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
        size = max(100, int(len(ordered) * fraction))
        subset = ordered.iloc[:size]
        train_x, train_y = build_features(subset), make_target(subset)
        model = make_pipeline(train_x, random_state, settings, 'lightgbm')
        model.fit(train_x, train_y)
        rows.append({
            'fraction': fraction, 'training_rows': size, 'end_date': subset['date'].max(),
            'train_average_precision': average_precision_score(
                train_y, model.predict_proba(train_x)[:, 1]),
            'validation_average_precision': average_precision_score(
                validation_y, model.predict_proba(validation_x)[:, 1]),
        })
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(curve['training_rows'], curve['train_average_precision'], marker='o',
            linewidth=2.2, color=DARK_GREY, label='Training AP')
    ax.plot(curve['training_rows'], curve['validation_average_precision'], marker='o',
            linewidth=2.4, color=ACCENT, label='2024 validation AP')
    ax.axhline(validation_y.mean(), color=ORANGE, linestyle='--', linewidth=1.5,
               label=f'2024 prevalence ({validation_y.mean():.1%})')
    ax.set(title='Chronological learning curve tests whether more history improves future-year ranking',
           xlabel='Chronologically accumulated training collisions',
           ylabel='Average precision')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    _save(fig, path, 'Training data end in 2023; validation is fixed to 2024. Fractions accumulate from the earliest date.')
    return curve
