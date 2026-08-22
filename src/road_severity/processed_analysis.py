'''
Summarise validated collision data and create presentation-ready figures.

This module builds KSI rate tables with uncertainty intervals, produces temporal,
categorical, interaction, and spatial charts, and returns a catalogue of the most
useful figures for reporting.
'''

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

ACCENT = "#39728C"
ACCENT_DARK = "#27566B"
ORANGE = "#C4773B"
HEAT_RED = "#B5524B"
GREY = "#B8BDC2"
DARK_GREY = "#555B61"
GRID_GREY = "#E5E7E9"
KSI_AXIS_MAX = 0.40
SPATIAL_MIN_COLLISIONS = 200
SOURCE = "Source: UK Department for Transport road collision data, 2021-2025."

LABELS = {
    "severity": {1: "Fatal", 2: "Serious", 3: "Slight"},
    "area": {1: "Urban", 2: "Rural", 3: "Unallocated"},
    "light": {1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit", 6: "Darkness - no lighting", 7: "Darkness - lighting unknown"},
    "road_type": {1: "Roundabout", 2: "One-way street", 3: "Dual carriageway", 6: "Single carriageway", 7: "Slip road", 9: "Unknown", 12: "One-way/slip road"},
    "weather": {1: "Fine, no high winds", 2: "Rain, no high winds", 3: "Snow, no high winds", 4: "Fine with high winds", 5: "Rain with high winds", 6: "Snow with high winds", 7: "Fog or mist", 8: "Other", 9: "Unknown"},
    "surface": {1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice", 5: "Flood", 6: "Oil or diesel", 7: "Mud", 9: "Unknown"},
    "weekday": {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"},
    "month": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"},
    "junction": {0: "Not at a junction", 13: "T or staggered junction", 16: "Crossroads", 17: ">4-arm junction", 18: "Private drive/entrance", 19: "Other junction", 99: "Unknown"},
    "pedestrian": {0: "No crossing within 50m", 11: "School crossing patrol", 12: "Other human control", 13: "Zebra crossing", 14: "Pedestrian light crossing", 15: "Pedestrian phase at signals", 16: "Footbridge or subway", 17: "Central refuge", 99: "Unknown"},
    "hazards": {0: "None", 11: "Defective traffic signals", 12: "Defective signs/markings", 13: "Roadworks", 14: "Oil or diesel", 15: "Mud", 16: "Vehicle load", 17: "Object in carriageway", 18: "Previous collision", 19: "Pedestrian in carriageway", 20: "Animal in carriageway", 21: "Defective road surface", 99: "Unknown"},
    "first_road_class": {1: "Motorway", 2: "A(M)", 3: "A road", 4: "B road", 5: "C road", 6: "Unclassified"},
    "trunk": {1: "Trunk road", 2: "Non-trunk road"},
}


def wilson_interval(successes: pd.Series | np.ndarray, totals: pd.Series | np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calculate Wilson confidence bounds for one or more observed proportions.
    '''
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    p = np.divide(successes, totals, out=np.zeros_like(successes), where=totals > 0)
    denominator = 1 + z**2 / totals
    centre = (p + z**2 / (2 * totals)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * totals)) / totals) / denominator
    return centre - margin, centre + margin


def proportion_summary(frame: pd.DataFrame, column: str, labels: dict | None = None, unknown_codes: set | None = None) -> pd.DataFrame:
    '''
    Summarise collision counts, KSI shares, intervals, and data status by category.
    '''
    summary = frame.groupby(column, dropna=False)["ksi"].agg(collisions="size", ksi_collisions="sum", ksi_rate="mean").reset_index()
    low, high = wilson_interval(summary["ksi_collisions"], summary["collisions"])
    summary["ci_low"] = low
    summary["ci_high"] = high
    summary["label"] = summary[column].map(labels or {}).astype("object")
    summary.loc[summary[column].isna(), "label"] = "Missing"
    summary["label"] = summary["label"].fillna(summary[column].astype("string"))
    unknown_codes = unknown_codes or set()
    summary["status"] = np.select(
        [summary[column].isna(), summary[column].isin(unknown_codes)],
        ["missing", "unknown"], default="observed",
    )
    return summary.sort_values("ksi_rate", ascending=False).reset_index(drop=True)


def build_all_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    '''
    Build every severity, time, road, and environment table used by Stage 2.
    '''
    severity = frame["collision_severity"].value_counts().sort_index().rename_axis("severity_code").reset_index(name="collisions")
    severity["severity"] = severity["severity_code"].map(LABELS["severity"])
    severity["share"] = severity["collisions"] / severity["collisions"].sum()

    annual = frame.groupby("collision_year", as_index=False).agg(
        collisions=("ksi", "size"), ksi_collisions=("ksi", "sum"), ksi_rate=("ksi", "mean"),
        injury_based_collisions=("collision_injury_based", "sum"), injury_based_share=("collision_injury_based", "mean"),
    )
    adjusted_ksi = frame["collision_severity"].eq(1).astype(float) + frame["collision_adjusted_severity_serious"]
    adjusted = adjusted_ksi.groupby(frame["collision_year"]).agg(["sum", "mean"]).reset_index()
    adjusted.columns = ["collision_year", "adjusted_ksi_expected", "adjusted_ksi_rate"]
    annual = annual.merge(adjusted, on="collision_year")
    annual["ksi_ci_low"], annual["ksi_ci_high"] = wilson_interval(annual["ksi_collisions"], annual["collisions"])
    annual["injury_based_ci_low"], annual["injury_based_ci_high"] = wilson_interval(annual["injury_based_collisions"], annual["collisions"])

    tables = {
        "severity_summary": severity,
        "annual_reporting_summary": annual,
        "hourly_summary": proportion_summary(frame, "hour"),
        "ksi_by_speed_limit": proportion_summary(frame, "speed_limit"),
        "ksi_by_area": proportion_summary(frame, "urban_or_rural_area", LABELS["area"], {3}),
        "ksi_by_light": proportion_summary(frame, "light_conditions", LABELS["light"], {7}),
        "ksi_by_road_type": proportion_summary(frame, "road_type", LABELS["road_type"], {9}),
        "ksi_by_weather": proportion_summary(frame, "weather_conditions", LABELS["weather"], {9}),
        "ksi_by_road_surface": proportion_summary(frame, "road_surface_conditions", LABELS["surface"], {9}),
        "ksi_by_month": proportion_summary(frame, "month", LABELS["month"]),
        "ksi_by_weekday": proportion_summary(frame, "day_of_week", LABELS["weekday"]),
        "ksi_by_junction": proportion_summary(frame, "junction_detail_unified", LABELS["junction"], {99}),
        "ksi_by_pedestrian_crossing": proportion_summary(frame, "pedestrian_crossing_unified", LABELS["pedestrian"], {99}),
        "ksi_by_carriageway_hazard": proportion_summary(frame, "carriageway_hazards_unified", LABELS["hazards"], {99}),
        "ksi_by_first_road_class": proportion_summary(frame, "first_road_class", LABELS["first_road_class"]),
        "ksi_by_trunk_road": proportion_summary(frame, "trunk_road_flag", LABELS["trunk"]),
    }
    return tables


def _style() -> None:
    '''
    Apply the shared colour-blind-safe visual theme to all generated figures.
    '''
    sns.set_theme(style="whitegrid", font_scale=1.0, rc={
        "axes.edgecolor": DARK_GREY, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID_GREY, "grid.linewidth": 0.7,
        "axes.titleweight": "normal", "figure.facecolor": "white",
    })


def _save(fig: plt.Figure, path: Path, footer: str | None = None, apply_tight_layout: bool = True) -> None:
    '''
    Lay out, annotate, save, and close a figure using consistent export settings.
    '''
    footer = f"{footer}  {SOURCE}" if footer else SOURCE
    if footer:
        fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=8, color=DARK_GREY)
    if apply_tight_layout:
        fig.tight_layout(rect=(0, 0.035, 1, 1) if footer else None)
    else:
        fig.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.92)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_severity(summary: pd.DataFrame, path: Path) -> None:
    '''Plot severity composition using direct labels and one KSI accent hue.'''
    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = 0.0
    styles = [(ACCENT, "///"), (ACCENT, None), (GREY, None)]
    for row, (color, hatch) in zip(summary.itertuples(), styles):
        ax.barh([0], [row.share], left=left, color=color, height=0.5, hatch=hatch)
        label = f"{row.severity}\n{row.share:.1%}\n(n={row.collisions:,})"
        if row.share < 0.05:
            ax.annotate(label, xy=(left + row.share / 2, 0.24), xytext=(left + 0.035, 0.60),
                        ha="left", va="bottom", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": DARK_GREY, "linewidth": 0.8})
        else:
            text_color = "white" if row.severity != "Slight" else "#222222"
            ax.text(left + row.share / 2, 0, label, ha="center", va="center", color=text_color, fontsize=9)
        left += row.share
    ax.set(xlim=(0, 1), ylim=(-0.45, 0.95), yticks=[], xlabel="Share of reported collisions (%)",
           title="Three in four reported collisions were recorded as slight")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    _save(fig, path, "Denominator: reported personal-injury collisions, 2021-2025.")

def plot_annual(summary: pd.DataFrame, path: Path) -> None:
    '''Combine annual collision volume and reporting-method trends on one time axis.'''
    fig, ax = plt.subplots(figsize=(9, 5.5))
    peak_index = summary["collisions"].idxmax()
    ax.bar(
        summary["collision_year"], summary["collisions"], width=0.62,
        color=[ACCENT if index == peak_index else GREY for index in summary.index],
    )
    ax.set(
        title="Collision volume stayed stable while injury-based reporting expanded",
        xlabel="Year", ylabel="Reported collisions",
        ylim=(0, 120_000),
    )
    ax.set_xticks(summary["collision_year"])
    ax.yaxis.set_major_locator(LinearLocator(5))
    peak = summary.loc[peak_index]
    ax.annotate(
        f"{int(peak['collisions']):,}",
        (peak["collision_year"], peak["collisions"]), xytext=(0, 6),
        textcoords="offset points", ha="center", va="bottom", fontsize=9, color=ACCENT_DARK,
    )
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)

    rate_ax = ax.twinx()
    rate_ax.plot(summary["collision_year"], summary["injury_based_share"], marker="o", color=ORANGE, linewidth=2.2)
    rate_ax.fill_between(
        summary["collision_year"], summary["injury_based_ci_low"], summary["injury_based_ci_high"],
        color=ORANGE, alpha=0.12,
    )
    rate_ax.set(ylabel="Injury-based severity reporting (%)", ylim=(0, 1))
    rate_ax.yaxis.set_major_locator(LinearLocator(5))
    rate_ax.yaxis.set_major_formatter(PercentFormatter(1))
    rate_ax.grid(False)
    rate_ax.spines["right"].set_visible(True)
    rate_ax.spines["right"].set_color(ORANGE)
    rate_ax.tick_params(axis="y", colors=ORANGE)
    rate_ax.yaxis.label.set_color(ORANGE)
    last = summary.iloc[-1]
    rate_ax.annotate(
        f"Injury-based reporting {last['injury_based_share']:.1%}",
        (last["collision_year"], last["injury_based_share"]), xytext=(0, 12),
        textcoords="offset points", ha="center", va="bottom", fontsize=9, color=ORANGE,
        annotation_clip=False,
    )
    _save(fig, path, "Bars: annual reported collisions. Line and shading: injury-based reporting share and 95% Wilson CI.")

def plot_hourly(summary: pd.DataFrame, path: Path) -> None:
    '''Combine hourly collision volume and KSI share on one time axis.'''
    summary = summary.sort_values("hour")
    fig, ax = plt.subplots(figsize=(10, 5.7))
    peak_hour = summary.loc[summary["collisions"].idxmax(), "hour"]
    ax.bar(summary["hour"], summary["collisions"], color=[DARK_GREY if hour == peak_hour else GREY for hour in summary["hour"]], width=0.78)
    ax.set(
        title="Collision volume peaks in the afternoon, but severity peaks overnight",
        xlabel="Hour of day", ylabel="Reported collisions", xticks=range(0, 24, 2),
        ylim=(0, 48_000),
    )
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)

    rate_ax = ax.twinx()
    rate_ax.plot(summary["hour"], summary["ksi_rate"], marker="o", markersize=4, color=ACCENT, linewidth=2.1)
    rate_ax.fill_between(summary["hour"], summary["ci_low"], summary["ci_high"], color=ACCENT, alpha=0.12)
    rate_ax.set(ylabel="KSI collisions (% of reported collisions)", ylim=(0, KSI_AXIS_MAX))
    rate_ax.yaxis.set_major_locator(LinearLocator(5))
    rate_ax.yaxis.set_major_formatter(PercentFormatter(1))
    rate_ax.grid(False)
    rate_ax.spines["right"].set_visible(True)
    rate_ax.spines["right"].set_color(ACCENT)
    rate_ax.tick_params(axis="y", colors=ACCENT)
    rate_ax.yaxis.label.set_color(ACCENT)
    peak_ksi = summary.loc[summary["ksi_rate"].idxmax()]
    rate_ax.annotate(
        f"Highest KSI share\n{int(peak_ksi['hour']):02d}:00  {peak_ksi['ksi_rate']:.1%}",
        (peak_ksi["hour"], peak_ksi["ksi_rate"]), xytext=(10, 8),
        textcoords="offset points", fontsize=9, color=ACCENT_DARK,
    )
    _save(fig, path, "Denominator: reported collisions in each hour. Shading: 95% Wilson CI; this is not exposure-normalised risk.")

def plot_proportion(summary: pd.DataFrame, path: Path, title: str, min_n: int = 100, include_unknown: bool = False) -> None:
    '''Plot category KSI shares; ordered categories retain their natural order.'''
    shown = summary[summary["collisions"].ge(min_n)].copy()
    if not include_unknown:
        shown = shown[shown["status"].eq("observed")]
    labels = set(shown["label"].astype(str))
    natural_orders = [list(LABELS["month"].values()), list(LABELS["weekday"].values()), ["20", "30", "40", "50", "60", "70"]]
    order = next((candidate for candidate in natural_orders if labels.issubset(set(candidate))), None)
    if order:
        shown["label"] = pd.Categorical(shown["label"].astype(str), categories=order, ordered=True)
        shown = shown.sort_values("label", ascending=False)
    else:
        shown = shown.sort_values("ksi_rate")
    fig_height = max(4.0, 0.48 * len(shown) + 1.7)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    highlight = shown["ksi_rate"].idxmax()
    ax.barh(shown["label"], shown["ksi_rate"], color=[ACCENT if i == highlight else GREY for i in shown.index])
    xerr = np.vstack([shown["ksi_rate"] - shown["ci_low"], shown["ci_high"] - shown["ksi_rate"]])
    ax.errorbar(shown["ksi_rate"], shown["label"], xerr=xerr, fmt="none", ecolor=DARK_GREY, capsize=2, linewidth=1)
    for row in shown.itertuples():
        ax.text(min(row.ksi_rate + 0.006, KSI_AXIS_MAX - 0.002), row.label,
                f"{row.ksi_rate:.1%}  n={row.collisions:,}", va="center", ha="left", fontsize=8)
    ax.set(title=title, xlabel="KSI collisions (% of reported collisions)", ylabel="", xlim=(0, KSI_AXIS_MAX))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    _save(fig, path, "Whiskers: 95% Wilson CI. Unknown/missing categories excluded from the conclusion chart but retained in its CSV table.")

def cross_summary(frame: pd.DataFrame, row: str, column: str, row_labels: dict, column_labels: dict, min_n: int = 100) -> pd.DataFrame:
    '''
    Build a labelled two-factor KSI summary for interaction heatmaps.
    '''
    grouped = frame.groupby([row, column], dropna=False, observed=True)["ksi"].agg(collisions="size", ksi_collisions="sum", ksi_rate="mean").reset_index()
    grouped = grouped[grouped["collisions"].ge(min_n)].copy()
    grouped["row_label"] = grouped[row].map(row_labels).fillna(grouped[row].astype("string"))
    grouped["column_label"] = grouped[column].map(column_labels).fillna(grouped[column].astype("string"))
    return grouped


def plot_heatmap(summary: pd.DataFrame, path: Path, title: str, base_color: str = ACCENT) -> None:
    '''Render a naturally ordered, annotated sequential heatmap.'''
    rates = summary.pivot(index="row_label", columns="column_label", values="ksi_rate")
    counts = summary.pivot(index="row_label", columns="column_label", values="collisions")
    annotations = rates.copy().astype(object)
    for row in rates.index:
        for column in rates.columns:
            rate, count = rates.loc[row, column], counts.loc[row, column]
            annotations.loc[row, column] = "" if pd.isna(rate) else f"{rate:.1%}\nn={int(count):,}"
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.75 * len(rates))))
    cmap = sns.light_palette(base_color, as_cmap=True)
    sns.heatmap(rates, annot=annotations, fmt="", cmap=cmap, vmin=0, vmax=KSI_AXIS_MAX,
                linewidths=1, linecolor="white", cbar_kws={"label": "KSI collisions (% of reported collisions)"}, ax=ax)
    ax.set(title=title, xlabel="", ylabel="")
    ax.collections[0].colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path, "Each cell shows KSI share and collision count; cells with fewer than 100 collisions are omitted.")

def plot_spatial_hex(frame: pd.DataFrame, path: Path, table_path: Path) -> None:
    '''Overlay collision density and sample-filtered KSI share on one spatial view.'''
    geo = frame.dropna(subset=["longitude", "latitude", "ksi"])
    extent = [geo["longitude"].min(), geo["longitude"].max(), geo["latitude"].min(), geo["latitude"].max()]
    fig = plt.figure(figsize=(9.2, 8.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.075], wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    colorbar_grid = grid[0, 1].subgridspec(2, 1, hspace=0.38)
    count_cax = fig.add_subplot(colorbar_grid[0, 0])
    rate_cax = fig.add_subplot(colorbar_grid[1, 0])
    counts = ax.hexbin(
        geo["longitude"], geo["latitude"], gridsize=48, mincnt=1, bins="log",
        extent=extent, cmap="Greys", alpha=0.72,
    )
    rates = ax.hexbin(
        geo["longitude"], geo["latitude"], C=geo["ksi"],
        reduce_C_function=np.mean, gridsize=48, mincnt=SPATIAL_MIN_COLLISIONS,
        extent=extent, alpha=0,
    )
    rate_offsets = rates.get_offsets().copy()
    rate_values = rates.get_array().copy()
    rates.remove()
    severity = ax.scatter(
        rate_offsets[:, 0], rate_offsets[:, 1], c=rate_values,
        cmap=sns.light_palette(HEAT_RED, as_cmap=True),
        vmin=float(rate_values.min()), vmax=float(rate_values.max()),
        marker="h", s=29, linewidths=0.25, edgecolors="white", alpha=0.92,
    )
    ax.set(
        title="Severity hotspots do not simply follow collision density",
        xlabel="Longitude", ylabel="Latitude",
    )
    ax.set_aspect(1 / np.cos(np.deg2rad(geo["latitude"].mean())))
    count_bar = fig.colorbar(counts, cax=count_cax)
    count_cax.set_title("Collision\ncount", fontsize=9, color=DARK_GREY, pad=8)
    count_bar.set_label("Log scale", fontsize=8, color=DARK_GREY)
    rate_bar = fig.colorbar(severity, cax=rate_cax)
    rate_cax.set_title("KSI share\n(n >= 200)", fontsize=9, color=HEAT_RED, pad=8)
    rate_bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(
        fig, path,
        f"Grey hexagons show collision density; red hexagons show KSI share where at least {SPATIAL_MIN_COLLISIONS} collisions were recorded. "
        "Results are not adjusted for traffic exposure or population.",
        apply_tight_layout=False,
    )

    count_table = pd.DataFrame(counts.get_offsets(), columns=["longitude", "latitude"])
    count_table["collisions"] = counts.get_array()
    rate_table = pd.DataFrame(rate_offsets, columns=["longitude", "latitude"])
    rate_table["ksi_rate"] = rate_values
    spatial = count_table.merge(rate_table, on=["longitude", "latitude"], how="left")
    spatial.to_csv(table_path, index=False)

def create_all_figures(frame: pd.DataFrame, tables: dict[str, pd.DataFrame], figure_dir: Path, table_dir: Path) -> pd.DataFrame:
    '''
    Generate all Stage 2 figures and return their reporting-priority catalogue.
    '''
    _style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_severity(tables["severity_summary"], figure_dir / "01_severity_composition.png")
    plot_annual(tables["annual_reporting_summary"], figure_dir / "02_annual_reporting_sensitivity.png")
    plot_hourly(tables["hourly_summary"], figure_dir / "03_hourly_volume_and_ksi.png")

    specifications = [
        ("ksi_by_speed_limit", "04_ksi_by_speed_limit.png", "KSI share peaks on 60 mph roads, not 70 mph roads"),
        ("ksi_by_area", "05_ksi_by_area.png", "Rural collisions have a higher recorded KSI share"),
        ("ksi_by_light", "06_ksi_by_light.png", "Unlit darkness has the highest recorded KSI share"),
        ("ksi_by_road_type", "07_ksi_by_road_type.png", "Single carriageways have the highest recorded KSI share"),
        ("ksi_by_weather", "08_ksi_by_weather.png", "Recorded KSI share differs across weather conditions"),
        ("ksi_by_road_surface", "09_ksi_by_road_surface.png", "Recorded KSI share differs across road surfaces"),
        ("ksi_by_month", "10_ksi_by_month.png", "Recorded KSI share is highest in summer months"),
        ("ksi_by_weekday", "11_ksi_by_weekday.png", "Weekend collisions have a higher recorded KSI share"),
        ("ksi_by_junction", "12_ksi_by_junction.png", "KSI share differs across harmonised junction types"),
        ("ksi_by_pedestrian_crossing", "13_ksi_by_pedestrian_crossing.png", "KSI share differs across harmonised crossing facilities"),
        ("ksi_by_carriageway_hazard", "14_ksi_by_carriageway_hazard.png", "Defective road surfaces show the highest KSI share among coded hazards"),
        ("ksi_by_first_road_class", "15_ksi_by_road_class.png", "Recorded KSI share differs by road class"),
        ("ksi_by_trunk_road", "16_ksi_by_trunk_road.png", "Recorded KSI share differs between trunk and non-trunk roads"),
    ]
    for table_name, filename, title in specifications:
        plot_proportion(tables[table_name], figure_dir / filename, title)

    area_labels = {1: "Urban", 2: "Rural"}
    speed_labels = {20: "20 mph", 30: "30 mph", 40: "40 mph", 50: "50 mph", 60: "60 mph", 70: "70 mph"}
    speed_area = cross_summary(frame[frame["urban_or_rural_area"].isin([1, 2])], "urban_or_rural_area", "speed_limit", area_labels, speed_labels)
    speed_area.to_csv(table_dir / "ksi_by_speed_and_area.csv", index=False)
    plot_heatmap(
        speed_area, figure_dir / "17_speed_by_area_heatmap.png",
        "Speed-limit patterns differ substantially between urban and rural roads",
        base_color=HEAT_RED,
    )

    periods = pd.cut(frame["hour"], bins=[-1, 5, 9, 15, 19, 23], labels=["00-05", "06-09", "10-15", "16-19", "20-23"])
    time_light_frame = frame.assign(time_period=periods)
    light_labels = {1: "Daylight", 4: "Dark - lights lit", 5: "Dark - lights unlit", 6: "Dark - no lighting"}
    time_light = cross_summary(time_light_frame[time_light_frame["light_conditions"].isin(light_labels)], "light_conditions", "time_period", light_labels, {})
    time_light.to_csv(table_dir / "ksi_by_time_and_light.csv", index=False)
    plot_heatmap(time_light, figure_dir / "18_time_by_light_heatmap.png", "Time and lighting jointly shape recorded KSI patterns")

    plot_spatial_hex(frame, figure_dir / "19_spatial_hex_analysis.png", table_dir / "spatial_hex_summary.csv")

    catalog = pd.DataFrame([
        ("01_severity_composition.png", True, 1, "Problem framing and class imbalance"),
        ("02_annual_reporting_sensitivity.png", True, 2, "Trend with reporting-method caveat"),
        ("03_hourly_volume_and_ksi.png", True, 3, "Volume-versus-severity contrast"),
        ("04_ksi_by_speed_limit.png", False, 10, "Useful but confounded by area and road type"),
        ("05_ksi_by_area.png", False, 11, "Simple context before interaction analysis"),
        ("06_ksi_by_light.png", True, 4, "Clear environmental contrast"),
        ("07_ksi_by_road_type.png", True, 5, "Clear road-design contrast"),
        ("12_ksi_by_junction.png", True, 6, "Demonstrates harmonised fields"),
        ("14_ksi_by_carriageway_hazard.png", False, 12, "Useful supporting diagnostic"),
        ("17_speed_by_area_heatmap.png", True, 7, "Shows confounding and interaction"),
        ("18_time_by_light_heatmap.png", False, 13, "Supporting interaction analysis"),
        ("19_spatial_hex_analysis.png", True, 8, "Spatial storytelling without claiming exposure risk"),
    ], columns=["figure", "recommended_for_slides", "priority", "purpose"]).sort_values("priority")
    return catalog
