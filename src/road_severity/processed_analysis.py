"""Processed-data summaries and presentation-ready explanatory visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

BLUE = "#4C78A8"
RED = "#E45756"
GREEN = "#59A14F"
PURPLE = "#B279A2"
GREY = "#9D9D9D"
KSI_AXIS_MAX = 0.40
SPATIAL_MIN_COLLISIONS = 200

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
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    p = np.divide(successes, totals, out=np.zeros_like(successes), where=totals > 0)
    denominator = 1 + z**2 / totals
    centre = (p + z**2 / (2 * totals)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * totals)) / totals) / denominator
    return centre - margin, centre + margin


def proportion_summary(frame: pd.DataFrame, column: str, labels: dict | None = None, unknown_codes: set | None = None) -> pd.DataFrame:
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
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)


def _save(fig: plt.Figure, path: Path, footer: str | None = None) -> None:
    if footer:
        fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_severity(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = 0.0
    colors = ["#7A0019", RED, BLUE]
    for row, color in zip(summary.itertuples(), colors):
        ax.barh([0], [row.share], left=left, color=color, height=0.5)
        label = f"{row.severity}\n{row.share:.1%}\n(n={row.collisions:,})"
        if row.share < 0.05:
            ax.annotate(
                label,
                xy=(left + row.share / 2, 0.24),
                xytext=(left + 0.035, 0.60),
                ha="left",
                va="bottom",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
            )
        else:
            ax.text(left + row.share / 2, 0, label, ha="center", va="center", color="white", fontsize=9)
        left += row.share
    ax.set(xlim=(0, 1), ylim=(-0.45, 0.95), yticks=[], xlabel="Share of reported collisions", title="Three in four reported collisions were recorded as slight")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path, "Denominator: reported personal-injury collisions, 2021-2025.")


def plot_annual(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].bar(summary["collision_year"], summary["collisions"], color=BLUE)
    axes[0].set(title="Recorded collision volume remained broadly stable", ylabel="Collisions")
    axes[0].set_ylim(0, summary["collisions"].max() * 1.15)
    for row in summary.itertuples():
        axes[0].text(row.collision_year, row.collisions + 1200, f"{row.collisions:,}", ha="center", fontsize=8)

    axes[1].plot(summary["collision_year"], summary["ksi_rate"], marker="o", color=RED, label="Recorded KSI")
    axes[1].fill_between(summary["collision_year"], summary["ksi_ci_low"], summary["ksi_ci_high"], color=RED, alpha=0.15)
    axes[1].plot(summary["collision_year"], summary["adjusted_ksi_rate"], marker="s", color=PURPLE, label="Severity-adjusted KSI")
    axes[1].set(title="KSI share rose under both recorded and adjusted definitions", ylabel="KSI share", ylim=(0, 0.32))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, ncol=2)

    axes[2].plot(summary["collision_year"], summary["injury_based_share"], marker="o", color=GREEN)
    axes[2].fill_between(summary["collision_year"], summary["injury_based_ci_low"], summary["injury_based_ci_high"], color=GREEN, alpha=0.15)
    axes[2].set(title="Injury-based severity reporting expanded sharply in 2025", ylabel="Injury-based share", xlabel="Year", ylim=(0, 1))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_xticks(summary["collision_year"])
    _save(fig, path, "Adjusted KSI = fatal indicator + DfT adjusted probability of serious severity. Shading: 95% Wilson CI where applicable.")


def plot_hourly(summary: pd.DataFrame, path: Path) -> None:
    summary = summary.sort_values("hour")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(summary["hour"], summary["collisions"], color=BLUE)
    axes[0].set(title="Collision volume peaks during afternoon travel", ylabel="Collisions")
    axes[1].plot(summary["hour"], summary["ksi_rate"], marker="o", color=RED)
    axes[1].fill_between(summary["hour"], summary["ci_low"], summary["ci_high"], color=RED, alpha=0.15)
    axes[1].set(title="KSI share is highest overnight, when collision volume is lower", ylabel="KSI share", xlabel="Hour", ylim=(0, KSI_AXIS_MAX), xticks=range(0, 24, 2))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path, "Denominator: reported collisions in each hour. Shading: 95% Wilson CI; this is not exposure-normalised risk.")


def plot_proportion(summary: pd.DataFrame, path: Path, title: str, min_n: int = 100, include_unknown: bool = False) -> None:
    shown = summary[summary["collisions"].ge(min_n)].copy()
    if not include_unknown:
        shown = shown[shown["status"].eq("observed")]
    shown = shown.sort_values("ksi_rate")
    fig_height = max(4.0, 0.48 * len(shown) + 1.7)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    colors = [GREY if status != "observed" else RED for status in shown["status"]]
    ax.barh(shown["label"], shown["ksi_rate"], color=colors)
    xerr = np.vstack([shown["ksi_rate"] - shown["ci_low"], shown["ci_high"] - shown["ksi_rate"]])
    ax.errorbar(shown["ksi_rate"], shown["label"], xerr=xerr, fmt="none", ecolor="#333333", capsize=2, linewidth=1)
    for row in shown.itertuples():
        ax.text(min(row.ksi_rate + 0.006, KSI_AXIS_MAX - 0.002), row.label, f"{row.ksi_rate:.1%}  n={row.collisions:,}", va="center", ha="left", fontsize=8)
    ax.set(title=title, xlabel="KSI share", ylabel="", xlim=(0, KSI_AXIS_MAX))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path, "Whiskers: 95% Wilson CI. Unknown/missing categories excluded from the conclusion chart but retained in its CSV table.")


def cross_summary(frame: pd.DataFrame, row: str, column: str, row_labels: dict, column_labels: dict, min_n: int = 100) -> pd.DataFrame:
    grouped = frame.groupby([row, column], dropna=False, observed=True)["ksi"].agg(collisions="size", ksi_collisions="sum", ksi_rate="mean").reset_index()
    grouped = grouped[grouped["collisions"].ge(min_n)].copy()
    grouped["row_label"] = grouped[row].map(row_labels).fillna(grouped[row].astype("string"))
    grouped["column_label"] = grouped[column].map(column_labels).fillna(grouped[column].astype("string"))
    return grouped


def plot_heatmap(summary: pd.DataFrame, path: Path, title: str) -> None:
    rates = summary.pivot(index="row_label", columns="column_label", values="ksi_rate")
    counts = summary.pivot(index="row_label", columns="column_label", values="collisions")
    annotations = rates.copy().astype(object)
    for row in rates.index:
        for column in rates.columns:
            rate, count = rates.loc[row, column], counts.loc[row, column]
            annotations.loc[row, column] = "" if pd.isna(rate) else f"{rate:.1%}\nn={int(count):,}"
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.75 * len(rates))))
    sns.heatmap(rates, annot=annotations, fmt="", cmap="YlOrRd", vmin=0.15, vmax=0.40, linewidths=0.5, cbar_kws={"label": "KSI share"}, ax=ax)
    ax.set(title=title, xlabel="", ylabel="")
    ax.collections[0].colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path, "Each cell shows KSI share and collision count; cells with fewer than 100 collisions are omitted.")


def plot_spatial_hex(frame: pd.DataFrame, path: Path, table_path: Path) -> None:
    geo = frame.dropna(subset=["longitude", "latitude", "ksi"])
    extent = [geo["longitude"].min(), geo["longitude"].max(), geo["latitude"].min(), geo["latitude"].max()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)
    counts = axes[0].hexbin(geo["longitude"], geo["latitude"], gridsize=48, mincnt=1, bins="log", extent=extent, cmap="Blues")
    rates = axes[1].hexbin(
        geo["longitude"], geo["latitude"], C=geo["ksi"],
        reduce_C_function=np.mean, gridsize=48, mincnt=SPATIAL_MIN_COLLISIONS,
        extent=extent, cmap="YlOrRd", vmin=0.15, vmax=0.40,
    )
    axes[0].set_title("Recorded collision density")
    axes[1].set_title(f"KSI share (minimum {SPATIAL_MIN_COLLISIONS} collisions)")
    for ax in axes:
        ax.set(xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect(1 / np.cos(np.deg2rad(geo["latitude"].mean())))
    fig.colorbar(counts, ax=axes[0], fraction=0.035, pad=0.03, label="Collision count (log scale)")
    rate_bar = fig.colorbar(rates, ax=axes[1], fraction=0.035, pad=0.03, label="KSI share")
    rate_bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Reported collisions and severity form different spatial patterns", y=0.96)
    _save(
        fig,
        path,
        f"Hexagons aggregate collision coordinates; KSI cells require at least {SPATIAL_MIN_COLLISIONS} collisions. "
        "Results are not adjusted for traffic exposure or population.",
    )

    count_table = pd.DataFrame(counts.get_offsets(), columns=["longitude", "latitude"])
    count_table["collisions"] = counts.get_array()
    rate_table = pd.DataFrame(rates.get_offsets(), columns=["longitude", "latitude"])
    rate_table["ksi_rate"] = rates.get_array()
    spatial = count_table.merge(rate_table, on=["longitude", "latitude"], how="left")
    spatial.to_csv(table_path, index=False)


def create_all_figures(frame: pd.DataFrame, tables: dict[str, pd.DataFrame], figure_dir: Path, table_dir: Path) -> pd.DataFrame:
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
    plot_heatmap(speed_area, figure_dir / "17_speed_by_area_heatmap.png", "Speed-limit patterns differ substantially between urban and rural roads")

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
