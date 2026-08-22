'''
Create restrained, presentation-ready diagnostics for a fitted binary classifier.
'''

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

ACCENT = "#39728C"
MATRIX_ACCENT = "#8064A2"
GREY = "#B8BDC2"
DARK_GREY = "#555B61"
GRID_GREY = "#E5E7E9"
SOURCE = "Source: UK Department for Transport road collision data; held-out 2025 test set."


def _style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": DARK_GREY, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID_GREY, "grid.linewidth": 0.7,
        "axes.titleweight": "normal", "figure.facecolor": "white",
    })


def _save(fig: plt.Figure, path: Path) -> None:
    fig.text(0.01, 0.008, SOURCE, ha="left", va="bottom", fontsize=8, color=DARK_GREY)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _direct_label(ax, x: float, y: float, text: str, color: str = DARK_GREY, dy: int = 0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color, fontsize=9, va="center")


def create_model_figures(model, X: pd.DataFrame, y: pd.Series, threshold: float, importance: pd.DataFrame, out_dir: str | Path) -> None:
    '''Generate the held-out evaluation figure set with consistent visual encoding.'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    prevalence = float(y.mean())

    precision, recall, pr_thresholds = precision_recall_curve(y, probabilities)
    ap = average_precision_score(y, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, color=ACCENT, linewidth=2)
    ax.axhline(prevalence, linestyle="--", color=DARK_GREY)
    ax.set(title=f"Average precision reaches {ap:.3f}, versus {prevalence:.3f} prevalence",
           xlabel="Recall (share of KSI collisions identified)", ylabel="Precision (share of alerts that are KSI)", xlim=(0, 1), ylim=(0, 1))
    _direct_label(ax, 0.98, prevalence, f"Prevalence {prevalence:.1%}", dy=7)
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, out_dir / "precision_recall_curve.png")

    fpr, tpr, _ = roc_curve(y, probabilities)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color=DARK_GREY)
    ax.set(title=f"The model separates KSI outcomes with ROC-AUC {roc_auc:.3f}", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, out_dir / "roc_curve.png")

    matrix = confusion_matrix(y, predictions)
    row_share = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.light_palette(MATRIX_ACCENT, as_cmap=True)
    labels = np.array([[f"{matrix[i, j]:,}\n{row_share[i, j]:.1%} of actual class" for j in range(2)] for i in range(2)])
    sns.heatmap(row_share, annot=labels, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False, square=True,
                xticklabels=["Predicted slight", "Predicted KSI"], yticklabels=["Actual slight", "Actual KSI"], ax=ax)
    ax.set(title=f"At threshold {threshold:.2f}, the model identifies {row_share[1, 1]:.1%} of KSI collisions", xlabel="Predicted class", ylabel="Actual class")
    _save(fig, out_dir / "confusion_matrix.png")

    threshold_frame = pd.DataFrame({"threshold": pr_thresholds, "precision": precision[:-1], "recall": recall[:-1]})
    denominator = threshold_frame["precision"] + threshold_frame["recall"]
    threshold_frame["f1"] = (2 * threshold_frame["precision"] * threshold_frame["recall"] / denominator).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = [("precision", GREY, "-"), ("recall", DARK_GREY, "--"), ("f1", ACCENT, "-")]
    for metric, color, linestyle in styles:
        ax.plot(threshold_frame["threshold"], threshold_frame[metric], color=color, linestyle=linestyle, linewidth=2)
        end = threshold_frame.iloc[-1]
        _direct_label(ax, end["threshold"], end[metric], metric.title(), color=color)
    ax.axvline(threshold, linestyle=":", color=DARK_GREY)
    ax.annotate(f"Selected {threshold:.2f}", (threshold, 0.04), xytext=(5, 0), textcoords="offset points", fontsize=9, color=DARK_GREY)
    ax.set(title="The validation-selected threshold balances KSI precision and recall", xlabel="Decision threshold", ylabel="Metric value", xlim=(0, 1), ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, out_dir / "threshold_metrics.png")

    observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
    max_gap = float(np.max(np.abs(observed - predicted)))
    upper = min(1.0, max(observed.max(), predicted.max()) * 1.12)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(predicted, observed, marker="o", color=ACCENT, linewidth=2)
    ax.plot([0, upper], [0, upper], "--", color=DARK_GREY)
    ax.set(title=f"The largest calibration gap across risk groups is {max_gap:.1%}", xlabel="Mean predicted KSI probability", ylabel="Observed KSI share", xlim=(0, upper), ylim=(0, upper))
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, out_dir / "calibration_curve.png")

    top = importance.head(15).sort_values("importance_mean")
    top = top.assign(feature_label=top["feature"].str.replace("_", " ").str.title())
    colors = [ACCENT if i == top["importance_mean"].idxmax() else GREY for i in top.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature_label"], top["importance_mean"], xerr=top["importance_std"], color=colors, ecolor=DARK_GREY, capsize=2)
    most_important = top.loc[top["importance_mean"].idxmax()]
    label_x = most_important["importance_mean"] + most_important["importance_std"] + 0.001
    ax.text(
        label_x, most_important["feature_label"], f"{most_important['importance_mean']:.3f}",
        va="center", ha="left", fontsize=9, color=DARK_GREY,
    )
    upper = float((top["importance_mean"] + top["importance_std"]).max()) * 1.20
    ax.set_xlim(0, upper)
    ax.set(title=f"{top.iloc[-1]['feature_label']} contributes the most test-year predictive information",
           xlabel="Decrease in average precision after permutation", ylabel="Feature")
    ax.grid(axis="y", visible=False)
    _save(fig, out_dir / "permutation_importance.png")
