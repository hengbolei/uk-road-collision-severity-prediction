"""Presentation-ready visual diagnostics for a fitted binary classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve


def create_model_figures(model, X: pd.DataFrame, y: pd.Series, threshold: float, importance: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=ax, curve_kwargs={"color": "#E45756"})
    ax.axhline(y.mean(), linestyle="--", color="#777777", label=f"Prevalence = {y.mean():.3f}")
    ax.set_title("Precision-recall performance on the future test year")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "precision_recall_curve.png", dpi=180, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    RocCurveDisplay.from_predictions(y, probabilities, ax=ax, curve_kwargs={"color": "#4C78A8"})
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set_title("ROC performance on the future test year")
    fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=180, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, predictions, display_labels=["Slight", "KSI"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix at validation-selected threshold ({threshold:.2f})")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=180, facecolor="white"); plt.close(fig)

    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    threshold_frame = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})
    denominator = threshold_frame["precision"] + threshold_frame["recall"]
    threshold_frame["f1"] = (2 * threshold_frame["precision"] * threshold_frame["recall"] / denominator).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, color in [("precision", "#4C78A8"), ("recall", "#E45756"), ("f1", "#59A14F")]:
        ax.plot(threshold_frame["threshold"], threshold_frame[metric], label=metric.title(), color=color)
    ax.axvline(threshold, linestyle="--", color="#777777", label=f"Selected threshold = {threshold:.2f}")
    ax.set(title="Validation-selected decision threshold on the future test year", xlabel="Decision threshold", ylabel="Score", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "threshold_metrics.png", dpi=180, facecolor="white"); plt.close(fig)

    observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(predicted, observed, marker="o", color="#59A14F", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
    ax.set(title="Predicted KSI risk needs calibration checking", xlabel="Mean predicted probability", ylabel="Observed KSI share")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "calibration_curve.png", dpi=180, facecolor="white"); plt.close(fig)

    top = importance.head(15).sort_values("importance_mean")
    top = top.assign(feature_label=top["feature"].str.replace("_", " ").str.title())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature_label"], top["importance_mean"], xerr=top["importance_std"], color="#4C78A8")
    ax.set(title="Predictive importance on the future test year", xlabel="Decrease in average precision after permutation", ylabel="Feature")
    fig.tight_layout(); fig.savefig(out_dir / "permutation_importance.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)
