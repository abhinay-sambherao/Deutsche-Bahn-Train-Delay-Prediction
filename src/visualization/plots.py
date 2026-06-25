"""
Visualization module.

Generates all EDA, model evaluation, and feature importance plots.
Supports both static (Matplotlib/Seaborn) and interactive (Plotly) outputs.
"""

import os
import warnings
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

from src.utils.config import CONFIG
from src.utils.logger import logger

warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def _savefig(name: str, dpi: int = 150) -> str:
    """Save a matplotlib figure to the reports directory.

    Args:
        name: File name (without extension).
        dpi: Resolution in dots per inch.

    Returns:
        Full path to the saved figure.
    """
    reports_dir = CONFIG.paths.reports
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.debug("Saved plot: %s", path)
    return path


def plot_missing_values(df: pd.DataFrame, name: str = "missing_values.png") -> str:
    """Plot missing value proportions for each column.

    Args:
        df: Input DataFrame.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots()
    missing = df.isnull().sum() / len(df) * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
    else:
        ax.barh(range(len(missing)), missing.values, color="coral")
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index)
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing Values by Column")
    plt.tight_layout()
    return _savefig(name)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    name: str = "correlation_heatmap.png",
) -> str:
    """Plot correlation heatmap of numeric features.

    Args:
        df: Input DataFrame.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdBu_r",
        center=0, square=True, linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    return _savefig(name)


def plot_delay_distribution(
    df: pd.DataFrame,
    name: str = "delay_distribution.png",
) -> str:
    """Plot delay minutes distribution.

    Args:
        df: DataFrame with 'delay_minutes' column.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    delay = df["delay_minutes"]
    axes[0].hist(delay, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Delay (minutes)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Delay Minutes")

    delay_clipped = delay.clip(upper=delay.quantile(0.95))
    axes[1].hist(delay_clipped, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Delay (minutes) [95th percentile capped]")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Delay Distribution (Capped at 95th Percentile)")

    plt.tight_layout()
    return _savefig(name)


def plot_target_distribution(
    df: pd.DataFrame,
    name: str = "target_distribution.png",
) -> str:
    """Plot binary target distribution.

    Args:
        df: DataFrame with 'is_delayed' column.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots()
    counts = df["is_delayed"].value_counts()
    labels = ["On Time (≤5 min)", "Delayed (>5 min)"]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts) * 0.01, f"{v:,} ({v / len(df) * 100:.1f}%)",
                ha="center", fontsize=12)
    ax.set_ylabel("Count")
    ax.set_title("Binary Target Distribution: Is Delayed?")
    plt.tight_layout()
    return _savefig(name)


def plot_delay_by_hour(df: pd.DataFrame, name: str = "delay_by_hour.png") -> str:
    """Plot mean delay by hour of day.

    Args:
        df: DataFrame with 'hour' and 'delay_minutes' columns.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots()
    hourly = df.groupby("hour")["delay_minutes"].mean()
    ax.plot(hourly.index, hourly.values, marker="o", color="steelblue", linewidth=2)
    ax.axhline(df["delay_minutes"].mean(), color="red", linestyle="--", alpha=0.7, label="Overall mean")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Mean Delay by Hour of Day")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    return _savefig(name)


def plot_delay_by_weekday(df: pd.DataFrame, name: str = "delay_by_weekday.png") -> str:
    """Plot mean delay by day of week.

    Args:
        df: DataFrame with 'day_of_week' and 'delay_minutes' columns.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots()
    weekday = df.groupby("day_of_week")["delay_minutes"].mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#3498db"] * 5 + ["#2ecc71", "#e74c3c"]
    ax.bar(labels, weekday.values, color=colors, edgecolor="white", width=0.6)
    ax.axhline(df["delay_minutes"].mean(), color="red", linestyle="--", alpha=0.7, label="Overall mean")
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Mean Delay by Day of Week")
    ax.legend()
    plt.tight_layout()
    return _savefig(name)


def plot_delay_by_month(df: pd.DataFrame, name: str = "delay_by_month.png") -> str:
    """Plot mean delay by month.

    Args:
        df: DataFrame with 'month' and 'delay_minutes' columns.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots()
    monthly = df.groupby("month")["delay_minutes"].mean()
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(labels, monthly.values, color="steelblue", edgecolor="white", width=0.6)
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Mean Delay by Month")
    plt.tight_layout()
    return _savefig(name)


def plot_weather_vs_delay(
    df: pd.DataFrame,
    weather_col: str,
    name: Optional[str] = None,
) -> str:
    """Scatter plot of a weather variable vs delay.

    Args:
        df: DataFrame with weather and delay columns.
        weather_col: Name of weather column to plot.
        name: Output file name (auto-generated if None).

    Returns:
        Path to saved figure.
    """
    if name is None:
        name = f"delay_vs_{weather_col}.png"
    fig, ax = plt.subplots()
    ax.scatter(
        df[weather_col], df["delay_minutes"],
        alpha=0.3, s=5, c="steelblue",
    )
    ax.set_xlabel(weather_col.replace("_", " ").title())
    ax.set_ylabel("Delay (minutes)")
    ax.set_title(f"Delay vs {weather_col.replace('_', ' ').title()}")
    plt.tight_layout()
    return _savefig(name)


def plot_train_type_analysis(df: pd.DataFrame, name: str = "delay_by_train_type.png") -> str:
    """Boxplot of delay minutes by train type.

    Args:
        df: DataFrame with 'train_type' and 'delay_minutes' columns.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby("train_type")["delay_minutes"].median().sort_values().index
    sns.boxplot(data=df, x="train_type", y="delay_minutes", order=order,
                palette="Set2", ax=ax, showfliers=False)
    ax.set_xlabel("Train Type")
    ax.set_ylabel("Delay (minutes)")
    ax.set_title("Delay Distribution by Train Type")
    plt.tight_layout()
    return _savefig(name)


def plot_season_analysis(df: pd.DataFrame, name: str = "delay_by_season.png") -> str:
    """Plot mean delay by season.

    Args:
        df: DataFrame with 'season' and 'delay_minutes' columns.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots()
    season_map = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
    seasonal = df.groupby("season")["delay_minutes"].mean()
    labels = [season_map[s] for s in seasonal.index]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    ax.bar(labels, seasonal.values, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Mean Delay by Season")
    plt.tight_layout()
    return _savefig(name)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    name: Optional[str] = None,
) -> str:
    """Plot confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Model name for title.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    if name is None:
        name = f"confusion_matrix_{model_name}.png"
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues",
                                             colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    return _savefig(name)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    name: Optional[str] = None,
) -> str:
    """Plot ROC curve.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities for positive class.
        model_name: Model name for legend.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    if name is None:
        name = f"roc_curve_{model_name}.png"
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = np.trapz(tpr, fpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return _savefig(name)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    name: Optional[str] = None,
) -> str:
    """Plot precision-recall curve.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities for positive class.
        model_name: Model name for legend.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    if name is None:
        name = f"pr_curve_{model_name}.png"
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=model_name, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    plt.tight_layout()
    return _savefig(name)


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 20,
    name: Optional[str] = None,
) -> str:
    """Plot feature importance as horizontal bar chart.

    Args:
        importance: Array of importance values.
        feature_names: List of feature names.
        model_name: Model name for title.
        top_n: Number of top features to show.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    if name is None:
        name = f"feature_importance_{model_name}.png"
    indices = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(range(len(indices)), importance[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features - {model_name}")
    plt.tight_layout()
    return _savefig(name)


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str,
    name: str,
) -> str:
    """Plot model comparison bar chart.

    Args:
        metrics_df: DataFrame with model names and metrics.
        metric_col: Column name of the metric to plot.
        title: Chart title.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    df = metrics_df.sort_values(metric_col, ascending=False).copy()
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
    ax.barh(range(len(df)), df[metric_col].values, color=colors[::-1])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"].values)
    ax.set_xlabel(metric_col)
    ax.set_title(title)
    for i, v in enumerate(df[metric_col].values):
        ax.text(v + max(df[metric_col]) * 0.01, i, f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    return _savefig(name)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    name: Optional[str] = None,
) -> str:
    """Plot predicted vs actual values for regression.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Model name for title.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    if name is None:
        name = f"pred_vs_actual_{model_name}.png"
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, c="steelblue")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect prediction")
    ax.set_xlabel("Actual Delay (minutes)")
    ax.set_ylabel("Predicted Delay (minutes)")
    ax.set_title(f"Predicted vs Actual - {model_name}")
    ax.legend()
    plt.tight_layout()
    return _savefig(name)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    name: Optional[str] = None,
) -> str:
    """Plot residual distribution for regression.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Model name for title.
        name: Output file name.

    Returns:
        Path to saved figure.
    """
    if name is None:
        name = f"residuals_{model_name}.png"
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual (minutes)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Residual Distribution - {model_name}")

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=5, c="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted Delay (minutes)")
    axes[1].set_ylabel("Residual (minutes)")
    axes[1].set_title("Residuals vs Predicted")

    plt.tight_layout()
    return _savefig(name)


def generate_all_eda_plots(df: pd.DataFrame) -> Dict[str, str]:
    """Generate all EDA plots from the merged dataset.

    Args:
        df: Merged DataFrame for EDA.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    logger.info("Generating EDA plots...")
    plots = {}

    plots["missing_values"] = plot_missing_values(df)
    plots["correlation_heatmap"] = plot_correlation_heatmap(df)
    plots["delay_distribution"] = plot_delay_distribution(df)
    plots["target_distribution"] = plot_target_distribution(df)
    plots["delay_by_hour"] = plot_delay_by_hour(df)
    plots["delay_by_weekday"] = plot_delay_by_weekday(df)
    plots["delay_by_month"] = plot_delay_by_month(df)
    plots["delay_by_season"] = plot_season_analysis(df)
    plots["delay_by_train_type"] = plot_train_type_analysis(df)

    for wc in ["temperature_2m", "rain", "snowfall", "wind_speed_10m"]:
        if wc in df.columns:
            plots[f"delay_vs_{wc}"] = plot_weather_vs_delay(df, wc)

    logger.info("Generated %d EDA plots", len(plots))
    return plots


def generate_all_model_plots(
    y_test_clf: np.ndarray,
    y_test_reg: np.ndarray,
    preds_dict_clf: Dict[str, Dict[str, np.ndarray]],
    preds_dict_reg: Dict[str, np.ndarray],
    metrics_clf_df: pd.DataFrame,
    metrics_reg_df: pd.DataFrame,
    best_clf_name: str,
    best_reg_name: str,
) -> Dict[str, str]:
    """Generate all model evaluation plots.

    Args:
        y_test_clf: Test classification labels.
        y_test_reg: Test regression targets.
        preds_dict_clf: Classification predictions dict.
        preds_dict_reg: Regression predictions dict.
        metrics_clf_df: Classification metrics DataFrame.
        metrics_reg_df: Regression metrics DataFrame.
        best_clf_name: Name of best classifier.
        best_reg_name: Name of best regressor.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    logger.info("Generating model evaluation plots...")
    plots = {}

    if best_clf_name in preds_dict_clf:
        info = preds_dict_clf[best_clf_name]
        plots["cm_best_clf"] = plot_confusion_matrix(
            y_test_clf, info["pred"], model_name=best_clf_name,
        )
        plots["roc_best_clf"] = plot_roc_curve(
            y_test_clf, info["proba"], model_name=best_clf_name,
        )
        plots["pr_best_clf"] = plot_precision_recall_curve(
            y_test_clf, info["proba"], model_name=best_clf_name,
        )

    if best_reg_name in preds_dict_reg:
        y_pred = preds_dict_reg[best_reg_name]
        plots["pred_vs_actual_best_reg"] = plot_predicted_vs_actual(
            y_test_reg, y_pred, model_name=best_reg_name,
        )
        plots["residuals_best_reg"] = plot_residuals(
            y_test_reg, y_pred, model_name=best_reg_name,
        )

    if metrics_clf_df is not None and not metrics_clf_df.empty:
        for m in ["accuracy", "f1", "roc_auc"]:
            if m in metrics_clf_df.columns:
                plots[f"clf_comparison_{m}"] = plot_model_comparison(
                    metrics_clf_df, m,
                    f"Classification Model Comparison - {m.upper()}",
                    f"clf_comparison_{m}.png",
                )

    if metrics_reg_df is not None and not metrics_reg_df.empty:
        for m in ["mae", "rmse", "r2"]:
            if m in metrics_reg_df.columns:
                ascending = m not in ["r2"]
                df = metrics_reg_df.sort_values(m, ascending=ascending)
                plots[f"reg_comparison_{m}"] = plot_model_comparison(
                    df, m,
                    f"Regression Model Comparison - {m.upper()}",
                    f"reg_comparison_{m}.png",
                )

    logger.info("Generated %d model evaluation plots", len(plots))
    return plots


def generate_feature_importance_plots(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model",
) -> Dict[str, str]:
    """Generate feature importance plots.

    Attempts to extract feature importance if available, otherwise
    uses permutation importance.

    Args:
        model: Trained model with optional feature_importances_ or coef_.
        feature_names: List of feature names.
        model_name: Model name for titles.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    plots = {}

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importance = np.abs(coef).flatten() if coef.ndim > 1 else np.abs(coef)
    elif hasattr(model, "feature_importance_"):
        importance = model.feature_importance_

    if importance is not None and len(importance) == len(feature_names):
        plots["feature_importance"] = plot_feature_importance(
            importance, feature_names, model_name=model_name,
        )
    else:
        logger.warning("Cannot extract feature importance for %s", model_name)

    return plots
