"""
Model evaluation module.

Provides comprehensive evaluation metrics and visualization utilities
for both classification and regression models.
"""

import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)

from src.utils.config import CONFIG
from src.utils.logger import logger


def evaluate_classification(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Evaluate a classification model and return metrics.

    Args:
        model: Trained classifier with predict and predict_proba.
        X_test: Test features.
        y_test: Test labels (binary).
        model_name: Name for logging.

    Returns:
        Dictionary of classification metrics.
    """
    logger.info("Evaluating classifier: %s", model_name)

    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError, NotImplementedError):
        y_proba = y_pred

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    logger.info(
        "%s -> Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, AUC: %.4f",
        model_name, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"], metrics["roc_auc"],
    )

    return metrics, y_pred, y_proba


def evaluate_regression(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Evaluate a regression model and return metrics.

    Args:
        model: Trained regressor.
        X_test: Test features.
        y_test: Test target values.
        model_name: Name for logging.

    Returns:
        Dictionary of regression metrics.
    """
    logger.info("Evaluating regressor: %s", model_name)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mask = y_test > 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.any() else np.nan

    metrics = {
        "model": model_name,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2) if not np.isnan(mape) else 0,
        "r2": round(r2, 4),
    }

    logger.info(
        "%s -> MAE: %.4f, RMSE: %.4f, R2: %.4f, MAPE: %.2f%%",
        model_name, mae, rmse, r2, mape if not np.isnan(mape) else 0,
    )

    return metrics, y_pred


def compare_classifiers(
    results: List[Dict[str, float]],
    save: bool = True,
) -> pd.DataFrame:
    """Compare classification model results and rank them.

    Args:
        results: List of metric dicts from evaluate_classification.
        save: Whether to save comparison table.

    Returns:
        DataFrame with ranked model comparison.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION MODEL COMPARISON")
    logger.info("=" * 60)
    for _, row in df.iterrows():
        logger.info(
            "  #%d %-25s Acc=%.4f F1=%.4f AUC=%.4f",
            row["rank"], row["model"], row["accuracy"], row["f1"], row["roc_auc"],
        )

    if save:
        os.makedirs(CONFIG.paths.reports, exist_ok=True)
        df.to_csv(CONFIG.paths.classification_metrics, index=False)
        logger.info("Saved classification comparison to %s", CONFIG.paths.classification_metrics)

    return df


def compare_regressors(
    results: List[Dict[str, float]],
    save: bool = True,
) -> pd.DataFrame:
    """Compare regression model results and rank them.

    Args:
        results: List of metric dicts from evaluate_regression.
        save: Whether to save comparison table.

    Returns:
        DataFrame with ranked model comparison.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("rmse", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    logger.info("\n" + "=" * 60)
    logger.info("REGRESSION MODEL COMPARISON")
    logger.info("=" * 60)
    for _, row in df.iterrows():
        logger.info(
            "  #%d %-30s MAE=%.4f RMSE=%.4f R2=%.4f",
            row["rank"], row["model"], row["mae"], row["rmse"], row["r2"],
        )

    if save:
        os.makedirs(CONFIG.paths.reports, exist_ok=True)
        df.to_csv(CONFIG.paths.regression_metrics, index=False)
        logger.info("Saved regression comparison to %s", CONFIG.paths.regression_metrics)

    return df


def save_model(
    model: Any,
    path: str,
    model_name: str = "model",
) -> None:
    """Save a trained model to disk.

    Args:
        model: Trained model object.
        path: Destination file path.
        model_name: Name for logging.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved %s to %s", model_name, path)


def load_model(path: str) -> Any:
    """Load a trained model from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        Loaded model object.
    """
    logger.info("Loading model from %s", path)
    return joblib.load(path)
