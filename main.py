#!/usr/bin/env python3
"""
Deutsche Bahn Train Delay Prediction Pipeline

End-to-end ML pipeline following the CRISP-DM methodology:

1. Business Understanding  - Problem definition
2. Data Understanding      - Exploratory data analysis
3. Data Preparation        - Data collection, merging, feature engineering
4. Modeling                - Train classification & regression models
5. Evaluation              - Compare, rank, and select best models
6. Deployment              - Save models and generate reports

Usage:
    python main.py                     # Full pipeline
    python main.py --skip-weather-api  # Use synthetic weather (no API call)
    python main.py --tune              # Enable hyperparameter tuning
    python main.py --quick             # Small dataset for testing

Author: ML Capstone Project
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.utils.config import CONFIG
from src.utils.logger import setup_logger, logger

from src.data.collect_train_data import generate_synthetic_train_data
from src.data.collect_weather_data import collect_all_weather, generate_synthetic_weather
from src.data.merge_data import run_data_merge
from src.features.feature_engineering import engineer_features
from src.models.train_classifier import train_all_classifiers
from src.models.train_regressor import train_all_regressors
from src.models.evaluate import (
    evaluate_classification,
    evaluate_regression,
    compare_classifiers,
    compare_regressors,
    save_model,
)
from src.visualization.plots import (
    generate_all_eda_plots,
    generate_all_model_plots,
    generate_feature_importance_plots,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deutsche Bahn Delay Prediction Pipeline",
    )
    parser.add_argument(
        "--skip-weather-api", action="store_true",
        help="Skip Open-Meteo API calls; use synthetic weather data",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable hyperparameter tuning (slower but better models)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use a small dataset for quick testing",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    return parser.parse_args()


def write_results_summary(
    clf_metrics_df: pd.DataFrame,
    reg_metrics_df: pd.DataFrame,
    best_clf_name: str,
    best_reg_name: str,
    elapsed: float,
    plot_paths: Dict[str, str],
) -> str:
    """Write a human-readable results summary to file.

    Args:
        clf_metrics_df: Classification comparison DataFrame.
        reg_metrics_df: Regression comparison DataFrame.
        best_clf_name: Name of best classifier.
        best_reg_name: Name of best regressor.
        elapsed: Total pipeline runtime in seconds.
        plot_paths: Dictionary of generated plot file paths.

    Returns:
        Path to the summary file.
    """
    lines = [
        "=" * 70,
        "DEUTSCHE BAHN DELAY PREDICTION - RESULTS SUMMARY",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)",
        "",
        "=" * 70,
        "CLASSIFICATION RESULTS",
        "=" * 70,
    ]

    if not clf_metrics_df.empty:
        lines.append(clf_metrics_df.to_string(index=False))

    lines.extend([
        "",
        f"Best classifier: {best_clf_name}",
        "",
        "=" * 70,
        "REGRESSION RESULTS",
        "=" * 70,
    ])

    if not reg_metrics_df.empty:
        lines.append(reg_metrics_df.to_string(index=False))

    lines.extend([
        "",
        f"Best regressor: {best_reg_name}",
        "",
        "=" * 70,
        "GENERATED PLOTS",
        "=" * 70,
    ])

    for name, path in sorted(plot_paths.items()):
        lines.append(f"  {name}: {path}")

    lines.extend([
        "",
        "=" * 70,
        "FILES SAVED",
        "=" * 70,
        f"  Train data:            {CONFIG.paths.train_raw}",
        f"  Weather data:          {CONFIG.paths.weather_raw}",
        f"  Merged data:           {CONFIG.paths.merged_raw}",
        f"  Engineered data:       {CONFIG.paths.engineered_data}",
        f"  Best classifier:       {CONFIG.paths.best_classifier_path}",
        f"  Best regressor:        {CONFIG.paths.best_regressor_path}",
        f"  Preprocessor:          {CONFIG.paths.preprocessor_path}",
        f"  Classifier metrics:    {CONFIG.paths.classification_metrics}",
        f"  Regressor metrics:     {CONFIG.paths.regression_metrics}",
        f"  Results summary:       {CONFIG.paths.results_summary}",
        "",
        "=" * 70,
        "END OF SUMMARY",
        "=" * 70,
    ])

    os.makedirs(CONFIG.paths.reports, exist_ok=True)
    with open(CONFIG.paths.results_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Results summary written to %s", CONFIG.paths.results_summary)
    return CONFIG.paths.results_summary


def main() -> None:
    """Run the complete end-to-end ML pipeline."""
    args = parse_args()

    global_logger = setup_logger("db_delay_pipeline", level=args.log_level)
    global_logger.info("=" * 60)
    global_logger.info("DEUTSCHE BAHN DELAY PREDICTION PIPELINE")
    global_logger.info("=" * 60)
    global_logger.info("Arguments: %s", vars(args))

    start_time = time.time()

    if args.quick:
        CONFIG.data.num_records = 5000

    try:
        # STEP 1: Data Collection (Business Understanding / Data Understanding)
        global_logger.info("")
        global_logger.info("=" * 60)
        global_logger.info("STEP 1: DATA COLLECTION")
        global_logger.info("=" * 60)

        global_logger.info("Generating synthetic Deutsche Bahn train data...")
        train_df = generate_synthetic_train_data()

        if args.skip_weather_api:
            global_logger.info("Skipping weather API, using synthetic weather data...")
            weather_df = generate_synthetic_weather()
        else:
            try:
                global_logger.info("Fetching weather data from Open-Meteo API...")
                weather_df = collect_all_weather(use_cache=False)
                if weather_df.empty:
                    raise ValueError("No weather data from API")
            except Exception as exc:
                global_logger.warning("Weather API failed (%s), using synthetic fallback...", exc)
                weather_df = generate_synthetic_weather()

        # STEP 2: Data Preparation (Data Preparation)
        global_logger.info("")
        global_logger.info("=" * 60)
        global_logger.info("STEP 2: DATA PREPARATION")
        global_logger.info("=" * 60)

        global_logger.info("Merging train and weather datasets...")
        merged_df = run_data_merge()

        global_logger.info("Performing feature engineering...")
        (
            X_train, X_val, X_test,
            y_train_clf, y_val_clf, y_test_clf,
            y_train_reg, y_test_reg,
            preprocessor, feature_names,
        ) = engineer_features(merged_df)

        # STEP 3: Exploratory Data Analysis (Data Understanding)
        global_logger.info("")
        global_logger.info("=" * 60)
        global_logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")
        global_logger.info("=" * 60)

        eda_df = merged_df.copy()
        if "scheduled_arrival" in eda_df.columns:
            eda_df["scheduled_arrival"] = pd.to_datetime(eda_df["scheduled_arrival"])
            eda_df["hour"] = eda_df["scheduled_arrival"].dt.hour
            eda_df["day_of_week"] = eda_df["scheduled_arrival"].dt.dayofweek
            eda_df["month"] = eda_df["scheduled_arrival"].dt.month
            eda_df["season"] = eda_df["month"].map(
                {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
            )
        eda_plots = generate_all_eda_plots(eda_df)

        # STEP 4: Modeling
        global_logger.info("")
        global_logger.info("=" * 60)
        global_logger.info("STEP 4: MODELING")
        global_logger.info("=" * 60)

        global_logger.info("Training classification models...")
        clf_models, clf_metrics_list, best_clf_info, clf_preds = train_all_classifiers(
            X_train=X_train, y_train=y_train_clf,
            X_test=X_test, y_test=y_test_clf,
            tune=args.tune,
        )
        best_clf_model = best_clf_info["model"]
        best_clf_name = best_clf_info["name"]

        global_logger.info("Training regression models...")
        reg_models, reg_metrics_list, best_reg_info, reg_preds = train_all_regressors(
            X_train=X_train, y_train=y_train_reg,
            X_test=X_test, y_test=y_test_reg,
            tune=args.tune,
        )
        best_reg_model = best_reg_info["model"]
        best_reg_name = best_reg_info["name"]

        # STEP 5: Evaluation
        global_logger.info("")
        global_logger.info("=" * 60)
        global_logger.info("STEP 5: EVALUATION")
        global_logger.info("=" * 60)

        clf_metrics_df = compare_classifiers(clf_metrics_list)
        reg_metrics_df = compare_regressors(reg_metrics_list)

        model_plots = generate_all_model_plots(
            y_test_clf=y_test_clf,
            y_test_reg=y_test_reg,
            preds_dict_clf=clf_preds,
            preds_dict_reg=reg_preds,
            metrics_clf_df=clf_metrics_df,
            metrics_reg_df=reg_metrics_df,
            best_clf_name=best_clf_name,
            best_reg_name=best_reg_name,
        )

        fi_plots = {}
        if best_clf_model is not None:
            fi_plots.update(generate_feature_importance_plots(
                best_clf_model, feature_names, model_name=best_clf_name,
            ))

        all_plots = {**eda_plots, **model_plots, **fi_plots}

        # STEP 6: Deployment
        global_logger.info("")
        global_logger.info("=" * 60)
        global_logger.info("STEP 6: DEPLOYMENT")
        global_logger.info("=" * 60)

        if best_clf_model is not None:
            save_model(best_clf_model, CONFIG.paths.best_classifier_path, "best_classifier")
        if best_reg_model is not None:
            save_model(best_reg_model, CONFIG.paths.best_regressor_path, "best_regressor")

        elapsed = time.time() - start_time

        write_results_summary(
            clf_metrics_df=clf_metrics_df,
            reg_metrics_df=reg_metrics_df,
            best_clf_name=best_clf_name,
            best_reg_name=best_reg_name,
            elapsed=elapsed,
            plot_paths=all_plots,
        )

        global_logger.info("")
        global_logger.info("=" * 60)
        global_logger.info("PIPELINE COMPLETE")
        global_logger.info("Total time: %.2f seconds (%.2f minutes)", elapsed, elapsed / 60)
        global_logger.info("Best classifier: %s", best_clf_name)
        global_logger.info("Best regressor:  %s", best_reg_name)
        global_logger.info("Results: %s", CONFIG.paths.results_summary)
        global_logger.info("=" * 60)

    except Exception as exc:
        global_logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
