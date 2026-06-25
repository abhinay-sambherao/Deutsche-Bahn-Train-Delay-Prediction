"""
Feature engineering module.

Generates temporal, weather-based, lag, rolling, and encoded features
from the merged train-weather dataset. Prepares data for both
classification and regression modeling.
"""

import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.utils.config import CONFIG
from src.utils.logger import logger


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from datetime columns.

    Generates: hour, day_of_week, month, season, is_weekend, is_holiday,
    is_rush_hour, is_morning_peak, is_evening_peak.

    Args:
        df: Input DataFrame with 'date' column (YYYY-MM-DD string).

    Returns:
        DataFrame with new temporal feature columns added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["hour"] = df.get("hour", df["scheduled_arrival"].dt.hour)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([16, 17, 18]).astype(int)

    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived weather features.

    Generates: weather_severity score from weather_code.

    Args:
        df: Input DataFrame with weather columns.

    Returns:
        DataFrame with weather_severity column added.
    """
    df = df.copy()

    severity_map = {
        0: 0, 1: 0, 2: 1, 3: 1, 45: 1, 48: 1,
        51: 2, 53: 2, 55: 2, 56: 2, 57: 2,
        61: 3, 63: 3, 65: 3, 66: 3, 67: 3,
        71: 3, 73: 3, 75: 3, 77: 3,
        80: 3, 81: 3, 82: 3, 85: 3, 86: 3,
        95: 4, 96: 4, 99: 4,
    }
    if "weather_code" in df.columns:
        df["weather_severity"] = df["weather_code"].map(severity_map).fillna(0).astype(int)
    else:
        df["weather_severity"] = 0

    return df


def create_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling window features for delay.

    Args:
        df: Input DataFrame with 'delay_minutes' column.

    Returns:
        DataFrame with 'lag_delay_1' and 'rolling_mean_delay_3' columns.
    """
    df = df.copy()
    df = df.sort_values(["station_name", "date", "hour"]).reset_index(drop=True)

    df["lag_delay_1"] = df.groupby("station_name")["delay_minutes"].shift(1)
    df["rolling_mean_delay_3"] = (
        df.groupby("station_name")["delay_minutes"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    df["lag_delay_1"] = df["lag_delay_1"].fillna(0)
    df["rolling_mean_delay_3"] = df["rolling_mean_delay_3"].fillna(
        df["delay_minutes"].mean()
    )

    return df


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for preprocessing.

    Args:
        numeric_cols: List of numeric column names to scale.
        categorical_cols: List of categorical column names to one-hot encode.

    Returns:
        Configured ColumnTransformer.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def engineer_features(
    df: pd.DataFrame,
    target_classification: str = "is_delayed",
    target_regression: str = "delay_minutes",
    test_size: float = 0.2,
    val_size: float = 0.1,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           ColumnTransformer, List[str]]:
    """Complete feature engineering pipeline.

    Creates all features, splits into train/val/test sets,
    applies preprocessing, and returns prepared datasets.

    Args:
        df: Merged train-weather DataFrame.
        target_classification: Name of classification target column.
        target_regression: Name of regression target column.
        test_size: Proportion of data for test set.
        val_size: Proportion of training data for validation.
        save: Whether to save outputs to disk.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train_clf, y_val_clf, y_test_clf,
                  y_train_reg, y_test_reg, preprocessor, feature_names).
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    df = df.copy()

    if "date" not in df.columns or "hour" not in df.columns:
        logger.error("Missing required columns: date/hour")
        raise ValueError("Input must contain 'date' and 'hour' columns")

    logger.info("Creating temporal features...")
    df = create_temporal_features(df)

    logger.info("Creating weather features...")
    df = create_weather_features(df)

    logger.info("Creating lag/rolling features...")
    df = create_lag_rolling_features(df)

    logger.info("Dataset shape after feature engineering: %s", df.shape)

    drop_cols = [
        "scheduled_arrival", "actual_arrival", "scheduled_departure",
        "actual_departure", "platform", "date", "time",
        "latitude", "longitude", "precipitation",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_df = df.drop(columns=drop_cols, errors="ignore")

    logger.info("Feature columns: %s", list(feature_df.columns))

    clf_target = feature_df.pop(target_classification)
    reg_target = feature_df.pop(target_regression)

    y_reg = reg_target.values
    y_clf = clf_target.values

    id_vars = [c for c in feature_df.columns if c.startswith("station_")]
    feature_cols = [c for c in feature_df.columns if c not in id_vars]

    X = feature_df[feature_cols]

    logger.info("Feature matrix shape: %s", X.shape)

    X_temp, X_test, y_clf_temp, y_test_clf, y_reg_temp, y_test_reg = train_test_split(
        X, y_clf, y_reg,
        test_size=test_size,
        random_state=CONFIG.data.random_seed,
        stratify=y_clf,
    )

    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train_clf, y_val_clf, y_train_reg, y_val_reg = train_test_split(
        X_temp, y_clf_temp, y_reg_temp,
        test_size=val_frac,
        random_state=CONFIG.data.random_seed,
        stratify=y_clf_temp,
    )

    logger.info(
        "Split: Train %s, Val %s, Test %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    numeric_cols = CONFIG.features.scale_cols
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    categorical_cols = CONFIG.features.encode_cols
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    cat_names = []
    if categorical_cols:
        cat_transformer = preprocessor.named_transformers_["cat"]
        encoder = cat_transformer.named_steps.get("encoder", cat_transformer)
        if hasattr(encoder, "get_feature_names_out"):
            cat_names = list(encoder.get_feature_names_out(categorical_cols))

    feature_names = numeric_cols + cat_names

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_processed, columns=feature_names, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    logger.info("Final feature set: %d features", len(feature_names))

    if save:
        os.makedirs(CONFIG.paths.data_processed, exist_ok=True)
        X_train_df.to_csv(CONFIG.paths.X_train_path, index=False)
        X_val_df.to_csv(CONFIG.paths.X_test_path.replace("X_test", "X_val"), index=False)
        X_test_df.to_csv(CONFIG.paths.X_test_path, index=False)
        pd.Series(y_train_clf).to_csv(CONFIG.paths.y_train_path, index=False)
        pd.Series(y_test_clf).to_csv(CONFIG.paths.y_test_path, index=False)
        pd.Series(y_train_reg).to_csv(CONFIG.paths.y_train_reg_path, index=False)
        pd.Series(y_test_reg).to_csv(CONFIG.paths.y_test_reg_path, index=False)

        os.makedirs(CONFIG.paths.models_dir, exist_ok=True)
        import joblib
        joblib.dump(preprocessor, CONFIG.paths.preprocessor_path)
        logger.info("Saved preprocessor to %s", CONFIG.paths.preprocessor_path)

    return (
        X_train_df, X_val_df, X_test_df,
        y_train_clf, y_val_clf, y_test_clf,
        y_train_reg, y_test_reg,
        preprocessor, feature_names,
    )


if __name__ == "__main__":
    merged = pd.read_csv(CONFIG.paths.merged_raw, low_memory=False)
    engineer_features(merged)
