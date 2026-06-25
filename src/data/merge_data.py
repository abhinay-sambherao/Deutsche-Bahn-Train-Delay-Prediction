"""
Data merging module.

Joins train schedule data with weather data using the composite key:
  - Station name
  - Date
  - Hour

Handles date/time parsing, key construction, and validation.
"""

import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.utils.config import CONFIG
from src.utils.logger import logger


def parse_datetime_column(
    df: pd.DataFrame,
    column: str,
    format: str = "%Y-%m-%d %H:%M:%S",
) -> pd.DataFrame:
    """Parse a datetime column and extract date/hour components.

    Args:
        df: Input DataFrame.
        column: Name of the datetime column to parse.
        format: Datetime format string.

    Returns:
        DataFrame with parsed datetime, plus 'date' and 'hour' columns added.
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column], format=format, errors="coerce")
    df["date"] = df[column].dt.date.astype(str)
    df["hour"] = df[column].dt.hour
    return df


def load_and_prepare_data(
    train_path: Optional[str] = None,
    weather_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and weather datasets and prepare them for merging.

    Args:
        train_path: Path to train CSV. Defaults to config path.
        weather_path: Path to weather CSV. Defaults to config path.

    Returns:
        Tuple of (prepared_train_df, prepared_weather_df).
    """
    if train_path is None:
        train_path = CONFIG.paths.train_raw
    if weather_path is None:
        weather_path = CONFIG.paths.weather_raw

    logger.info("Loading train data from %s", train_path)
    train_df = pd.read_csv(train_path, low_memory=False)
    logger.info("Loading weather data from %s", weather_path)
    weather_df = pd.read_csv(weather_path, low_memory=False)

    logger.info("Train records: %s, Weather records: %s",
                f"{len(train_df):,}", f"{len(weather_df):,}")

    train_df = parse_datetime_column(train_df, "scheduled_arrival")
    weather_df["time"] = pd.to_datetime(weather_df["time"], errors="coerce")
    weather_df["date"] = weather_df["time"].dt.date.astype(str)
    weather_df["hour"] = weather_df["time"].dt.hour

    return train_df, weather_df


def merge_train_weather(
    train_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Merge train and weather data on (station_name, date, hour).

    Args:
        train_df: Prepared train DataFrame.
        weather_df: Prepared weather DataFrame.
        save: Whether to save the merged dataset.

    Returns:
        Merged DataFrame.
    """
    logger.info("Merging train and weather data...")

    merge_keys = ["station_name", "date", "hour"]
    for key in merge_keys:
        if key not in train_df.columns:
            logger.warning("Key '%s' not in train data", key)
        if key not in weather_df.columns:
            logger.warning("Key '%s' not in weather data", key)

    merged = pd.merge(
        train_df,
        weather_df,
        on=merge_keys,
        how="left",
        suffixes=("_train", "_weather"),
    )

    match_rate = merged["temperature_2m"].notna().mean() * 100
    logger.info(
        "Merged dataset: %s records, %.1f%% weather match rate",
        f"{len(merged):,}", match_rate,
    )

    merged = merged.drop(columns=["latitude_weather", "longitude_weather"], errors="ignore")
    merged = merged.rename(
        columns={"latitude_train": "latitude", "longitude_train": "longitude"},
        errors="ignore",
    )

    if save:
        os.makedirs(CONFIG.paths.data_interim, exist_ok=True)
        merged.to_csv(CONFIG.paths.merged_raw, index=False)
        logger.info("Saved merged data to %s", CONFIG.paths.merged_raw)

    return merged


def run_data_merge(
    train_path: Optional[str] = None,
    weather_path: Optional[str] = None,
    save: bool = True,
) -> pd.DataFrame:
    """End-to-end data merge pipeline.

    Loads, prepares, and merges train and weather datasets.

    Args:
        train_path: Override train data path.
        weather_path: Override weather data path.
        save: Whether to save intermediate and final outputs.

    Returns:
        Merged DataFrame ready for feature engineering.
    """
    logger.info("=" * 60)
    logger.info("DATA MERGE PIPELINE")
    logger.info("=" * 60)

    train_df, weather_df = load_and_prepare_data(train_path, weather_path)
    merged = merge_train_weather(train_df, weather_df, save=save)

    logger.info("Merge complete. Final shape: %s", merged.shape)
    return merged


if __name__ == "__main__":
    run_data_merge()
