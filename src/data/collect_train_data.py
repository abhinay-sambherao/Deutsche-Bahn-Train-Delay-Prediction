"""
Train data collection and synthetic generation module.

Since historical Deutsche Bahn data is not publicly available as a simple
download, this module generates realistic synthetic train schedule data
that mimics the statistical properties of real DB operations.

The generated dataset includes station metadata, scheduled/actual times,
delay minutes, train types, and platform information.
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import CONFIG
from src.utils.logger import logger


def generate_synthetic_train_data(
    num_records: Optional[int] = None,
    seed: Optional[int] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Generate realistic synthetic Deutsche Bahn train schedule data.

    Creates a DataFrame with station, schedule, delay, and train type
    information that mimics the statistical patterns of real DB operations.

    Args:
        num_records: Number of records to generate. Defaults to config value.
        seed: Random seed for reproducibility. Defaults to config value.
        save: Whether to save the dataset to disk.

    Returns:
        DataFrame with synthetic train data.
    """
    if num_records is None:
        num_records = CONFIG.data.num_records
    if seed is None:
        seed = CONFIG.data.random_seed

    random.seed(seed)
    np.random.seed(seed)

    stations = CONFIG.data.stations
    train_types = CONFIG.data.train_types
    delay_threshold = CONFIG.data.delay_threshold_minutes

    logger.info(
        "Generating %d synthetic train records (seed=%d)...",
        num_records, seed,
    )

    date_start = datetime.strptime(CONFIG.data.date_start, "%Y-%m-%d")
    date_end = datetime.strptime(CONFIG.data.date_end, "%Y-%m-%d")
    total_days = (date_end - date_start).days

    records = []

    for i in tqdm(range(num_records), desc="Generating train data"):
        station = random.choice(stations)
        train_type = random.choice(train_types)
        days_offset = random.randint(0, total_days)
        date = date_start + timedelta(days=days_offset)

        scheduled_hour = random.randint(0, 23)
        scheduled_minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        scheduled_time = date.replace(
            hour=scheduled_hour, minute=scheduled_minute, second=0, microsecond=0,
        )

        scheduled_dep_hour = (scheduled_hour + random.randint(0, 2)) % 24
        scheduled_dep_minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])

        delay_base = np.random.exponential(scale=3.0)
        weather_effect = np.random.gamma(shape=0.5, scale=2.0)
        train_type_effect = {
            "ICE": np.random.normal(0, 1),
            "IC": np.random.normal(1, 1.5),
            "EC": np.random.normal(1, 1.5),
            "RE": np.random.normal(2, 2),
            "RB": np.random.normal(3, 2.5),
            "S-Bahn": np.random.normal(1, 1),
            "Regionalbahn": np.random.normal(3, 2.5),
        }[train_type]

        hour_effect = np.sin(2 * np.pi * (scheduled_hour - 6) / 24) * 1.5
        delay_minutes = max(0, delay_base + weather_effect + train_type_effect + hour_effect)

        is_delayed = int(delay_minutes > delay_threshold)
        actual_arrival = scheduled_time + timedelta(minutes=int(delay_minutes))
        actual_departure = scheduled_time.replace(
            hour=scheduled_dep_hour, minute=scheduled_dep_minute,
        ) + timedelta(minutes=int(delay_minutes * random.uniform(0.5, 1.5)))

        platform = random.randint(1, 30)

        records.append({
            "station_name": station["name"],
            "latitude": station["lat"],
            "longitude": station["lon"],
            "date": date.strftime("%Y-%m-%d"),
            "scheduled_arrival": scheduled_time.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_arrival": actual_arrival.strftime("%Y-%m-%d %H:%M:%S"),
            "scheduled_departure": scheduled_time.replace(
                hour=scheduled_dep_hour, minute=scheduled_dep_minute,
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "actual_departure": actual_departure.strftime("%Y-%m-%d %H:%M:%S"),
            "train_type": train_type,
            "delay_minutes": round(delay_minutes, 1),
            "is_delayed": is_delayed,
            "platform": platform,
        })

    df = pd.DataFrame(records)

    logger.info(
        "Generated %d records. Delay rate: %.2f%%, Mean delay: %.2f min",
        len(df), df["is_delayed"].mean() * 100, df["delay_minutes"].mean(),
    )

    if save:
        os.makedirs(CONFIG.paths.data_raw, exist_ok=True)
        df.to_csv(CONFIG.paths.train_raw, index=False)
        logger.info("Saved train data to %s", CONFIG.paths.train_raw)

    return df


if __name__ == "__main__":
    generate_synthetic_train_data()
