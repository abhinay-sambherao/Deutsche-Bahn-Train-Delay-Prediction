"""
Weather data collection module.

Fetches historical weather data from the Open-Meteo Archive API
for all stations in the project. Provides caching and retry logic.

API: https://archive-api.open-meteo.com/v1/archive
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.utils.config import CONFIG
from src.utils.logger import logger


def fetch_station_weather(
    lat: float,
    lon: float,
    station_name: str,
    start_date: str,
    end_date: str,
    variables: Optional[List[str]] = None,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch historical weather data for a single station from Open-Meteo.

    Args:
        lat: Station latitude.
        lon: Station longitude.
        station_name: Station name (for logging).
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        variables: List of weather variable names. Defaults to config.
        max_retries: Number of retry attempts on failure.

    Returns:
        DataFrame with hourly weather data, or None if all retries fail.
    """
    if variables is None:
        variables = CONFIG.weather.variables

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "Europe/Berlin",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                CONFIG.weather.base_url,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if "hourly" not in data:
                logger.warning("No hourly data for %s", station_name)
                return None

            hourly = data["hourly"]
            df = pd.DataFrame(hourly)
            df["station_name"] = station_name
            df["latitude"] = lat
            df["longitude"] = lon

            time_col = df.pop("time")
            df.insert(0, "time", time_col)

            logger.debug(
                "Fetched %d weather records for %s", len(df), station_name,
            )
            return df

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt + 1, max_retries, station_name, exc,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.error("All retries exhausted for %s", station_name)
    return None


def collect_all_weather(
    save: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Collect weather data for all configured stations.

    Checks for cached data before making API calls.

    Args:
        save: Whether to save the combined dataset to disk.
        use_cache: Whether to use cached data if available.

    Returns:
        Combined DataFrame with weather data for all stations.
    """
    weather_cache = CONFIG.paths.weather_raw

    if use_cache and os.path.exists(weather_cache):
        logger.info("Loading cached weather data from %s", weather_cache)
        return pd.read_csv(weather_cache, low_memory=False)

    stations = CONFIG.data.stations
    start = f"{CONFIG.weather.start_year}-01-01"
    end = f"{CONFIG.weather.end_year}-12-31"

    logger.info(
        "Fetching weather data for %d stations from %s to %s",
        len(stations), start, end,
    )

    all_frames = []
    for station in tqdm(stations, desc="Fetching weather data"):
        df = fetch_station_weather(
            lat=station["lat"],
            lon=station["lon"],
            station_name=station["name"],
            start_date=start,
            end_date=end,
        )
        if df is not None and not df.empty:
            all_frames.append(df)
        time.sleep(0.2)

    if not all_frames:
        logger.error("No weather data collected from any station")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    logger.info(
        "Collected %s weather records across %d stations",
        f"{len(combined):,}", len(all_frames),
    )

    if save:
        os.makedirs(CONFIG.paths.data_raw, exist_ok=True)
        combined.to_csv(weather_cache, index=False)
        logger.info("Saved weather data to %s", weather_cache)

    return combined


def generate_synthetic_weather(
    save: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic weather data when API is unavailable.

    Creates realistic weather patterns based on station locations and
    seasonal norms across Germany.

    Args:
        save: Whether to save the dataset to disk.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic hourly weather data.
    """
    if seed is None:
        seed = CONFIG.data.random_seed
    rng = __import__("numpy").random.default_rng(seed)
    np_sin = __import__("numpy").sin
    np_pi = __import__("numpy").pi
    np_clip = __import__("numpy").clip

    stations = CONFIG.data.stations
    start = datetime.strptime(f"{CONFIG.weather.start_year}-01-01", "%Y-%m-%d")
    end = datetime.strptime(f"{CONFIG.weather.end_year}-12-31", "%Y-%m-%d")

    logger.info("Generating synthetic weather data as fallback...")

    records = []
    for station in tqdm(stations, desc="Generating weather"):
        current = start
        base_temp = {
            "Berlin Hbf": 10.0, "Hamburg Hbf": 9.5, "München Hbf": 9.0,
            "Frankfurt Hbf": 11.0, "Köln Hbf": 11.0, "Stuttgart Hbf": 10.5,
            "Düsseldorf Hbf": 10.5, "Hannover Hbf": 9.5, "Leipzig Hbf": 9.5,
            "Nürnberg Hbf": 9.5, "Dresden Hbf": 9.0, "Bremen Hbf": 9.5,
            "Essen Hbf": 10.5, "Dortmund Hbf": 10.5, "Freiburg Hbf": 11.5,
        }.get(station["name"], 10.0)

        while current <= end:
            hour = current.hour
            doy = current.timetuple().tm_yday
            seasonal_temp = base_temp + 10 * np_sin(2 * np_pi * (doy - 80) / 365)
            hourly_variation = -3 * np_sin(2 * np_pi * (hour - 14) / 24)
            temp = seasonal_temp + hourly_variation + rng.normal(0, 2)

            rain = max(0, rng.exponential(0.3) * (1 + 0.5 * np_sin(2 * np_pi * doy / 365)))
            snowfall = max(0, rng.exponential(0.1) * max(0, (5 - temp) / 10)) if temp < 5 else 0
            humidity = 60 + 20 * np_sin(2 * np_pi * doy / 365) + rng.normal(0, 10)
            humidity = np_clip(humidity, 10, 100)
            pressure = 1013 + 10 * np_sin(2 * np_pi * doy / 365) + rng.normal(0, 5)
            wind = rng.exponential(3) + 2
            cloud = np_clip(rng.beta(3, 3) * 100 + 10 * np_sin(2 * np_pi * hour / 24), 0, 100)
            visibility = np_clip(rng.normal(15, 5) - cloud * 0.05, 0.5, 30)

            records.append({
                "time": current.strftime("%Y-%m-%dT%H:00"),
                "temperature_2m": round(temp, 1),
                "precipitation": round(rain + snowfall, 1),
                "rain": round(rain, 1),
                "snowfall": round(snowfall, 2),
                "relative_humidity_2m": round(humidity, 1),
                "cloud_cover": round(cloud, 1),
                "surface_pressure": round(pressure, 1),
                "wind_speed_10m": round(wind, 1),
                "weather_code": int(rng.choice([0, 1, 2, 3, 45, 51, 61, 71, 80, 95], p=[0.6, 0.1, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.01])),
                "visibility": round(visibility, 1),
                "station_name": station["name"],
                "latitude": station["lat"],
                "longitude": station["lon"],
            })
            current += __import__("datetime").timedelta(hours=1)

    combined = pd.DataFrame(records)
    logger.info("Generated %s synthetic weather records", f"{len(combined):,}")

    if save:
        os.makedirs(CONFIG.paths.data_raw, exist_ok=True)
        combined.to_csv(CONFIG.paths.weather_raw, index=False)
        logger.info("Saved synthetic weather to %s", CONFIG.paths.weather_raw)

    return combined


if __name__ == "__main__":
    try:
        df = collect_all_weather(use_cache=False)
        if df.empty:
            logger.warning("API failed, falling back to synthetic data")
            df = generate_synthetic_weather()
    except Exception as exc:
        logger.error("Weather collection failed: %s", exc)
        df = generate_synthetic_weather()
