"""
Predict delays for future/present Deutsche Bahn trains using trained models.

Usage:
    # Quick prediction (auto-fetches weather from Open-Meteo):
    python predict.py --station "Berlin Hbf" --train-type ICE --date 2025-06-25 --hour 14

    # Manual weather (no API call):
    python predict.py -s "München Hbf" -t RE -d 2025-06-25 -H 8 \\
        --temp 2 --rain 0 --snowfall 0 --humidity 80 --pressure 1013 --wind 15 --cloud 90

    # Interactive mode:
    python predict.py --interactive

    # Batch prediction from CSV:
    python predict.py --batch trips.csv
"""

import argparse
import csv
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.utils.config import CONFIG
from src.utils.logger import logger


STATIONS = {s["name"]: s for s in CONFIG.data.stations}
TRAIN_TYPES = CONFIG.data.train_types
WEATHER_VARS = CONFIG.weather.variables


# ── Helpers ──────────────────────────────────────────────────────────

def resolve_station(name: str) -> dict:
    """Find station info by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    for full_name, info in STATIONS.items():
        if full_name.lower() == name_lower:
            return {**info, "matched_name": full_name}
        if name_lower in full_name.lower():
            return {**info, "matched_name": full_name}
    logger.warning("Station '%s' not found in list; using Berlin Hbf defaults", name)
    return {**STATIONS["Berlin Hbf"], "matched_name": "Berlin Hbf"}


def fetch_weather_openmeteo(station: dict, date: str, hour: int) -> dict:
    """Fetch hourly weather from Open-Meteo Archive API for a station/date/hour."""
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "Europe/Berlin",
    }
    try:
        resp = requests.get(CONFIG.weather.base_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        idx = hour
        weather = {}
        for var in WEATHER_VARS:
            vals = hourly.get(var, [])
            weather[var] = vals[idx] if idx < len(vals) else 0.0
        return weather
    except Exception as e:
        logger.warning("Open-Meteo API call failed: %s", e)
        return {}


def compute_temporal_features(date: str, hour: int) -> dict:
    """Build temporal features the same way as the training pipeline."""
    dt = pd.to_datetime(date)
    dow = dt.dayofweek
    month = dt.month
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    return {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "season": season_map.get(month, 0),
        "is_weekend": int(dow in (5, 6)),
        "is_rush_hour": int(hour in (7, 8, 9, 16, 17, 18)),
        "is_morning_peak": int(hour in (7, 8, 9)),
        "is_evening_peak": int(hour in (16, 17, 18)),
    }


def weather_severity(code: float) -> int:
    """Map WMO weather code to severity (same as training)."""
    severity_map = {
        0: 0, 1: 0, 2: 1, 3: 1, 45: 1, 48: 1,
        51: 2, 53: 2, 55: 2, 56: 2, 57: 2,
        61: 3, 63: 3, 65: 3, 66: 3, 67: 3,
        71: 3, 73: 3, 75: 3, 77: 3,
        80: 3, 81: 3, 82: 3, 85: 3, 86: 3,
        95: 4, 96: 4, 99: 4,
    }
    return severity_map.get(int(code), 0)


def build_feature_row(
    station_name: str,
    train_type: str,
    date: str,
    hour: int,
    weather: dict,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching preprocessor's feature_names_in_."""
    temporal = compute_temporal_features(date, hour)

    w = {}
    for var in WEATHER_VARS:
        w[var] = weather.get(var, 0.0)

    row = {
        "train_type": train_type,
        "hour": temporal["hour"],
        "temperature_2m": w.get("temperature_2m", 0.0),
        "rain": w.get("rain", 0.0),
        "snowfall": w.get("snowfall", 0.0),
        "relative_humidity_2m": w.get("relative_humidity_2m", 0.0),
        "cloud_cover": w.get("cloud_cover", 0.0),
        "surface_pressure": w.get("surface_pressure", 0.0),
        "wind_speed_10m": w.get("wind_speed_10m", 0.0),
        "weather_code": w.get("weather_code", 0),
        "visibility": w.get("visibility", 0.0),
        "day_of_week": temporal["day_of_week"],
        "month": temporal["month"],
        "season": temporal["season"],
        "is_weekend": temporal["is_weekend"],
        "is_rush_hour": temporal["is_rush_hour"],
        "is_morning_peak": temporal["is_morning_peak"],
        "is_evening_peak": temporal["is_evening_peak"],
        "weather_severity": weather_severity(w.get("weather_code", 0)),
        "lag_delay_1": 0.0,
        "rolling_mean_delay_3": 0.0,
    }

    return pd.DataFrame([row])


def predict_single(
    station_name: str,
    train_type: str,
    date: str,
    hour: int,
    weather: dict = None,
    verbose: bool = True,
):
    """Make a prediction for a single train trip."""
    # Validate & normalize
    train_type_normalized = None
    for tt in TRAIN_TYPES:
        if tt.lower() == train_type.lower():
            train_type_normalized = tt
            break
    if train_type_normalized is None:
        logger.warning("Unknown train type '%s'; valid: %s", train_type, TRAIN_TYPES)
        train_type_normalized = train_type
    train_type = train_type_normalized

    station = resolve_station(station_name)
    matched_station = station["matched_name"]

    # Fetch weather if not provided
    if weather is None:
        if verbose:
            print(f"  Fetching weather for {matched_station} on {date} @ {hour}:00 ...")
        weather = fetch_weather_openmeteo(station, date, hour)
        if not weather:
            print("  (weather API unavailable; using neutral defaults)")
            weather = {}

    # Build feature row
    df = build_feature_row(matched_station, train_type, date, hour, weather)

    # Load models
    try:
        preprocessor = joblib.load(CONFIG.paths.preprocessor_path)
        classifier = joblib.load(CONFIG.paths.best_classifier_path)
        regressor = joblib.load(CONFIG.paths.best_regressor_path)
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        print("  Run 'python main.py' first to train models!")
        return None

    # Transform
    X = preprocessor.transform(df)

    # Predict
    is_delayed = bool(classifier.predict(X)[0])
    proba = classifier.predict_proba(X)[0][1]
    delay_mins = float(regressor.predict(X)[0])

    # Build result
    result = {
        "station": matched_station,
        "train_type": train_type,
        "date": date,
        "hour": hour,
        "is_delayed": is_delayed,
        "probability_delayed": round(proba, 4),
        "predicted_delay_minutes": round(max(delay_mins, 0), 2),
        "weather": {
            k: weather.get(k, "N/A")
            for k in ["temperature_2m", "rain", "snowfall", "wind_speed_10m", "cloud_cover"]
        },
    }

    return result


def print_result(result: dict):
    """Pretty-print a prediction result."""
    if result is None:
        return

    print()
    print("=" * 58)
    print("  DB DELAY PREDICTION RESULT")
    print("=" * 58)
    print(f"  Station:        {result['station']}")
    print(f"  Train Type:     {result['train_type']}")
    print(f"  Date:           {result['date']}")
    print(f"  Hour:           {result['hour']}:00")
    print(f"  Temperature:    {result['weather'].get('temperature_2m', '?')} °C")
    print(f"  Rain:           {result['weather'].get('rain', '?')} mm")
    print(f"  Snowfall:       {result['weather'].get('snowfall', '?')} cm")
    print(f"  Wind Speed:     {result['weather'].get('wind_speed_10m', '?')} km/h")
    print(f"  Cloud Cover:    {result['weather'].get('cloud_cover', '?')}%")
    print("-" * 58)

    if result["is_delayed"]:
        prob = result["probability_delayed"] * 100
        print(f"  ⚠ DELAYED  (confidence: {prob:.0f}%)")
    else:
        prob = (1 - result["probability_delayed"]) * 100
        print(f"  ✅ ON TIME  (confidence: {prob:.0f}%)")

    print(f"  Expected delay: ~{result['predicted_delay_minutes']} minutes")
    print("=" * 58)
    print()


def interactive_mode():
    """Interactive input loop for multiple predictions."""
    print()
    print("═" * 58)
    print("  DB Delay Predictor - Interactive Mode")
    print("═" * 58)
    print("  Press Ctrl+C or type 'quit' to exit.")
    print()

    try:
        while True:
            station = input("  Station [Berlin Hbf]: ").strip() or "Berlin Hbf"
            if station.lower() == "quit":
                break
            train_type = input("  Train type [ICE]: ").strip() or "ICE"
            if train_type.lower() == "quit":
                break
            date = input("  Date (YYYY-MM-DD) [2025-06-25]: ").strip() or "2025-06-25"
            if date.lower() == "quit":
                break
            try:
                hour = int(input("  Hour (0-23) [14]: ").strip() or "14")
            except ValueError:
                hour = 14
            print()
            result = predict_single(station, train_type, date, hour, verbose=True)
            print_result(result)
    except KeyboardInterrupt:
        print("\n  Bye!")
        sys.exit(0)


def batch_mode(csv_path: str):
    """Predict for multiple trips from a CSV file."""
    print(f"\n  Loading trips from {csv_path} ...")
    try:
        trips = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        sys.exit(1)

    required = {"station_name", "train_type", "date", "hour"}
    if not required.issubset(trips.columns):
        logger.error("CSV must have columns: %s", required)
        sys.exit(1)

    results = []
    for _, row in trips.iterrows():
        weather = {}
        for var in WEATHER_VARS:
            if var in row:
                weather[var] = row[var]
        result = predict_single(
            row["station_name"],
            row["train_type"],
            str(row["date"]),
            int(row["hour"]),
            weather=weather if weather else None,
            verbose=False,
        )
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        out_path = "predictions_output.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  Saved {len(results)} predictions to {out_path}")
        print(f"  Summary: {sum(r['is_delayed'] for r in results)} delayed / {len(results)} total")
        print()
        for r in results[:5]:
            status = "⚠ DELAYED" if r["is_delayed"] else "✅ ON TIME"
            print(f"    {r['date']} {r['hour']:02d}:00 | {r['station']:20s} | {r['train_type']:12s} | {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict Deutsche Bahn train delays using trained ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py -s "Berlin Hbf" -t ICE -d 2025-06-25 -H 14
  python predict.py -s "München Hbf" -t RE -d 2025-12-24 -H 8 --temp -2 --snowfall 5
  python predict.py -i
  python predict.py -b trips.csv
        """,
    )

    g = parser.add_argument_group("Trip Details")
    g.add_argument("-s", "--station", default=None, help="Station name (e.g. 'Berlin Hbf')")
    g.add_argument("-t", "--train-type", default=None, help="Train type (e.g. ICE, RE, RB, S-Bahn)")
    g.add_argument("-d", "--date", default=None, help="Date (YYYY-MM-DD)")
    g.add_argument("-H", "--hour", type=int, default=None, help="Hour (0-23)")

    w = parser.add_argument_group("Weather (optional - auto-fetched if omitted)")
    w.add_argument("--temp", type=float, default=None, help="Temperature (C)")
    w.add_argument("--rain", type=float, default=None, help="Rain (mm)")
    w.add_argument("--snowfall", type=float, default=None, help="Snowfall (cm)")
    w.add_argument("--humidity", type=float, default=None, help="Relative humidity (pct)")
    w.add_argument("--pressure", type=float, default=None, help="Surface pressure (hPa)")
    w.add_argument("--wind", type=float, default=None, help="Wind speed (km/h)")
    w.add_argument("--cloud", type=float, default=None, help="Cloud cover (pct)")
    w.add_argument("--weather-code", type=int, default=None, help="WMO weather code")

    m = parser.add_argument_group("Modes")
    m.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    m.add_argument("-b", "--batch", default=None, help="Batch predict from CSV file")

    args = parser.parse_args()

    # Mode dispatch
    if args.interactive:
        interactive_mode()
        return

    if args.batch:
        batch_mode(args.batch)
        return

    # Single prediction mode
    if not all([args.station, args.train_type, args.date is not None, args.hour is not None]):
        parser.print_help()
        print("\n  Error: --station, --train-type, --date, and --hour are required (or use -i / -b)")
        sys.exit(1)

    # Build weather dict from CLI args if provided
    manual_weather = {}
    provided_any = any([
        args.temp is not None, args.rain is not None, args.snowfall is not None,
        args.humidity is not None, args.pressure is not None, args.wind is not None,
        args.cloud is not None, args.weather_code is not None,
    ])
    if provided_any:
        manual_weather["temperature_2m"] = args.temp if args.temp is not None else 0.0
        manual_weather["rain"] = args.rain if args.rain is not None else 0.0
        manual_weather["snowfall"] = args.snowfall if args.snowfall is not None else 0.0
        manual_weather["relative_humidity_2m"] = args.humidity if args.humidity is not None else 0.0
        manual_weather["surface_pressure"] = args.pressure if args.pressure is not None else 0.0
        manual_weather["wind_speed_10m"] = args.wind if args.wind is not None else 0.0
        manual_weather["cloud_cover"] = args.cloud if args.cloud is not None else 0.0
        manual_weather["weather_code"] = args.weather_code if args.weather_code is not None else 0
        manual_weather["visibility"] = 0.0

    weather = manual_weather if provided_any else None
    result = predict_single(
        args.station, args.train_type, args.date, args.hour,
        weather=weather, verbose=True,
    )
    print_result(result)


if __name__ == "__main__":
    main()
