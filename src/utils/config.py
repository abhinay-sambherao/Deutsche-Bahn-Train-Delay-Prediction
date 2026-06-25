"""
Configuration module for the Deutsche Bahn Delay Prediction project.

This module centralizes all configuration constants, file paths, model parameters,
data source URLs, and feature definitions used across the project.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class PathsConfig:
    """Centralized file and directory path configuration."""
    root: str = PROJECT_ROOT
    data_raw: str = os.path.join(PROJECT_ROOT, "data", "raw")
    data_interim: str = os.path.join(PROJECT_ROOT, "data", "interim")
    data_processed: str = os.path.join(PROJECT_ROOT, "data", "processed")
    data_external: str = os.path.join(PROJECT_ROOT, "data", "external")
    notebooks: str = os.path.join(PROJECT_ROOT, "notebooks")
    reports: str = os.path.join(PROJECT_ROOT, "reports")
    models_dir: str = os.path.join(PROJECT_ROOT, "models")
    logs_dir: str = os.path.join(PROJECT_ROOT, "logs")

    train_raw: str = os.path.join(data_raw, "deutsche_bahn_raw.csv")
    weather_raw: str = os.path.join(data_raw, "weather_raw.csv")
    merged_raw: str = os.path.join(data_interim, "merged_data.csv")
    engineered_data: str = os.path.join(data_processed, "engineered_data.csv")
    X_train_path: str = os.path.join(data_processed, "X_train.csv")
    X_test_path: str = os.path.join(data_processed, "X_test.csv")
    y_train_path: str = os.path.join(data_processed, "y_train.csv")
    y_test_path: str = os.path.join(data_processed, "y_test.csv")
    y_train_reg_path: str = os.path.join(data_processed, "y_train_reg.csv")
    y_test_reg_path: str = os.path.join(data_processed, "y_test_reg.csv")

    classification_metrics: str = os.path.join(reports, "classification_metrics.csv")
    regression_metrics: str = os.path.join(reports, "regression_metrics.csv")
    best_classifier_path: str = os.path.join(models_dir, "best_classifier.pkl")
    best_regressor_path: str = os.path.join(models_dir, "best_regressor.pkl")
    preprocessor_path: str = os.path.join(models_dir, "preprocessor.pkl")
    results_summary: str = os.path.join(reports, "results_summary.txt")


@dataclass
class DataConfig:
    """Data generation and collection configuration."""
    random_seed: int = 42
    num_records: int = 50000
    train_test_split: float = 0.2
    val_split: float = 0.1
    delay_threshold_minutes: int = 5

    stations: List[Dict[str, object]] = field(default_factory=lambda: [
        {"name": "Berlin Hbf", "lat": 52.5250, "lon": 13.3690},
        {"name": "Hamburg Hbf", "lat": 53.5527, "lon": 10.0067},
        {"name": "München Hbf", "lat": 48.1402, "lon": 11.5581},
        {"name": "Frankfurt Hbf", "lat": 50.1072, "lon": 8.6628},
        {"name": "Köln Hbf", "lat": 50.9429, "lon": 6.9581},
        {"name": "Stuttgart Hbf", "lat": 48.7833, "lon": 9.1820},
        {"name": "Düsseldorf Hbf", "lat": 51.2203, "lon": 6.7928},
        {"name": "Hannover Hbf", "lat": 52.3765, "lon": 9.7410},
        {"name": "Leipzig Hbf", "lat": 51.3455, "lon": 12.3823},
        {"name": "Nürnberg Hbf", "lat": 49.4461, "lon": 11.0818},
        {"name": "Dresden Hbf", "lat": 51.0417, "lon": 13.7311},
        {"name": "Bremen Hbf", "lat": 53.0864, "lon": 8.8136},
        {"name": "Essen Hbf", "lat": 51.4515, "lon": 7.0128},
        {"name": "Dortmund Hbf", "lat": 51.5177, "lon": 7.4591},
        {"name": "Freiburg Hbf", "lat": 47.9975, "lon": 7.8419},
    ])

    train_types: List[str] = field(default_factory=lambda: [
        "ICE", "IC", "EC", "RE", "RB", "S-Bahn", "Regionalbahn"
    ])

    date_start: str = "2021-01-01"
    date_end: str = "2024-12-31"


@dataclass
class WeatherConfig:
    """Open-Meteo API configuration."""
    base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    variables: List[str] = field(default_factory=lambda: [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "relative_humidity_2m",
        "cloud_cover",
        "surface_pressure",
        "wind_speed_10m",
        "weather_code",
        "visibility",
    ])
    start_year: int = 2021
    end_year: int = 2024


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    temporal_features: List[str] = field(default_factory=lambda: [
        "hour", "day_of_week", "month", "season", "is_weekend",
        "is_holiday", "is_rush_hour", "is_morning_peak", "is_evening_peak"
    ])
    weather_features: List[str] = field(default_factory=lambda: [
        "temperature_2m", "rain", "snowfall", "relative_humidity_2m",
        "surface_pressure", "wind_speed_10m", "cloud_cover", "weather_code"
    ])
    lag_features: List[str] = field(default_factory=lambda: ["lag_delay_1"])
    rolling_features: List[str] = field(default_factory=lambda: ["rolling_mean_delay_3"])
    encode_cols: List[str] = field(default_factory=lambda: ["train_type", "station_name"])
    scale_cols: List[str] = field(default_factory=lambda: [
        "temperature_2m", "rain", "snowfall", "relative_humidity_2m",
        "surface_pressure", "wind_speed_10m", "cloud_cover",
        "scheduled_arrival_hour", "scheduled_departure_hour"
    ])


@dataclass
class ModelConfig:
    """Model training and evaluation configuration."""
    classification_models: Dict[str, Dict] = field(default_factory=lambda: {
        "LogisticRegression": {"random_state": 42, "max_iter": 1000, "n_jobs": -1},
        "DecisionTree": {"random_state": 42, "max_depth": 10},
        "RandomForest": {"random_state": 42, "n_estimators": 100, "n_jobs": -1},
        "GradientBoosting": {"random_state": 42, "n_estimators": 100},
        "XGBoost": {"random_state": 42, "n_estimators": 100, "eval_metric": "logloss", "verbosity": 0},
        "LightGBM": {"random_state": 42, "n_estimators": 100, "verbose": -1},
        "CatBoost": {"random_state": 42, "iterations": 100, "verbose": 0},
        "SVM": {"random_state": 42, "probability": True, "max_iter": 1000},
        "KNN": {"n_neighbors": 5, "n_jobs": -1},
        "MLPClassifier": {"random_state": 42, "max_iter": 300, "hidden_layer_sizes": (100,)},
    })

    regression_models: Dict[str, Dict] = field(default_factory=lambda: {
        "LinearRegression": {"n_jobs": -1},
        "Ridge": {"random_state": 42},
        "Lasso": {"random_state": 42, "max_iter": 10000},
        "ElasticNet": {"random_state": 42, "max_iter": 10000},
        "RandomForestRegressor": {"random_state": 42, "n_estimators": 100, "n_jobs": -1},
        "GradientBoostingRegressor": {"random_state": 42, "n_estimators": 100},
        "XGBoostRegressor": {"random_state": 42, "n_estimators": 100, "eval_metric": "rmse", "verbosity": 0},
        "LightGBMRegressor": {"random_state": 42, "n_estimators": 100, "verbose": -1},
        "CatBoostRegressor": {"random_state": 42, "iterations": 100, "verbose": 0},
        "MLPRegressor": {"random_state": 42, "max_iter": 300, "hidden_layer_sizes": (100,)},
    })

    cv_folds: int = 5
    n_iter_tuning: int = 20
    scoring_classification: str = "roc_auc"
    scoring_regression: str = "neg_root_mean_squared_error"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    log_level: str = "INFO"
    log_file: str = "pipeline.log"


CONFIG = AppConfig()
