# Deutsche Bahn Train Delay Prediction

Predict whether a Deutsche Bahn train will be delayed (classification) and estimate delay duration in minutes (regression) using historical train schedules and weather data.

---

## Project Overview

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology across six phases:

| Phase | Description |
|-------|-------------|
| **1. Business Understanding** | Define objectives: predict delays to improve passenger experience and operational planning |
| **2. Data Understanding** | Collect train schedule data, fetch historical weather from Open-Meteo API, perform EDA |
| **3. Data Preparation** | Clean, merge, and engineer features from train and weather datasets |
| **4. Modeling** | Train 10 classifiers + 10 regressors with automated comparison |
| **5. Evaluation** | Rank models by ROC AUC (classification) and RMSE (regression) |
| **6. Deployment** | Save best models, generate plots, produce a final report |

---

## Installation

### Prerequisites

- Python 3.12+
- pip or poetry

### Using pip

```bash
# Clone or navigate to the project directory
cd project

# Install dependencies
pip install -r requirements.txt
```

### Using Poetry

```bash
poetry install
```

---

## Project Structure

```
project/
├── data/
│   ├── raw/              # Raw train and weather CSV files
│   ├── interim/          # Merged dataset
│   ├── processed/        # Feature-engineered, train/test splits
│   └── external/         # External reference data
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Data_Preparation.ipynb
│   ├── 04_Baseline_Model.ipynb
│   ├── 05_Model_Comparison.ipynb
│   ├── 06_Hyperparameter_Tuning.ipynb
│   └── 07_Explainability.ipynb
├── src/
│   ├── data/
│   │   ├── collect_train_data.py      # Synthetic DB train data generation
│   │   ├── collect_weather_data.py    # Open-Meteo API / synthetic fallback
│   │   └── merge_data.py             # Join on (station, date, hour)
│   ├── features/
│   │   └── feature_engineering.py     # Temporal, weather, lag features
│   ├── models/
│   │   ├── train_classifier.py        # 10 classification models
│   │   ├── train_regressor.py         # 10 regression models
│   │   └── evaluate.py               # Metrics, comparison, save/load
│   ├── visualization/
│   │   └── plots.py                   # All EDA & model evaluation plots
│   └── utils/
│       ├── config.py                  # Centralized configuration
│       └── logger.py                  # Logging setup
├── models/                # Saved trained models (.pkl)
├── reports/               # Metrics CSVs, plots, results summary
├── requirements.txt
├── README.md
└── main.py                # End-to-end pipeline orchestrator
```

---

## Data Sources

### 1. Deutsche Bahn Train Data (Synthetic)

Since real historical DB data is not publicly available as a simple download, a realistic synthetic dataset is generated with:

- **15 major German train stations** (Berlin Hbf, Hamburg Hbf, Munchen Hbf, etc.)
- **7 train types** (ICE, IC, EC, RE, RB, S-Bahn, Regionalbahn)
- **Delays** following exponential + gamma distributions with train-type and hour effects
- **50,000+ records** spanning 2021-2024

### 2. Historical Weather Data

Fetched from the [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive):

- Temperature, Precipitation, Rain, Snowfall
- Relative Humidity, Cloud Cover, Surface Pressure
- Wind Speed, Weather Code, Visibility

A **synthetic fallback** generates realistic weather patterns when the API is unavailable.

### Join Key

Datasets are merged on `(station_name, date, hour)`.

---

## How to Run

### Full Pipeline

```bash
python main.py
```

This runs the complete end-to-end pipeline:
1. Generate synthetic train data
2. Fetch or generate weather data
3. Merge datasets
4. Engineer features
5. Train all models
6. Evaluate and rank
7. Generate all visualizations
8. Save best models
9. Write results summary

### Options

| Flag | Description |
|------|-------------|
| `--skip-weather-api` | Use synthetic weather (no API call) |
| `--tune` | Enable hyperparameter tuning |
| `--quick` | Use 5,000 records (fast testing) |
| `--log-level DEBUG` | Verbose logging |

### Examples

```bash
# Quick test with synthetic weather
python main.py --quick --skip-weather-api

# Full pipeline with tuning
python main.py --tune

# Debug mode
python main.py --log-level DEBUG
```

---

## Making Predictions

After training, use `predict.py` to check if a future/present train will be delayed:

### Quick Check (auto-fetches weather)

```bash
# Check an ICE from Berlin on a summer afternoon
python predict.py -s "Berlin Hbf" -t ICE -d 2025-06-25 -H 14
```

### Custom Weather

```bash
# Check a regional train in Munich with snowy conditions
python predict.py -s "Munchen Hbf" -t RE -d 2025-01-15 -H 8 \
    --temp -5 --snowfall 8 --wind 30 --cloud 95
```

### Interactive Mode

```bash
python predict.py --interactive
```

### Batch Mode

```bash
python predict.py --batch trips.csv
```

CSV must have columns: `station_name`, `train_type`, `date`, `hour`. Optional weather columns will be auto-fetched.

### CLI Reference

| Flag | Description |
|------|-------------|
| `-s` / `--station` | Station name (e.g. "Berlin Hbf") |
| `-t` / `--train-type` | Train type (e.g. ICE, RE, S-Bahn) |
| `-d` / `--date` | Date in YYYY-MM-DD format |
| `-H` / `--hour` | Hour (0-23) |
| `--temp`, `--rain`, `--snowfall`, ... | Weather values (optional) |
| `-i` / `--interactive` | Interactive mode |
| `-b` / `--batch` | Batch CSV mode |

---

## Targets

### Primary: Binary Classification

| Class | Label | Condition |
|-------|-------|-----------|
| 0 | On Time | Delay <= 5 minutes |
| 1 | Delayed | Delay > 5 minutes |

### Secondary: Regression

Predict `delay_minutes` directly.

---

## Features

| Category | Features |
|----------|----------|
| **Temporal** | Hour, day_of_week, month, season, is_weekend, is_holiday, is_rush_hour, morning/evening peak |
| **Weather** | Temperature, rain, snowfall, humidity, pressure, wind speed, cloud cover, weather severity |
| **Lag/Rolling** | Lag-1 delay, rolling mean delay (window=3) |
| **Encoded** | Train type (one-hot), station name (one-hot) |

---

## Models

### Classification (10 models)

Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, KNN, MLPClassifier

### Regression (10 models)

Linear Regression, Ridge, Lasso, ElasticNet, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, LightGBM Regressor, CatBoost Regressor, MLPRegressor

---

## Evaluation

| Task | Metrics |
|------|---------|
| Classification | Accuracy, Precision, Recall, F1, ROC AUC |
| Regression | MAE, MSE, RMSE, MAPE, R2 |

Automatic model comparison with ranking tables and comparison plots.

---

## Outputs

All outputs are saved to the `reports/` and `models/` directories:

| Output | Location |
|--------|----------|
| Clean datasets | `data/processed/` |
| Trained models | `models/*.pkl` |
| Metrics tables | `reports/*_metrics.csv` |
| Confusion matrices | `reports/confusion_matrix_*.png` |
| ROC curves | `reports/roc_curve_*.png` |
| Feature importance | `reports/feature_importance_*.png` |
| EDA plots | `reports/*.png` |
| Results summary | `reports/results_summary.txt` |

---

## Code Quality

- **Type hints** throughout
- **Docstrings** (Google style) on all functions
- **Logging** with timestamps, levels, and file output
- **Configuration** via dataclasses in `src/utils/config.py`
- **Error handling** with retries, fallbacks, and graceful degradation
- **Progress bars** with `tqdm`
- **Modular architecture** with reusable functions

---

## Future Improvements

- [ ] Integrate real DB API data (when available)
- [ ] Deep learning models (LSTM, Transformer for time series)
- [ ] Real-time prediction with streaming data
- [ ] Web dashboard (Streamlit / FastAPI)
- [ ] MLOps with MLflow model registry
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Drift monitoring for production deployment
- [ ] More granular station/route-level features

---

## License

MIT - Academic Capstone Project
