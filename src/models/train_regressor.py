"""
Regression model training module.

Trains, tunes, and evaluates all regression models including
Linear Regression, Ridge, Lasso, ElasticNet, Random Forest,
Gradient Boosting, XGBoost, LightGBM, CatBoost, and MLPRegressor.
"""

from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.models.evaluate import evaluate_regression


_REGRESSOR_REGISTRY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "XGBoostRegressor": None,
    "LightGBMRegressor": None,
    "CatBoostRegressor": None,
    "MLPRegressor": MLPRegressor,
}

_TUNING_GRIDS = {
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "XGBoostRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    },
    "LightGBMRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
    },
}


def _get_regressor(name: str, params: Dict) -> Any:
    """Instantiate a regressor by name with given parameters.

    Args:
        name: Regressor name from the registry.
        params: Parameter dictionary.

    Returns:
        Instantiated regressor.
    """
    cls = _REGRESSOR_REGISTRY.get(name)
    if cls is not None:
        return cls(**params)

    if name == "XGBoostRegressor":
        import xgboost
        return xgboost.XGBRegressor(**params)
    elif name == "LightGBMRegressor":
        import lightgbm
        return lightgbm.LGBMRegressor(**params)
    elif name == "CatBoostRegressor":
        import catboost
        return catboost.CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unknown regressor: {name}")


def train_all_regressors(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    tune: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, float]], Dict[str, Any], Dict[str, np.ndarray]]:
    """Train every regression model, evaluate, and rank.

    Args:
        X_train: Training features.
        y_train: Training targets (delay minutes).
        X_test: Test features.
        y_test: Test targets.
        tune: Whether to perform hyperparameter tuning.

    Returns:
        Tuple of (trained_models, metrics_list, best_model_info, preds_dict).
    """
    models_config = CONFIG.models.regression_models
    logger.info("=" * 60)
    logger.info("TRAINING %d REGRESSION MODELS", len(models_config))
    logger.info("=" * 60)

    trained_models: Dict[str, Any] = {}
    metrics_list: List[Dict[str, float]] = []
    preds_dict: Dict[str, np.ndarray] = {}
    best_model = None
    best_score = float("inf")
    best_name = ""

    for name, params in models_config.items():
        logger.info("Training regressor: %s", name)

        try:
            model = _get_regressor(name, params)

            if tune and name in _TUNING_GRIDS:
                logger.info("  Tuning hyperparameters for %s...", name)
                grid = _TUNING_GRIDS[name]
                search = RandomizedSearchCV(
                    model, grid, n_iter=CONFIG.models.n_iter_tuning,
                    cv=min(3, CONFIG.models.cv_folds),
                    scoring=CONFIG.models.scoring_regression,
                    random_state=CONFIG.data.random_seed,
                    n_jobs=-1, verbose=0,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                logger.info("  Best params: %s", search.best_params_)
            else:
                model.fit(X_train, y_train)

            trained_models[name] = model

            metrics, y_pred = evaluate_regression(
                model, X_test, y_test, model_name=name,
            )
            metrics_list.append(metrics)
            preds_dict[name] = y_pred

            if metrics["rmse"] < best_score:
                best_score = metrics["rmse"]
                best_model = model
                best_name = name

        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc, exc_info=True)
            continue

    best_model_info = {"model": best_model, "name": best_name, "score": best_score}

    logger.info(
        "Best regressor: %s (RMSE: %.4f)", best_name, best_score,
    )

    return trained_models, metrics_list, best_model_info, preds_dict
