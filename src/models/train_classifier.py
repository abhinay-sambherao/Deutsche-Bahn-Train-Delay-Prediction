"""
Classification model training module.

Trains, tunes, and evaluates all classification models including
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
XGBoost, LightGBM, CatBoost, SVM, KNN, and MLPClassifier.

Supports hyperparameter tuning via GridSearchCV and RandomizedSearchCV.
"""

from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.models.evaluate import evaluate_classification


_CLASSIFIER_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "XGBoost": None,
    "LightGBM": None,
    "CatBoost": None,
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "MLPClassifier": MLPClassifier,
}

_TUNING_GRIDS = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
    },
}


def _get_classifier(name: str, params: Dict) -> Any:
    """Instantiate a classifier by name with given parameters.

    Args:
        name: Classifier name from the registry.
        params: Parameter dictionary.

    Returns:
        Instantiated classifier.
    """
    cls = _CLASSIFIER_REGISTRY.get(name)
    if cls is not None:
        return cls(**params)

    if name == "XGBoost":
        import xgboost
        return xgboost.XGBClassifier(**params)
    elif name == "LightGBM":
        import lightgbm
        return lightgbm.LGBMClassifier(**params)
    elif name == "CatBoost":
        import catboost
        return catboost.CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def train_all_classifiers(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    tune: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, float]], Dict[str, Any], Dict[str, np.ndarray]]:
    """Train every classification model, evaluate, and rank.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        tune: Whether to perform hyperparameter tuning.

    Returns:
        Tuple of (trained_models, metrics_list, best_model_info, preds_dict).
    """
    models_config = CONFIG.models.classification_models
    logger.info("=" * 60)
    logger.info("TRAINING %d CLASSIFICATION MODELS", len(models_config))
    logger.info("=" * 60)

    trained_models: Dict[str, Any] = {}
    metrics_list: List[Dict[str, float]] = []
    preds_dict: Dict[str, np.ndarray] = {}
    best_model = None
    best_score = -1.0
    best_name = ""

    for name, params in models_config.items():
        logger.info("Training classifier: %s", name)

        try:
            model = _get_classifier(name, params)

            if tune and name in _TUNING_GRIDS:
                logger.info("  Tuning hyperparameters for %s...", name)
                grid = _TUNING_GRIDS[name]
                search = RandomizedSearchCV(
                    model, grid, n_iter=CONFIG.models.n_iter_tuning,
                    cv=min(3, CONFIG.models.cv_folds),
                    scoring=CONFIG.models.scoring_classification,
                    random_state=CONFIG.data.random_seed,
                    n_jobs=-1, verbose=0,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                logger.info("  Best params: %s", search.best_params_)
            else:
                model.fit(X_train, y_train)

            trained_models[name] = model

            metrics, y_pred, y_proba = evaluate_classification(
                model, X_test, y_test, model_name=name,
            )
            metrics_list.append(metrics)
            preds_dict[name] = {"pred": y_pred, "proba": y_proba}

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model
                best_name = name

        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc, exc_info=True)
            continue

    best_model_info = {"model": best_model, "name": best_name, "score": best_score}

    logger.info(
        "Best classifier: %s (ROC AUC: %.4f)", best_name, best_score,
    )

    return trained_models, metrics_list, best_model_info, preds_dict
