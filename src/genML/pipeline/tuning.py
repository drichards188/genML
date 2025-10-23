"""Hyperparameter optimization objective functions."""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import optuna
import xgboost as xgb

from src.genML.gpu_utils import (
    get_gpu_memory_usage,
    get_linear_model_classifier,
    get_random_forest_classifier,
    get_random_forest_regressor,
    get_xgboost_params,
    is_catboost_gpu_available,
    is_cuml_available,
    is_xgboost_gpu_available,
)
from src.genML.pipeline import config
from src.genML.pipeline.optional_dependencies import (
    CATBOOST_AVAILABLE,
    TABNET_AVAILABLE,
    cb,
    torch,
    TabNetClassifier,
    TabNetRegressor,
)

logger = logging.getLogger(__name__)


def optimize_random_forest(trial: optuna.trial.Trial, X, y, problem_type: str, cv):
    """Optuna objective function for Random Forest hyperparameter optimization."""
    logger.info("[RF Trial %s] Starting Random Forest optimization trial", trial.number)
    logger.info("[RF Trial %s] Input shapes: X=%s, y=%s", trial.number, X.shape, y.shape)
    logger.info("[RF Trial %s] Problem type: %s", trial.number, problem_type)
    logger.info(
        "[RF Trial %s] Using %s",
        trial.number,
        "cuML (GPU)" if is_cuml_available() else "sklearn (CPU)",
    )

    params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 50, config.MEMORY_CONFIG["max_trees_random_forest"]),
        "max_depth": trial.suggest_int("rf_max_depth", 5, 15),
        "random_state": 42,
    }

    if not is_cuml_available():
        params.update(
            {
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
                "n_jobs": config.MEMORY_CONFIG["max_parallel_jobs"],
            }
        )

    if problem_type == "regression":
        RandomForestRegressorClass = get_random_forest_regressor()
        model = RandomForestRegressorClass(**params)
        scoring = "neg_mean_squared_error"
    else:
        RandomForestClassifierClass = get_random_forest_classifier()
        model = RandomForestClassifierClass(**params)
        scoring = "accuracy"

    cv_n_jobs = config.MEMORY_CONFIG["cv_n_jobs_limit"] if not is_cuml_available() else 1

    if is_cuml_available():
        memory_gb = get_gpu_memory_usage()
        if memory_gb is not None:
            logger.info("[RF Trial %s] GPU memory before CV: %.2fGB", trial.number, memory_gb)

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=cv_n_jobs)
    mean_score = scores.mean()

    del model
    del scores
    gc.collect()

    if is_cuml_available():
        try:
            import cupy as cp  # type: ignore

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            memory_gb = get_gpu_memory_usage()
            if memory_gb is not None:
                logger.info("[RF Trial %s] GPU memory after cleanup: %.2fGB", trial.number, memory_gb)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("[RF Trial %s] GPU cleanup failed: %s", trial.number, exc)

    return mean_score


def optimize_xgboost(trial: optuna.trial.Trial, X, y, problem_type: str, cv):
    """Optuna objective function for XGBoost hyperparameter optimization."""
    logger.info("[XGB Trial %s] Starting XGBoost optimization trial", trial.number)

    params = get_xgboost_params()
    params.update(
        {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 800),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),
        }
    )

    if problem_type == "regression":
        params["objective"] = "reg:squarederror"
        scoring = "neg_mean_squared_error"
        ModelClass = xgb.XGBRegressor
    else:
        params["objective"] = "binary:logistic"
        scoring = "accuracy"
        ModelClass = xgb.XGBClassifier

    if is_xgboost_gpu_available() and not config.GPU_CONFIG["force_cpu"]:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        device = "GPU"
    else:
        params["tree_method"] = "hist"
        params["predictor"] = "cpu_predictor"
        device = "CPU"

    logger.info("[XGB Trial %s] Using %s acceleration", trial.number, device)

    model = ModelClass(**params)

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=config.MEMORY_CONFIG["cv_n_jobs_limit"])
    mean_score = scores.mean()

    del model
    del scores
    gc.collect()

    return mean_score


def optimize_linear_model(trial: optuna.trial.Trial, X, y, problem_type: str, cv):
    """Optuna objective function for linear/logistic regression."""
    alpha = trial.suggest_float("ridge_alpha", 1e-4, 10.0, log=True)

    if problem_type == "regression":
        model = Ridge(alpha=alpha, random_state=42)
        scoring = "neg_mean_squared_error"
    else:
        LinearModel = get_linear_model_classifier()
        model = LinearModel(C=1.0 / alpha if alpha > 0 else 1.0, random_state=42, max_iter=1000)
        scoring = "accuracy"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=config.MEMORY_CONFIG["cv_n_jobs_limit"])
    return scores.mean()


def optimize_catboost(trial: optuna.trial.Trial, X, y, problem_type: str, cv):
    """Optuna objective function for CatBoost models."""
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("CatBoost is not installed")

    logger.info("[CatBoost Trial %s] Starting CatBoost optimization trial", trial.number)

    n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
    max_iterations = 1000
    if n_samples <= 1000:
        max_iterations = 400
    elif n_samples <= 5000:
        max_iterations = 700

    logger.info("[CatBoost Trial %s] Sample count=%s, iteration cap=%s", trial.number, n_samples, max_iterations)

    params: dict[str, Any] = {
        "iterations": trial.suggest_int("cb_iterations", 200, max_iterations),
        "learning_rate": trial.suggest_float("cb_learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("cb_depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("cb_l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("cb_subsample", 0.6, 1.0),
        "random_seed": 42,
        "logging_level": "Silent",
    }

    if is_catboost_gpu_available() and not config.GPU_CONFIG["force_cpu"]:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    else:
        params["thread_count"] = config.MEMORY_CONFIG["max_parallel_jobs"]

    # Align bootstrap configuration with subsample usage to avoid CatBoost errors.
    if params["subsample"] < 0.999:
        params["bootstrap_type"] = "Poisson" if params.get("task_type") == "GPU" else "Bernoulli"

    if problem_type == "regression":
        ModelClass = cb.CatBoostRegressor
        scoring = "neg_mean_squared_error"
    else:
        ModelClass = cb.CatBoostClassifier
        scoring = "accuracy"

    model = ModelClass(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=config.MEMORY_CONFIG["cv_n_jobs_limit"])
    mean_score = scores.mean()

    del model
    gc.collect()

    return mean_score


def optimize_tabnet(trial: optuna.trial.Trial, X, y, problem_type: str, cv):
    """Optuna objective function for TabNet models."""
    if not TABNET_AVAILABLE:
        raise RuntimeError("TabNet is not installed")

    params = {
        "n_d": trial.suggest_int("tabnet_n_d", 8, 64),
        "n_a": trial.suggest_int("tabnet_n_a", 8, 64),
        "n_steps": trial.suggest_int("tabnet_n_steps", 3, 10),
        "gamma": trial.suggest_float("tabnet_gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("tabnet_lambda_sparse", 1e-6, 1e-2, log=True),
        "optimizer_params": {"lr": trial.suggest_float("tabnet_lr", 1e-4, 1e-2, log=True)},
        "scheduler_params": {"step_size": 50, "gamma": 0.95},
        "mask_type": "sparsemax",
        "seed": 42,
        "verbose": 0,
    }

    params["device_name"] = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    if problem_type == "regression":
        from sklearn.base import BaseEstimator, RegressorMixin

        class TabNetRegressorWrapper(BaseEstimator, RegressorMixin):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model = None

            def fit(self, X, y):
                self.model = TabNetRegressor(**self.kwargs)
                X_train = X.astype(np.float32)
                y_train = y.astype(np.float32).reshape(-1, 1)
                self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                return self

            def predict(self, X):
                X_test = X.astype(np.float32)
                return self.model.predict(X_test).flatten()

        model = TabNetRegressorWrapper(**params)
        scoring = "neg_mean_squared_error"
    else:
        from sklearn.base import BaseEstimator

        class TabNetClassifierWrapper(BaseEstimator):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model = None

            def fit(self, X, y):
                self.model = TabNetClassifier(**self.kwargs)
                X_train = X.astype(np.float32)
                y_train = y.astype(np.int64)
                self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                return self

            def predict(self, X):
                X_test = X.astype(np.float32)
                return self.model.predict(X_test)

        model = TabNetClassifierWrapper(**params)
        scoring = "accuracy"

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        mean_score = scores.mean()
    except Exception as exc:
        logger.warning("TabNet optimization failed: %s", exc)
        mean_score = float("-inf") if problem_type == "regression" else 0.0

    del model
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_score
