"""Hyperparameter optimization objective functions."""

from __future__ import annotations

import gc
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import optuna

from src.genML.gpu_utils import (
    get_gpu_memory_usage,
    get_linear_model_classifier,
    get_random_forest_classifier,
    get_random_forest_regressor,
    is_cuml_available,
)
from src.genML.pipeline import config
from src.genML.pipeline.oof import FoldFeatures, fold_split
from src.genML.pipeline.optional_dependencies import (
    TABNET_AVAILABLE,
    lgb,
    torch,
    TabNetClassifier,
    TabNetRegressor,
)

logger = logging.getLogger(__name__)


def optimize_random_forest(trial: optuna.trial.Trial, X, y, problem_type: str, cv, scoring=None):
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
        default_scoring = "neg_mean_squared_error"
    else:
        RandomForestClassifierClass = get_random_forest_classifier()
        model = RandomForestClassifierClass(**params)
        default_scoring = "accuracy"

    cv_n_jobs = config.MEMORY_CONFIG["cv_n_jobs_limit"] if not is_cuml_available() else 1

    if is_cuml_available():
        memory_gb = get_gpu_memory_usage()
        if memory_gb is not None:
            logger.info("[RF Trial %s] GPU memory before CV: %.2fGB", trial.number, memory_gb)

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring or default_scoring, n_jobs=cv_n_jobs)
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


# --------------------------------------------------------------------------- #
# Gradient-boosted trees: shared folds, nested encodings, early stopping
# --------------------------------------------------------------------------- #

# Tree-count ceiling during tuning; early stopping picks the real count, and the
# mean best iteration over tuning folds becomes the model's fixed tree count for
# the out-of-fold stage (so OOF rows never decide when their own model stops).
MAX_TREES = 6000
EARLY_STOPPING_ROUNDS = 200

GBDT_KINDS = {"XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "cb", "GLM-LightGBM": "lgb"}
TREE_COUNT_PARAM = {"xgb": "n_estimators", "lgb": "n_estimators", "cb": "iterations"}


# Search spaces: param -> (type, low, high, log scale). Ranges reach down to
# stumps (depth 1, 2 leaves): Playground data is often generated from
# near-additive rules, where shallow trees trained long win. Diagnostics read
# these bounds to flag tuned values that sit on an edge (diagnostics.range_edges).
# A problem can narrow or move any range with budget.search_space (gbdt_space()).
GBDT_SPACE: Dict[str, Dict[str, Tuple[str, float, float, bool]]] = {
    "xgb": {
        "max_depth": ("int", 1, 8, False),
        "learning_rate": ("float", 0.02, 0.2, True),
        "min_child_weight": ("float", 1.0, 100.0, True),
        "subsample": ("float", 0.5, 1.0, False),
        "colsample_bytree": ("float", 0.3, 1.0, False),
        "reg_alpha": ("float", 1e-3, 10.0, True),
        "reg_lambda": ("float", 1e-3, 50.0, True),
    },
    "lgb": {
        "num_leaves": ("int", 2, 64, True),
        "learning_rate": ("float", 0.02, 0.2, True),
        "min_child_samples": ("int", 10, 500, True),
        "subsample": ("float", 0.5, 1.0, False),
        "colsample_bytree": ("float", 0.3, 1.0, False),
        "reg_alpha": ("float", 1e-3, 10.0, True),
        "reg_lambda": ("float", 1e-3, 50.0, True),
    },
    "cb": {
        "depth": ("int", 1, 8, False),
        "learning_rate": ("float", 0.02, 0.2, True),
        "l2_leaf_reg": ("float", 1.0, 50.0, True),
        "subsample": ("float", 0.6, 1.0, False),
    },
}
GBDT_FIXED: Dict[str, Dict[str, Any]] = {"lgb": {"subsample_freq": 1}}
# Outer limits for budget.search_space overrides: param -> (min, max)
GBDT_LIMITS: Dict[str, Tuple[float, float]] = {
    "max_depth": (1, 16), "depth": (1, 12), "num_leaves": (2, 1024),
    "learning_rate": (1e-3, 1.0), "min_child_weight": (1e-3, 1e4), "min_child_samples": (1, 10_000),
    "subsample": (0.1, 1.0), "colsample_bytree": (0.1, 1.0),
    "reg_alpha": (1e-8, 1e3), "reg_lambda": (1e-8, 1e3), "l2_leaf_reg": (1e-3, 1e3),
}
MAX_TREES_LIMITS = (100, 30_000)


def check_search_space(search_space: Any) -> list:
    """Problems with a ``budget.search_space`` mapping (empty list = valid).

    Shape: ``{model name: {param: [low, high]}}`` for XGBoost, LightGBM, CatBoost
    (and models tuned like them), params from ``GBDT_SPACE``, within ``GBDT_LIMITS``.
    """
    if not isinstance(search_space, dict) or not search_space:
        return ["budget.search_space must be a non-empty mapping of model -> {param: [low, high]}"]
    errors = []
    for model, ranges in search_space.items():
        kind = GBDT_KINDS.get(model)
        if kind is None:
            errors.append(f"budget.search_space.{model}: only {', '.join(GBDT_KINDS)} have search spaces")
            continue
        if not isinstance(ranges, dict) or not ranges:
            errors.append(f"budget.search_space.{model} must map params to [low, high]")
            continue
        for param, bounds in ranges.items():
            where = f"budget.search_space.{model}.{param}"
            if param not in GBDT_SPACE[kind]:
                errors.append(f"{where} is not tuned for {model} (tuned: {', '.join(GBDT_SPACE[kind])})")
                continue
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or not all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds)):
                errors.append(f"{where} must be [low, high], got {bounds!r}")
                continue
            low, high = bounds
            if GBDT_SPACE[kind][param][0] == "int" and not all(float(b).is_integer() for b in bounds):
                errors.append(f"{where} takes whole numbers, got {bounds!r}")
            if not low < high:
                errors.append(f"{where}: low must be below high, got {bounds!r}")
            lo_limit, hi_limit = GBDT_LIMITS[param]
            if low < lo_limit or high > hi_limit:
                errors.append(f"{where}={list(bounds)} is outside {lo_limit}..{hi_limit}")
    return errors


def gbdt_space(kind: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[str, float, float, bool]]:
    """``GBDT_SPACE[kind]`` with ``{param: [low, high]}`` overrides (type and log scale kept)."""
    if kind not in GBDT_SPACE:
        raise ValueError(f"Unknown GBDT kind '{kind}'")
    space = dict(GBDT_SPACE[kind])
    for param, (low, high) in (overrides or {}).items():
        kind_of_value, _, _, log = space[param]
        space[param] = (kind_of_value, low, high, log)
    return space


def model_search_space(model: str, budget: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The effective search space of a GBDT model under a problem budget (None for other models)."""
    kind = GBDT_KINDS.get(model)
    if kind is None:
        return None
    return gbdt_space(kind, ((budget or {}).get("search_space") or {}).get(model))


def suggest_gbdt_params(trial: optuna.trial.Trial, kind: str,
                        space: Optional[Dict[str, Tuple[str, float, float, bool]]] = None) -> Dict[str, Any]:
    """Sample one GBDT family's parameters from ``space`` (default ``GBDT_SPACE[kind]``) plus fixed entries."""
    if kind not in GBDT_SPACE:
        raise ValueError(f"Unknown GBDT kind '{kind}'")
    params: Dict[str, Any] = {}
    for name, (kind_of_value, low, high, log) in (space or GBDT_SPACE[kind]).items():
        key = f"{kind}_{name}"
        if kind_of_value == "int":
            params[name] = trial.suggest_int(key, int(low), int(high), log=log)
        else:
            params[name] = trial.suggest_float(key, float(low), float(high), log=log)
    params.update(GBDT_FIXED.get(kind, {}))
    return params


def _fit_early_stopping(model, kind: str, X_train, y_train, X_valid, y_valid, rounds: int) -> int:
    """Fit with early stopping on the validation rows; return the number of trees used."""
    if kind == "xgb":
        model.set_params(early_stopping_rounds=rounds)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return int(model.best_iteration) + 1
    if kind == "lgb":
        callbacks = [lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)]
        if "eval_X" in inspect.signature(model.fit).parameters:  # LightGBM >= 4.6 renamed eval_set
            model.fit(X_train, y_train, eval_X=(X_valid,), eval_y=(y_valid,), callbacks=callbacks)
        else:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=callbacks)
        return int(model.best_iteration_ or model.n_estimators)
    if kind == "cb":
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=rounds,
                  use_best_model=True, verbose=False)
        return int(model.get_best_iteration()) + 1
    raise ValueError(f"Unknown GBDT kind '{kind}'")


def fold_cv_score(factory: Callable, params: Optional[Dict[str, Any]], X: np.ndarray, y: np.ndarray,
                  fold_ids: np.ndarray, folds: Sequence[int], scoring,
                  fold_features: Optional[FoldFeatures] = None, kind: Optional[str] = None,
                  early_stopping_rounds: Optional[int] = None) -> Tuple[float, list, Optional[int]]:
    """Score a model on some of the shared folds (higher is better).

    Uses the problem's fixed folds and, when present, each fold's nested encodings,
    so tuning sees exactly the features the OOF stage will. With ``kind`` and
    ``early_stopping_rounds`` the tree count is chosen per fold by early stopping.
    Returns (mean score, per-fold scores, mean trees used or None).
    """
    scorer = get_scorer(scoring)
    scores, trees = [], []
    for fold in folds:
        train_idx, valid_idx, X_train, X_valid, _ = fold_split(X, None, fold_ids, fold, fold_features)
        model = factory(params)
        if kind and early_stopping_rounds:
            trees.append(_fit_early_stopping(model, kind, X_train, y[train_idx], X_valid, y[valid_idx],
                                             early_stopping_rounds))
        else:
            model.fit(X_train, y[train_idx])
        scores.append(float(scorer(model, X_valid, y[valid_idx])))
        del model, X_train, X_valid
        gc.collect()
    mean_trees = int(round(float(np.mean(trees)))) if trees else None
    return float(np.mean(scores)), scores, mean_trees


def optimize_gbdt(trial: optuna.trial.Trial, kind: str, factory: Callable, X: np.ndarray, y: np.ndarray,
                  fold_ids: np.ndarray, folds: Sequence[int], scoring,
                  fold_features: Optional[FoldFeatures] = None,
                  space: Optional[Dict[str, Tuple[str, float, float, bool]]] = None,
                  max_trees: Optional[int] = None) -> float:
    """Optuna objective for XGBoost / LightGBM / CatBoost on the shared folds.

    ``space`` replaces the default search space (``gbdt_space``); ``max_trees`` the
    tree-count ceiling. The chosen tree count is stored as the trial's ``n_trees``
    user attribute.
    """
    params = suggest_gbdt_params(trial, kind, space)
    params[TREE_COUNT_PARAM[kind]] = int(max_trees or MAX_TREES)
    score, fold_scores, n_trees = fold_cv_score(
        factory, params, X, y, fold_ids, folds, scoring, fold_features,
        kind=kind, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    trial.set_user_attr("n_trees", n_trees)
    trial.set_user_attr("fold_scores", fold_scores)
    logger.info("[%s trial %s] score=%.6f trees=%s", kind, trial.number, score, n_trees)
    return score


def optimize_linear_model(trial: optuna.trial.Trial, X, y, problem_type: str, cv, scoring=None):
    """Optuna objective function for linear/logistic regression.

    Parameter names carry the model's prefix ('linreg_alpha', 'lr_C') so that, once training strips
    it, they are the estimator's own arguments. Columns are standardized inside each fit, the same
    way the model factories do it (``training.build_model_factories``).
    """
    if problem_type == "regression":
        alpha = trial.suggest_float("linreg_alpha", 1e-4, 10.0, log=True)
        model = Ridge(alpha=alpha, random_state=42)
        default_scoring = "neg_mean_squared_error"
    else:
        LinearModel = get_linear_model_classifier()
        # cuML LogisticRegression doesn't support random_state parameter
        # sklearn LogisticRegression does support it
        model_params = {'C': trial.suggest_float("lr_C", 0.1, 1e4, log=True), 'max_iter': 1000}

        # Check if this is cuML or sklearn
        if 'cuml' not in LinearModel.__module__:
            # sklearn uses random_state
            model_params['random_state'] = 42

        model = LinearModel(**model_params)
        default_scoring = "accuracy"

    scores = cross_val_score(make_pipeline(StandardScaler(), model), X, y, cv=cv, scoring=scoring or default_scoring, n_jobs=config.MEMORY_CONFIG["cv_n_jobs_limit"])
    return scores.mean()


def optimize_tabnet(trial: optuna.trial.Trial, X, y, problem_type: str, cv, scoring=None):
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
        default_scoring = "neg_mean_squared_error"
    else:
        from sklearn.base import BaseEstimator

        from sklearn.base import ClassifierMixin

        class TabNetClassifierWrapper(ClassifierMixin, BaseEstimator):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model = None

            def fit(self, X, y):
                self.model = TabNetClassifier(**self.kwargs)
                X_train = X.astype(np.float32)
                y_train = y.astype(np.int64)
                self.classes_ = np.unique(y_train)
                self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                return self

            def predict(self, X):
                X_test = X.astype(np.float32)
                return self.model.predict(X_test)

            def predict_proba(self, X):
                return self.model.predict_proba(X.astype(np.float32))

        model = TabNetClassifierWrapper(**params)
        default_scoring = "accuracy"

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring or default_scoring, n_jobs=1)
        mean_score = scores.mean()
    except Exception as exc:
        logger.warning("TabNet optimization failed: %s", exc)
        mean_score = float("-inf") if problem_type == "regression" else 0.0

    del model
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_score
