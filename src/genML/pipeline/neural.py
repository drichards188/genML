"""Neural-network and foundation-model estimators: RealMLP, TabM (pytabkit) and TabPFN.

All three are sklearn-style and fit per fold like any other model (inside the OOF
stage and its nested encodings), so they stack with the gradient-boosted trees.

* ``RealMLP``: an MLP with robust scaling, numeric embeddings and tuned defaults
  (pytabkit ``RealMLP_TD_*``). Strong on medium and large data.
* ``TabM``: an efficient ensemble of MLPs sharing most weights (pytabkit ``TabM_D_*``),
  early-stopped on a validation split of the training fold.
* ``TabPFN``: a pretrained tabular foundation model that predicts from the training
  rows in context, with no training. Best on small data (up to about 10k-50k rows);
  on larger folds it uses a random sample of the rows as context (``max_context``),
  optionally several samples averaged (``context_bags``). Its weights are gated: they
  download only after the owner accepts Prior Labs' license (``TABPFN_TOKEN``).

The pytabkit models keep an internal validation split (``val_fraction``, 20% of the
training fold) to pick the best epoch, so they never see the fold they are scored on.
Fixed parameters come from ``budget.model_params: {RealMLP: {n_epochs: 32}}``
(checked against ``MODEL_PARAM_RULES``); there is no Optuna tuning for these models.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

NEURAL_MODELS = ("RealMLP", "TabM", "TabPFN")

# param -> (type, (min, max)) or (type, None); the settings an experiment may fix
MODEL_PARAM_RULES: Dict[str, Dict[str, tuple]] = {
    "RealMLP": {"n_epochs": (int, (1, 1000)), "batch_size": (int, (16, 65536)), "lr": (float, (1e-5, 1.0)),
                "hidden_sizes": (list, None), "p_drop": (float, (0.0, 0.9)), "wd": (float, (0.0, 1.0)),
                "val_fraction": (float, (0.05, 0.5))},
    "TabM": {"n_epochs": (int, (1, 10000)), "batch_size": (int, (16, 65536)), "lr": (float, (1e-5, 1.0)),
             "patience": (int, (1, 200)), "k": (int, (1, 64)), "d_block": (int, (16, 4096)),
             "n_blocks": (int, (1, 8)), "dropout": (float, (0.0, 0.9)), "val_fraction": (float, (0.05, 0.5))},
    "TabPFN": {"max_context": (int, (100, 100_000)), "context_bags": (int, (1, 16)),
               "n_estimators": (int, (1, 32)), "softmax_temperature": (float, (0.1, 10.0)),
               "predict_batch": (int, (100, 200_000))},
}


def pytabkit_available() -> bool:
    return importlib.util.find_spec("pytabkit") is not None and importlib.util.find_spec("torch") is not None


def tabpfn_available() -> bool:
    """TabPFN is installed and its weights can load (a license token, or weights already cached)."""
    if importlib.util.find_spec("tabpfn") is None:
        return False
    if os.environ.get("TABPFN_TOKEN"):
        return True
    cache = Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", Path.home() / ".cache" / "tabpfn"))
    return any(p.suffix in (".ckpt", ".safetensors") and not p.name.startswith(".") for p in cache.glob("*"))


def available_models() -> Dict[str, bool]:
    return {"RealMLP": pytabkit_available(), "TabM": pytabkit_available(), "TabPFN": tabpfn_available()}


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def check_model_params(model_params: Any) -> List[str]:
    """Problems with ``budget.model_params`` (empty list = acceptable)."""
    if not isinstance(model_params, dict):
        return ["budget.model_params must be a mapping of model name to parameters"]
    errors = []
    for model, params in model_params.items():
        rules = MODEL_PARAM_RULES.get(model)
        if rules is None:
            errors.append(f"budget.model_params.{model}: fixed parameters are supported for {list(MODEL_PARAM_RULES)}")
            continue
        if not isinstance(params, dict):
            errors.append(f"budget.model_params.{model} must be a mapping")
            continue
        for key, value in params.items():
            if key not in rules:
                errors.append(f"budget.model_params.{model}.{key} is not a known parameter ({sorted(rules)})")
                continue
            kind, bounds = rules[key]
            numeric = kind in (int, float) and isinstance(value, (int, float)) and not isinstance(value, bool)
            if kind is int and isinstance(value, float):
                numeric = False
            if not (numeric or (kind is list and isinstance(value, list))):
                errors.append(f"budget.model_params.{model}.{key} must be {kind.__name__}, got {value!r}")
            elif bounds and not bounds[0] <= value <= bounds[1]:
                errors.append(f"budget.model_params.{model}.{key}={value} is outside {bounds[0]}..{bounds[1]}")
    return errors


def default_epochs(n_rows: int) -> int:
    """RealMLP epochs by data size: its default 256 is for small data (about 1 h per fold at 560k rows)."""
    return int(min(256, max(16, round(5_000_000 / max(n_rows, 1)))))


class _ParamsMixin:
    """sklearn get/set_params over an open-ended parameter dict (for seeds and cloning)."""

    _params: Dict[str, Any]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any):
        self._params.update(params)
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._params})"


class _Imputer:
    """Column medians for NaN/inf (the pytabkit models expect finite inputs)."""

    def fit(self, X: np.ndarray) -> "_Imputer":
        X = np.asarray(X, dtype=np.float32)
        finite = np.where(np.isfinite(X), X, np.nan)
        medians = np.nanmedian(finite, axis=0) if len(X) else np.zeros(X.shape[1])
        self.medians_ = np.nan_to_num(medians, nan=0.0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float32, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.medians_, np.where(bad)[1])
        return X


class PyTabKitModel(_ParamsMixin, BaseEstimator):
    """RealMLP or TabM from pytabkit, with NaN imputation and size-aware defaults."""

    def __init__(self, kind: str = "RealMLP", task: str = "binary", **params: Any):
        if kind not in ("RealMLP", "TabM"):
            raise ValueError(f"kind must be RealMLP or TabM, got {kind!r}")
        self.kind, self.task = kind, task
        self._params = {"random_state": None, **params}

    def _estimator(self, n_rows: int):
        import pytabkit

        regression = self.task == "regression"
        suffix = "Regressor" if regression else "Classifier"
        cls = getattr(pytabkit, f"RealMLP_TD_{suffix}" if self.kind == "RealMLP" else f"TabM_D_{suffix}")
        params = {"device": _device(), "verbosity": 0, **self._params}
        if self.kind == "RealMLP" and params.get("n_epochs") is None:
            params["n_epochs"] = default_epochs(n_rows)
        if params.get("random_state") is None:
            params["random_state"] = 0
        return cls(**params)

    def fit(self, X, y, **_ignored: Any):
        self.imputer_ = _Imputer().fit(X)
        X = self.imputer_.transform(X)
        y = np.asarray(y)
        self.model_ = self._estimator(len(X))
        self.model_.fit(X, y.astype(np.float32) if self.task == "regression" else y)
        if self.task != "regression":
            self.classes_ = np.asarray(getattr(self.model_, "classes_", np.unique(y)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self.imputer_.transform(X)), dtype=float)

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict(self.imputer_.transform(X)))


class PyTabKitClassifier(ClassifierMixin, PyTabKitModel):
    pass


class PyTabKitRegressor(RegressorMixin, PyTabKitModel):
    pass


class TabPFNModel(_ParamsMixin, BaseEstimator):
    """TabPFN with the training fold sampled down to ``max_context`` rows (``context_bags`` samples)."""

    def __init__(self, task: str = "binary", estimator_factory: Optional[Callable[..., Any]] = None, **params: Any):
        self.task = task
        self.estimator_factory = estimator_factory  # tests inject a stand-in for the gated weights
        self._params = {"random_state": None, "max_context": 10_000, "context_bags": 1, "n_estimators": 4,
                        "predict_batch": 20_000, **params}

    def _make(self, seed: int):
        model_params = {k: v for k, v in self._params.items()
                        if k in ("n_estimators", "softmax_temperature") and v is not None}
        if self.estimator_factory is not None:
            return self.estimator_factory(random_state=seed, **model_params)
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        cls = TabPFNRegressor if self.task == "regression" else TabPFNClassifier
        return cls(device=_device(), random_state=seed, ignore_pretraining_limits=True, **model_params)

    def _context(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, size = len(y), int(self._params["max_context"])
        if n <= size:
            return np.arange(n)
        if self.task == "regression":
            return np.sort(rng.choice(n, size, replace=False))
        rows = []
        for value in np.unique(y):  # stratified, keeping at least one row of every class
            members = np.where(y == value)[0]
            take = max(1, int(round(size * len(members) / n)))
            rows.append(rng.choice(members, min(take, len(members)), replace=False))
        return np.sort(np.concatenate(rows))

    def fit(self, X, y, **_ignored: Any):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        seed = int(self._params.get("random_state") or 0)
        rng = np.random.default_rng(seed)
        self.models_ = []
        for bag in range(int(self._params["context_bags"])):
            rows = self._context(y, rng)
            model = self._make(seed + bag)
            model.fit(X[rows], y[rows])
            self.models_.append(model)
        if self.task != "regression":
            self.classes_ = np.unique(y)
        return self

    def _batched(self, method: str, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        step = int(self._params["predict_batch"])
        outputs = []
        for model in self.models_:
            parts = [np.asarray(getattr(model, method)(X[i:i + step]), dtype=float) for i in range(0, len(X), step)]
            outputs.append(np.concatenate(parts) if parts else np.empty((0,)))
        return np.mean(outputs, axis=0)

    def predict_proba(self, X) -> np.ndarray:
        proba = self._batched("predict_proba", X)
        model_classes = np.asarray(getattr(self.models_[0], "classes_", self.classes_))
        if len(model_classes) != len(self.classes_):  # a context sample missed a rare class
            full = np.zeros((proba.shape[0], len(self.classes_)))
            full[:, np.searchsorted(self.classes_, model_classes)] = proba
            proba = full
        return proba

    def predict(self, X) -> np.ndarray:
        if self.task == "regression":
            return self._batched("predict", X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TabPFNClassifierModel(ClassifierMixin, TabPFNModel):
    pass


class TabPFNRegressorModel(RegressorMixin, TabPFNModel):
    pass


def neural_factories(task: str, model_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Callable]:
    """Factories for the models usable here: name -> factory(overrides)."""
    model_params = model_params or {}
    regression = task == "regression"
    factories: Dict[str, Callable] = {}

    def pytabkit_factory(kind: str) -> Callable:
        cls = PyTabKitRegressor if regression else PyTabKitClassifier

        def factory(overrides=None):
            return cls(kind=kind, task=task, **{**model_params.get(kind, {}), **(overrides or {})})
        return factory

    def tabpfn_factory(overrides=None):
        cls = TabPFNRegressorModel if regression else TabPFNClassifierModel
        return cls(task=task, **{**model_params.get("TabPFN", {}), **(overrides or {})})

    availability = available_models()
    if availability["RealMLP"]:
        factories["RealMLP"] = pytabkit_factory("RealMLP")
        factories["TabM"] = pytabkit_factory("TabM")
    if availability["TabPFN"]:
        factories["TabPFN"] = tabpfn_factory
    return factories
