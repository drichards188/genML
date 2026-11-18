"""Metric registry: the single source of truth for how a problem is scored.

Each metric says which sklearn scorer to use for cross-validation (always
"higher is better", following sklearn's ``neg_*`` convention), what the
submission must contain, and which Kaggle ``evaluationMetric`` names map to it.

Output kinds follow AutoGluon's scorer flags:
- ``label``: hard class labels (accuracy, F1, MCC, ...)
- ``probability``: positive-class probability for binary problems (ROC AUC, ...)
- ``class_probabilities``: one probability column per class (multiclass log loss)
- ``top_k``: space-separated top-k labels (MAP@K)
- ``value``: raw regression values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn import metrics as _skm
from sklearn.metrics import make_scorer

Scorer = Union[str, Callable]

# Registry scoring names that are not sklearn built-ins; built lazily
_CUSTOM_SCORERS: Dict[str, Callable[[], Callable]] = {}


def _map_at_k(y_true, proba, k: int = 3) -> float:
    """Mean average precision at k for single-label multiclass targets."""
    y_true = np.asarray(y_true)
    top_k = np.argsort(-np.asarray(proba), axis=1)[:, :k]
    hits = top_k == y_true[:, None]
    # For single-label targets precision@rank is 1/rank at the first hit
    ranks = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, np.inf)
    return float(np.mean(1.0 / ranks))


def _make_map_at_3():
    try:
        return make_scorer(_map_at_k, response_method="predict_proba", k=3)
    except TypeError:  # sklearn < 1.4
        return make_scorer(_map_at_k, needs_proba=True, k=3)


_CUSTOM_SCORERS["map_at_3"] = _make_map_at_3


def _normalized_gini(y_true, p) -> float:
    """Normalized Gini = 2 * AUC - 1 (ranks predictions exactly like ROC AUC)."""
    return float(2 * _skm.roc_auc_score(y_true, p) - 1)


def _amex_metric(y_true, p) -> float:
    """American Express default-prediction metric: mean of a weighted normalized Gini and
    the share of defaults caught in the top 4% of predictions.

    Negatives count 20 times (Amex subsampled them at 5%). Follows the competition's
    reference implementation.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)[::-1]
    y_sorted = y[order]
    weight = np.where(y_sorted == 0, 20.0, 1.0)
    cutoff = int(0.04 * weight.sum())
    top_four = y_sorted[np.cumsum(weight) <= cutoff].sum() / y.sum()

    def weighted_gini(labels: np.ndarray) -> float:
        w = np.where(labels == 0, 20.0, 1.0)
        random = np.cumsum(w / w.sum())
        lorentz = np.cumsum(labels * w) / (labels * w).sum()
        return float(((lorentz - random) * w).sum())

    gini = weighted_gini(y_sorted) / weighted_gini(y[np.argsort(y)[::-1]])
    return float(0.5 * (gini + top_four))


def _probability_scorer(function):
    def build():
        try:
            return make_scorer(function, response_method="predict_proba")
        except TypeError:  # sklearn < 1.4
            return make_scorer(function, needs_proba=True)
    return build


_CUSTOM_SCORERS["normalized_gini"] = _probability_scorer(_normalized_gini)
_CUSTOM_SCORERS["amex"] = _probability_scorer(_amex_metric)


@dataclass(frozen=True)
class Metric:
    """How a problem is evaluated and what its submission must contain."""

    name: str
    tasks: Tuple[str, ...]
    output: str
    # sklearn scoring per task type; "higher is better" always
    scoring: Dict[str, Scorer]
    description: str
    kaggle_names: Tuple[str, ...] = ()
    # Optional target transform that makes the CV score match the metric
    target_transform: Optional[str] = None
    top_k: Optional[int] = None
    greater_is_better: bool = True

    def score(self, y_true, pred, task: str) -> float:
        """Score predictions on the metric's own scale.

        ``pred`` is a class-probability matrix for classification (shape
        n x n_classes) and raw values on the original target scale for
        regression. ``y_true`` holds encoded class indices or original values.
        """
        return float(_SCORERS[self.name](np.asarray(y_true), np.asarray(pred), task))

    def signed(self, value: float) -> float:
        """Convert a native score to "higher is better" for comparisons."""
        return value if self.greater_is_better else -value

    def scoring_for(self, task: str) -> Scorer:
        """Scorer for cross_val_score / Optuna (an sklearn name or scorer object)."""
        if task not in self.tasks:
            raise ValueError(f"Metric '{self.name}' does not support task '{task}'")
        scorer = self.scoring[task]
        return _CUSTOM_SCORERS[scorer]() if scorer in _CUSTOM_SCORERS else scorer


_REGISTRY: Dict[str, Metric] = {}


def _register(metric: Metric) -> None:
    _REGISTRY[metric.name] = metric


_register(Metric(
    name="accuracy",
    tasks=("binary", "multiclass"),
    output="label",
    scoring={"binary": "accuracy", "multiclass": "accuracy"},
    description="Fraction of correct class labels",
    kaggle_names=("categorization accuracy", "accuracy", "accuracy score"),
))
_register(Metric(
    name="balanced_accuracy",
    tasks=("binary", "multiclass"),
    output="label",
    scoring={"binary": "balanced_accuracy", "multiclass": "balanced_accuracy"},
    description="Mean per-class recall",
    kaggle_names=("balanced accuracy", "balanced accuracy score"),
))
_register(Metric(
    name="f1",
    tasks=("binary", "multiclass"),
    output="label",
    scoring={"binary": "f1", "multiclass": "f1_macro"},
    description="F1 score (macro-averaged for multiclass)",
    kaggle_names=("f1 score", "f-score", "mean f1-score", "macro f1 score"),
))
_register(Metric(
    name="mcc",
    tasks=("binary",),
    output="label",
    scoring={"binary": "matthews_corrcoef"},
    description="Matthews correlation coefficient",
    kaggle_names=("matthews correlation coefficient", "mcc"),
))
_register(Metric(
    name="roc_auc",
    tasks=("binary", "multiclass"),
    output="probability",
    scoring={"binary": "roc_auc", "multiclass": "roc_auc_ovr"},
    description="Area under the ROC curve; needs probabilities, not labels",
    kaggle_names=("roc auc score", "auc", "area under receiver operating characteristic curve",
                  "roc auc", "area under the roc curve"),
))
_register(Metric(
    name="normalized_gini",
    tasks=("binary",),
    output="probability",
    scoring={"binary": "normalized_gini"},
    description="Normalized Gini (2 x AUC - 1); needs probabilities",
    kaggle_names=("normalized gini index", "normalized gini", "gini"),
))
_register(Metric(
    name="amex",
    tasks=("binary",),
    output="probability",
    scoring={"binary": "amex"},
    description="Amex default metric: mean of a weighted normalized Gini and the default rate captured in "
                "the top 4% (negatives weighted 20x)",
    kaggle_names=("amex custom gini and x% percentage capture",),
))
_register(Metric(
    name="log_loss",
    tasks=("binary", "multiclass"),
    output="class_probabilities",
    scoring={"binary": "neg_log_loss", "multiclass": "neg_log_loss"},
    description="Cross-entropy of predicted probabilities",
    kaggle_names=("log loss", "logloss", "multiclass loss", "multi class log loss",
                  "binary cross entropy", "logarithmic loss"),
    greater_is_better=False,
))
_register(Metric(
    name="map_at_3",
    tasks=("multiclass",),
    output="top_k",
    scoring={"multiclass": "map_at_3"},
    description="Mean average precision of the top 3 predicted labels",
    kaggle_names=("map@3", "mean average precision @ 3", "map@k"),
    top_k=3,
))
_register(Metric(
    name="rmse",
    tasks=("regression",),
    output="value",
    scoring={"regression": "neg_root_mean_squared_error"},
    description="Root mean squared error",
    # Kaggle reports several RMSE competitions as "Mean Squared Error"; the
    # ranking of models is identical for MSE and RMSE
    kaggle_names=("root mean squared error", "rmse", "mean squared error", "mse"),
    greater_is_better=False,
))
_register(Metric(
    name="rmsle",
    tasks=("regression",),
    output="value",
    # Scored as RMSE on the log1p-transformed target, which equals RMSLE
    scoring={"regression": "neg_root_mean_squared_error"},
    description="Root mean squared log error (target trained on log1p scale)",
    kaggle_names=("root mean squared logarithmic error", "rmsle",
                  "mean squared log error", "root mean squared log error"),
    target_transform="log1p",
    greater_is_better=False,
))
_register(Metric(
    name="mae",
    tasks=("regression",),
    output="value",
    scoring={"regression": "neg_mean_absolute_error"},
    description="Mean absolute error",
    kaggle_names=("mean absolute error", "mae"),
    greater_is_better=False,
))
_register(Metric(
    name="r2",
    tasks=("regression",),
    output="value",
    scoring={"regression": "r2"},
    description="Coefficient of determination",
    kaggle_names=("r2 score", "r2", "coefficient of determination"),
))


def _labels(pred: np.ndarray) -> np.ndarray:
    return np.argmax(pred, axis=1)


def _positive(pred: np.ndarray) -> np.ndarray:
    return pred[:, 1] if pred.ndim == 2 else pred


_SCORERS: Dict[str, Callable] = {
    "accuracy": lambda y, p, t: _skm.accuracy_score(y, _labels(p)),
    "balanced_accuracy": lambda y, p, t: _skm.balanced_accuracy_score(y, _labels(p)),
    "f1": lambda y, p, t: _skm.f1_score(y, _labels(p), average="binary" if t == "binary" else "macro"),
    "mcc": lambda y, p, t: _skm.matthews_corrcoef(y, _labels(p)),
    "roc_auc": lambda y, p, t: (
        _skm.roc_auc_score(y, _positive(p)) if t == "binary"
        else _skm.roc_auc_score(y, p, multi_class="ovr")
    ),
    "log_loss": lambda y, p, t: _skm.log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=np.arange(p.shape[1])),
    "map_at_3": lambda y, p, t: _map_at_k(y, p, k=3),
    "normalized_gini": lambda y, p, t: _normalized_gini(y, _positive(p)),
    "amex": lambda y, p, t: _amex_metric(y, _positive(p)),
    "rmse": lambda y, p, t: float(np.sqrt(_skm.mean_squared_error(y, p))),
    "rmsle": lambda y, p, t: float(np.sqrt(_skm.mean_squared_error(np.log1p(y), np.log1p(np.clip(p, 0, None))))),
    "mae": lambda y, p, t: _skm.mean_absolute_error(y, p),
    "r2": lambda y, p, t: _skm.r2_score(y, p),
}


def get_metric(name: str) -> Metric:
    """Look up a metric by registry name (case-insensitive)."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Known metrics: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def list_metrics() -> List[str]:
    return sorted(_REGISTRY)


def metric_from_kaggle_name(kaggle_name: Optional[str]) -> Optional[Metric]:
    """Map a Kaggle ``evaluationMetric`` string to a registry metric, if known."""
    if not kaggle_name:
        return None
    key = kaggle_name.strip().lower()
    for metric in _REGISTRY.values():
        if key == metric.name or key in metric.kaggle_names:
            return metric
    return None


def default_metric_for(task: str, sample_value_type: Optional[str] = None) -> Tuple[Metric, str]:
    """Fallback metric when a spec does not declare one.

    A sample submission with probability placeholders implies a probability
    metric; otherwise accuracy (classification) or RMSE (regression).
    """
    if task == "regression":
        return get_metric("rmse"), "metric not declared; RMSE is the regression default"
    if sample_value_type == "probability":
        if task == "binary":
            return get_metric("roc_auc"), (
                "metric not declared; sample submission holds probabilities, so assuming ROC AUC"
            )
        return get_metric("log_loss"), (
            "metric not declared; sample submission holds probabilities, so assuming log loss"
        )
    return get_metric("accuracy"), "metric not declared; accuracy is the classification default"
