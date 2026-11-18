"""Blend experiments from the OOF store.

Two methods, both working on stored out-of-fold predictions:

- ``stack``: a linear level-2 model (logistic regression on log-odds for
  classification, Ridge for regression). Scored with nested cross-validation on
  the same folds, so the reported score is out-of-fold, not in-sample.
- ``hill_climb``: greedy weighted averaging (Caruana ensemble selection). Its
  score is computed on the OOF rows it was fitted on, so it is optimistic;
  recent Playground winners found linear stackers generalize better.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

logger = logging.getLogger(__name__)

EPS = 1e-6


def _stack_features(predictions: Sequence[np.ndarray], task: str) -> np.ndarray:
    columns = []
    for pred in predictions:
        if task == "regression":
            columns.append(pred.reshape(-1, 1))
        elif task == "binary":
            p = np.clip(pred[:, 1], EPS, 1 - EPS)
            columns.append(np.log(p / (1 - p)).reshape(-1, 1))
        else:
            columns.append(np.log(np.clip(pred, EPS, 1)))
    return np.hstack(columns)


def _level2_model(task: str, strength: float):
    if task == "regression":
        return Ridge(alpha=strength)
    return LogisticRegression(C=1.0 / strength, max_iter=2000)


def _level2_predict(model, X: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    if task == "regression":
        return model.predict(X)
    proba = model.predict_proba(X)
    if proba.shape[1] != n_classes:
        full = np.zeros((len(X), n_classes))
        full[:, model.classes_.astype(int)] = proba
        proba = full
    return proba


def stack(y: np.ndarray, oofs: Sequence[np.ndarray], tests: Sequence[np.ndarray], fold_ids: np.ndarray,
          task: str, n_classes: int, strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Fit a linear stacker; returns (nested OOF predictions, test predictions, info)."""
    X_oof = _stack_features(oofs, task)
    X_test = _stack_features(tests, task)
    stacked_oof = np.zeros_like(oofs[0], dtype=float)
    for fold in range(int(fold_ids.max()) + 1):
        train_idx, valid_idx = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
        model = _level2_model(task, strength).fit(X_oof[train_idx], y[train_idx])
        stacked_oof[valid_idx] = _level2_predict(model, X_oof[valid_idx], task, n_classes)
    final = _level2_model(task, strength).fit(X_oof, y)
    stacked_test = _level2_predict(final, X_test, task, n_classes)
    coef = np.atleast_2d(getattr(final, "coef_", np.zeros((1, X_oof.shape[1]))))
    return stacked_oof, stacked_test, {
        "method": "stack",
        "level2": type(final).__name__,
        "strength": strength,
        "coefficients": coef.round(4).tolist(),
    }


def hill_climb(oofs: Sequence[np.ndarray], tests: Sequence[np.ndarray], score: Callable[[np.ndarray], float],
               max_iterations: int = 50) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Greedy forward selection with replacement; ``score`` is higher-is-better."""
    scores = [score(pred) for pred in oofs]
    counts = np.zeros(len(oofs), dtype=int)
    best = int(np.argmax(scores))
    counts[best] = 1
    blend = oofs[best].astype(float).copy()
    best_score = scores[best]
    history = [best_score]
    for _ in range(max_iterations):
        n = counts.sum()
        candidates = [score((blend * n + pred) / (n + 1)) for pred in oofs]
        choice = int(np.argmax(candidates))
        if candidates[choice] <= best_score + 1e-9:
            break
        blend = (blend * n + oofs[choice]) / (n + 1)
        counts[choice] += 1
        best_score = candidates[choice]
        history.append(best_score)
    weights = counts / counts.sum()
    test_blend = sum(w * t for w, t in zip(weights, tests) if w > 0)
    return blend, test_blend, {"method": "hill_climb", "weights": weights.round(4).tolist(), "history": history}


def select_top(experiments: List, signed_scores: List[float], top: int) -> List[int]:
    """Indices of the ``top`` best experiments by score."""
    order = np.argsort(signed_scores)[::-1]
    return [int(i) for i in order[:top]]
