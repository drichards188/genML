"""Boosting from a GLM: LightGBM trees that start at a logistic regression's logit.

For binary problems whose labels are close to a logistic formula (common in
Playground data), a logistic regression gets the additive part right and
shallow trees then only fit what it gets wrong. Per fit:

1. a logistic regression on the features: target-encoded columns (names ending
   in ``_te``) as logits, other columns as signed log1p, all standardized;
2. LightGBM trained with the regression's logit as ``init_score``;
3. predictions are ``sigmoid(glm logit + trees' raw score)``.

Both parts see only the rows passed to ``fit``, so inside the OOF stage and
tuning they are refitted per fold like any other model. Tuned like LightGBM
(``tuning.GBDT_KINDS``); ``glm_C`` is the regression's inverse regularization.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.genML.pipeline.optional_dependencies import lgb

GLM_DEFAULTS = {"glm_C": 0.3, "glm_max_iter": 200}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class GLMBoostClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier: logistic regression, then LightGBM boosting from its logit.

    Every keyword other than ``glm_C``, ``glm_max_iter`` and ``te_columns`` goes to
    ``lightgbm.LGBMClassifier`` (``n_estimators``, ``num_leaves``, ``random_state``...).
    """

    def __init__(self, te_columns: Optional[Sequence[int]] = None, **params: Any):
        self.te_columns = list(te_columns) if te_columns is not None else []
        self._params = {**GLM_DEFAULTS, "random_state": None, **params}

    # sklearn-style parameters (the LightGBM ones are open-ended, so handled by hand)
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"te_columns": self.te_columns, **self._params}

    def set_params(self, **params: Any) -> "GLMBoostClassifier":
        if "te_columns" in params:
            self.te_columns = list(params.pop("te_columns") or [])
        self._params.update(params)
        return self

    def _lgb_params(self) -> Dict[str, Any]:
        return {k: v for k, v in self._params.items() if k not in GLM_DEFAULTS}

    def __repr__(self) -> str:
        shown = {k: v for k, v in self._params.items() if k in ("glm_C", "num_leaves", "learning_rate", "n_estimators")}
        return f"GLMBoostClassifier({shown})"

    @property
    def n_estimators(self) -> int:
        return int(self._params.get("n_estimators", 100))

    # ------------------------------------------------------------------ #
    def _design(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.sign(X) * np.log1p(np.abs(X))
        if self.te_columns:
            te = np.clip(X[:, self.te_columns], 1e-5, 1 - 1e-5)
            out[:, self.te_columns] = np.log(te / (1 - te))
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.scaler_ = StandardScaler().fit(out)
        return self.scaler_.transform(out)

    def _glm_logit(self, X: np.ndarray) -> np.ndarray:
        return self.glm_.decision_function(self._design(X))

    def fit(self, X, y, eval_X=None, eval_y=None, eval_set=None, callbacks: Optional[List] = None,
            **_ignored: Any) -> "GLMBoostClassifier":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GLM-LightGBM supports binary targets only")
        y01 = (y == self.classes_[1]).astype(int)
        self.glm_ = LogisticRegression(C=float(self._params["glm_C"]), solver="newton-cholesky",
                                       tol=1e-8, max_iter=int(self._params["glm_max_iter"]))
        self.glm_.fit(self._design(X, fit=True), y01)
        z_train = self._glm_logit(X)

        fit_kwargs: Dict[str, Any] = {"init_score": z_train, "callbacks": callbacks}
        if eval_set is not None and eval_X is None:
            eval_X, eval_y = tuple(x for x, _ in eval_set), tuple(t for _, t in eval_set)
        if eval_X is not None:
            eval_X = eval_X if isinstance(eval_X, tuple) else (eval_X,)
            eval_y = eval_y if isinstance(eval_y, tuple) else (eval_y,)
            eval_y = tuple((np.asarray(t) == self.classes_[1]).astype(int) for t in eval_y)
            fit_kwargs["eval_init_score"] = [self._glm_logit(x) for x in eval_X]
            if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
                fit_kwargs.update(eval_X=eval_X, eval_y=eval_y)
            else:
                fit_kwargs["eval_set"] = list(zip(eval_X, eval_y))
        self.booster_ = lgb.LGBMClassifier(**self._lgb_params())
        self.booster_.fit(X, y01, **fit_kwargs)
        self.best_iteration_ = self.booster_.best_iteration_
        self.glm_iterations_ = int(np.max(self.glm_.n_iter_))
        return self

    def decision_function(self, X) -> np.ndarray:
        return self._glm_logit(X) + self.booster_.predict(X, raw_score=True)

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


def te_column_indices(feature_names: Optional[Sequence[str]]) -> List[int]:
    """Positions of target-encoded features (``*_te``) in the feature matrix."""
    return [i for i, name in enumerate(feature_names or []) if str(name).endswith("_te")]
