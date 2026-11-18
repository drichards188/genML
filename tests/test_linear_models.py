"""Tuned linear models: the parameters Optuna picks must be ones the model factory accepts.

They used to be tuned as 'ridge_alpha', which no estimator takes, so every tuned linear or
logistic regression failed in the out-of-fold stage and was dropped without a word.
"""

import numpy as np
import optuna
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from src.genML.pipeline.training import build_model_factories
from src.genML.pipeline.tuning import optimize_linear_model


@pytest.mark.parametrize("problem_type, task, name, prefix, trial_params, scoring", [
    ("classification", "binary", "Logistic Regression", "lr_", {"lr_C": 2.0}, "roc_auc"),
    ("regression", "regression", "Linear Regression", "linreg_", {"linreg_alpha": 0.5}, "neg_root_mean_squared_error"),
])
def test_tuned_linear_params_fit_the_factory(problem_type, task, name, prefix, trial_params, scoring):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)) * np.array([1.0, 10.0, 100.0, 1000.0])    # very different scales
    signal = X[:, 0] + X[:, 1] / 10
    y = (signal > 0).astype(int) if problem_type == "classification" else signal + rng.normal(size=300)
    cv = (StratifiedKFold if problem_type == "classification" else KFold)(3, shuffle=True, random_state=0)

    score = optimize_linear_model(optuna.trial.FixedTrial(trial_params), X, y, problem_type, cv, scoring=scoring)
    assert np.isfinite(score)

    params = {key.replace(prefix, ""): value for key, value in trial_params.items()}   # as training strips it
    model = build_model_factories(problem_type, task, len(y), X.shape[1])[name](params).fit(X, y)
    assert model.predict(X).shape == (300,)
    assert model.steps[0][0] == "standardscaler"                         # scaled as when tuned
