"""RealMLP, TabM and TabPFN in the model catalog: wrappers, settings checks and a full run."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.genML.pipeline import neural
from src.genML.pipeline.neural import (PyTabKitClassifier, TabPFNClassifierModel, check_model_params, default_epochs,
                                       neural_factories)


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1])))).astype(int)
    X[::17, 2] = np.nan  # the pytabkit wrappers impute; TabPFN takes NaN as is
    return X, y


def test_model_params_are_checked():
    assert check_model_params({"RealMLP": {"n_epochs": 32, "lr": 0.04}, "TabPFN": {"max_context": 5000}}) == []
    assert check_model_params({"XGBoost": {"max_depth": 3}})              # not a fixed-parameter model
    assert check_model_params({"TabM": {"n_epochs": 2.5}})                # wrong type
    assert check_model_params({"TabPFN": {"max_context": 10}})            # out of range
    assert check_model_params({"RealMLP": {"layers": 3}})                 # unknown parameter
    assert check_model_params([1, 2])


def test_realmlp_epochs_shrink_with_data_size():
    assert default_epochs(5_000) == 256
    assert default_epochs(100_000) == 50
    assert default_epochs(560_000) == 16


@pytest.mark.parametrize("kind", ["RealMLP", "TabM"])
def test_pytabkit_models_fit_predict_and_take_seeds(kind):
    pytest.importorskip("pytabkit")
    from src.genML.pipeline.oof import _with_seed

    X, y = _data()
    model = PyTabKitClassifier(kind=kind, task="binary", n_epochs=4)
    model.fit(X[:600], y[:600])
    proba = model.predict_proba(X[600:])
    assert proba.shape == (200, 2) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert roc_auc_score(y[600:], proba[:, 1]) > 0.75
    assert list(model.classes_) == [0, 1]
    seeded = _with_seed(lambda params: PyTabKitClassifier(kind=kind, task="binary", **(params or {})),
                        {"n_epochs": 2}, 7)
    assert seeded.get_params()["random_state"] == 7 and seeded.get_params()["n_epochs"] == 2


class _StandIn:
    """Plays TabPFN in tests (its real weights are gated); records the context it was given."""

    seen = []

    def __init__(self, random_state=0, n_estimators=4, **_):
        self.model = LogisticRegression()

    def fit(self, X, y):
        _StandIn.seen.append((len(X), np.bincount(y, minlength=3)))
        self.model.fit(np.nan_to_num(X), y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.nan_to_num(X))


def test_tabpfn_samples_context_and_averages_bags():
    X, y = _data(3000)
    y[:5] = 2  # a rare third class must survive the sample
    _StandIn.seen = []
    model = TabPFNClassifierModel(task="multiclass", estimator_factory=_StandIn, max_context=500,
                                  context_bags=3, predict_batch=700)
    model.fit(X, y)
    assert len(_StandIn.seen) == 3
    for size, counts in _StandIn.seen:
        assert size <= 503 and counts[2] >= 1
    proba = model.predict_proba(X[:1000])
    assert proba.shape == (1000, 3) and np.allclose(proba.sum(axis=1), 1)
    assert list(model.predict(X[:5])) == list(model.classes_[np.argmax(proba[:5], axis=1)])

    small = TabPFNClassifierModel(task="binary", estimator_factory=_StandIn, max_context=10_000)
    _StandIn.seen = []
    small.fit(X[:400], (y[:400] > 0).astype(int))
    assert _StandIn.seen[0][0] == 400  # small data: every row is context


def test_tabpfn_is_unavailable_without_license_token(monkeypatch, tmp_path):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(tmp_path))
    if neural.importlib.util.find_spec("tabpfn") is not None:
        assert not neural.tabpfn_available()
        monkeypatch.setenv("TABPFN_TOKEN", "x")
        assert neural.tabpfn_available()
    monkeypatch.setattr(neural, "tabpfn_available", lambda: False)
    assert "TabPFN" not in neural_factories("binary")


def test_experiments_accept_the_new_models_and_reject_unavailable_ones(monkeypatch):
    from src.genML.experiments import check_patch

    monkeypatch.setattr(neural, "available_models", lambda: {"RealMLP": True, "TabM": True, "TabPFN": False})
    ok = {"models": ["LightGBM", "RealMLP", "TabM"], "budget": {"model_params": {"RealMLP": {"n_epochs": 24}}}}
    assert check_patch(ok) == []
    errors = check_patch({"models": ["TabPFN"]})
    assert errors and "license" in errors[0]
    assert check_patch({"budget": {"model_params": {"TabM": {"k": 0}}}})


def test_realmlp_and_tabm_end_to_end(tmp_path):
    pytest.importorskip("pytabkit")
    from src.genML.flow import run_pipeline
    from tests.test_competitive import _spec, _write_problem

    data = _write_problem(tmp_path / "data")
    spec = _spec(data, models=["RealMLP", "TabM", "LightGBM"], ensemble={"method": "stack"},
                 budget={"n_trials": 2, "timeout_per_model": 60, "tuning_folds": 2,
                         "model_params": {"RealMLP": {"n_epochs": 3}, "TabM": {"n_epochs": 3}}},
                 feature_engineering={"base": "raw", "encodings": {"target_encode": "auto"}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    reports = tmp_path / "out" / "toy" / summary["run_id"] / "reports"
    results = json.loads((reports / "model_training_report.json").read_text())["all_results"]
    for name in ("RealMLP", "TabM"):
        assert results[name]["oof_score"] > 0.7, name
        assert len(results[name]["individual_scores"]) == 2   # screened on the 2 tuning folds only
