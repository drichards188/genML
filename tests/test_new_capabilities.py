"""Digit keys, per-value encoding, search-space overrides and GLM-initialised boosting."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.genML.features.encoding import DIGIT_KINDS, build_encodings, digit_key, nested_target_encodings
from src.genML.flow import run_pipeline
from src.genML.pipeline.oof import make_folds
from tests.test_competitive import _spec, _write_problem


# --------------------------------------------------------------------------- #
# Digit keys and per-value encoding
# --------------------------------------------------------------------------- #

def test_digit_keys_describe_how_a_number_is_written():
    values = pd.Series([1371.37, 7.5, 0.05, -12.34, np.nan])
    keys = {kind: digit_key(values, kind) for kind in DIGIT_KINDS}
    assert list(keys["int"][:4]) == [1371, 7, 0, -12]
    assert list(keys["int10"][:2]) == [137, 0]
    assert list(keys["tenths"][:2]) == [13713, 75]
    assert list(keys["cents"][:4]) == [37, 50, 5, 34]
    assert list(keys["lead1"][:3]) == [41, 17, 10]      # digit count * 10 + leading digit
    assert list(keys["lead2"][:3]) == [413, 107, 100]   # digit count * 100 + leading two digits
    assert keys["lead1"][3] == -21                      # negative values keep their sign
    assert len({keys[k][4] for k in DIGIT_KINDS}) == 1  # missing values share one key
    with pytest.raises(ValueError):
        digit_key(values, "mantissa")


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    price = np.round(rng.uniform(1, 500, n), 2)
    count = rng.integers(0, 400, n)
    # Signal hidden in the first decimal digit, which a tree on the raw value cannot see easily
    hot = ((np.rint(price * 100).astype(int) // 10) % 10) == 4
    y = (rng.random(n) < np.where(hot, 0.6, 0.1)).astype(int)
    return pd.DataFrame({"price": price, "count": count}), y


def test_digits_and_per_value_encoding_are_target_encoded_and_nested():
    frame, y = _frame()
    train, test = frame.iloc[:1500].reset_index(drop=True), frame.iloc[1500:].reset_index(drop=True)
    y_train = y[:1500]
    folds = make_folds(y_train, {"method": "stratified_kfold", "n_splits": 5, "shuffle": True, "random_state": 0})
    options = {"target_encode": "auto", "numeric_as_categorical": "all", "digits": "auto",
               "digit_keys": ["cents", "tenths", "int"]}
    tr, te, report = build_encodings(train, test, y_train, folds, "binary", 0, [], ["price", "count"], options)

    assert report["numeric_as_categorical"] == ["price", "count"]          # all, not only <= max_unique
    assert report["digit_keys"] == [["price", "cents"], ["price", "tenths"], ["price", "int"]]  # count: no decimals
    assert {"price__cents_te", "price__tenths_te", "price__int_te", "price_te", "count_te"} <= set(tr.columns)
    # The cents key carries the hidden signal; the raw value does not
    raw_auc = roc_auc_score(y_train, train["price"])
    assert roc_auc_score(y_train, tr["price__cents_te"]) > max(raw_auc, 1 - raw_auc) + 0.08

    blocks = nested_target_encodings(train, test, y_train, folds, "binary", 0, report, options)
    assert blocks["columns"] == [c for c in tr.columns if c.endswith("_te")]
    # Block k never depends on fold k's labels
    flipped = y_train.copy()
    flipped[folds == 2] = 1 - flipped[folds == 2]
    again = nested_target_encodings(train, test, flipped, folds, "binary", 0, report, options)
    np.testing.assert_array_equal(blocks["train"][2], again["train"][2])
    np.testing.assert_array_equal(blocks["test"][2], again["test"][2])
    assert not np.array_equal(blocks["train"][0], again["train"][0])


def test_digits_off_by_default():
    frame, y = _frame(n=500)
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 5, "shuffle": True, "random_state": 0})
    _, _, report = build_encodings(frame, frame, y, folds, "binary", 0, [], ["price"], {})
    assert report["digit_keys"] == []


# --------------------------------------------------------------------------- #
# Search-space overrides
# --------------------------------------------------------------------------- #

def test_search_space_overrides_and_edges():
    import optuna

    from src.genML.pipeline.diagnostics import range_edges
    from src.genML.pipeline.tuning import check_search_space, gbdt_space, model_search_space, suggest_gbdt_params

    assert check_search_space({"LightGBM": {"num_leaves": [2, 6], "learning_rate": [0.01, 0.05]}}) == []
    budget = {"search_space": {"LightGBM": {"num_leaves": [2, 6]}}}
    space = model_search_space("LightGBM", budget)
    assert space["num_leaves"] == ("int", 2, 6, True)
    assert space["learning_rate"] == gbdt_space("lgb")["learning_rate"]  # untouched params keep defaults
    assert model_search_space("Random Forest", budget) is None

    study = optuna.create_study(direction="maximize")
    for _ in range(10):
        trial = study.ask()
        assert 2 <= suggest_gbdt_params(trial, "lgb", space)["num_leaves"] <= 6
        study.tell(trial, 0.5)

    # num_leaves=6 is on the moved upper edge, not on the default range's
    assert any(e["param"] == "num_leaves" and e["edge"] == "high" for e in range_edges("LightGBM", {"num_leaves": 6},
                                                                                        budget=budget))
    assert not any(e["param"] == "num_leaves" for e in range_edges("LightGBM", {"num_leaves": 6}))


# --------------------------------------------------------------------------- #
# GLM-initialised boosting
# --------------------------------------------------------------------------- #

def _logistic_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    z = 1.5 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + (X[:, 3] > 1) * 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y


def test_glm_boost_fits_early_stops_and_predicts():
    from src.genML.pipeline.glm_boost import GLMBoostClassifier
    from src.genML.pipeline.oof import _with_seed
    from src.genML.pipeline.tuning import _fit_early_stopping

    X, y = _logistic_data()
    model = GLMBoostClassifier(n_estimators=2000, num_leaves=4, learning_rate=0.05, verbosity=-1)
    trees = _fit_early_stopping(model, "lgb", X[:2000], y[:2000], X[2000:], y[2000:], rounds=50)
    assert 1 <= trees < 2000
    proba = model.predict_proba(X[2000:])
    assert proba.shape == (1000, 2) and np.allclose(proba.sum(axis=1), 1)
    glm_only = roc_auc_score(y[2000:], model._glm_logit(X[2000:]))
    assert roc_auc_score(y[2000:], proba[:, 1]) > 0.75 and glm_only > 0.75
    assert set(model.predict(X[:10])) <= {0, 1}

    seeded = _with_seed(lambda params: GLMBoostClassifier(**(params or {})), {"num_leaves": 4}, 7)
    assert seeded.get_params()["random_state"] == 7 and seeded.get_params()["num_leaves"] == 4

    with pytest.raises(ValueError):
        GLMBoostClassifier(n_estimators=5).fit(X[:30], np.arange(30) % 3)


def test_glm_boost_uses_logits_of_target_encoded_columns():
    from src.genML.pipeline.glm_boost import GLMBoostClassifier, te_column_indices

    assert te_column_indices(["x1", "shop_te", "shop_count", "x1__cents_te"]) == [1, 3]
    X = np.column_stack([np.linspace(0.01, 0.99, 50), np.linspace(-3, 3, 50)])
    model = GLMBoostClassifier(te_columns=[0])
    design = model._design(X, fit=True)
    raw = np.column_stack([np.log(X[:, 0] / (1 - X[:, 0])), np.sign(X[:, 1]) * np.log1p(np.abs(X[:, 1]))])
    np.testing.assert_allclose(design, (raw - raw.mean(axis=0)) / raw.std(axis=0), atol=1e-9)


def test_glm_lightgbm_digits_and_search_space_end_to_end(tmp_path):
    data = _write_problem(tmp_path / "data")
    spec = _spec(data, models=["GLM-LightGBM", "LightGBM"], ensemble={"method": "none"},
                 budget={"n_trials": 2, "timeout_per_model": 120, "tuning_folds": 2, "max_trees": 400,
                         "search_space": {"GLM-LightGBM": {"num_leaves": [2, 4]}}},
                 feature_engineering={"base": "raw", "encodings": {
                     "target_encode": "auto", "numeric_as_categorical": "all", "digits": "auto",
                     "digit_keys": ["cents", "lead2"]}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    reports = tmp_path / "out" / "toy" / summary["run_id"] / "reports"
    features = json.loads((reports / "feature_engineering_report.json").read_text())
    assert {"x1__cents_te", "x2__lead2_te", "x1_te", "shop_te"} <= set(features["features_used"])
    training = json.loads((reports / "model_training_report.json").read_text())
    glm = training["all_results"]["GLM-LightGBM"]
    assert glm["tuned"] is True and 2 <= glm["best_params"]["num_leaves"] <= 4
    assert 1 <= glm["best_params"]["n_estimators"] <= 400
    assert glm["oof_score"] > 0.8
