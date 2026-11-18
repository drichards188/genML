"""Real-data capabilities: several tables, grouped and time-ordered folds, label-free features, new metrics."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.genML.features.derived import (add_group_stats, apply, check_settings, evaluate, expression_columns)
from src.genML.pipeline.metrics import _amex_metric, _normalized_gini, get_metric
from src.genML.pipeline.oof import (fold_signature, make_folds, run_folds, tuning_fold_indices)
from src.genML.pipeline.problem import ProblemSpec, locate_data_dir
from src.genML.pipeline.tables import assemble, check_tables


# --------------------------------------------------------------------------- #
# Several tables
# --------------------------------------------------------------------------- #

def _relational(folder):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cid": [1, 2, 3], "amt": [10.0, 20.0, 30.0], "y": [0, 1, 0]}).to_csv(folder / "app_train.csv",
                                                                                        index=False)
    pd.DataFrame({"cid": [4, 5], "amt": [40.0, 50.0]}).to_csv(folder / "app_test.csv", index=False)
    pd.DataFrame({"loan": [11, 12, 13, 14], "cid": [1, 1, 2, 4], "debt": [1.0, 3.0, 5.0, 7.0],
                  "kind": ["a", "b", "a", "c"], "day": [1, 2, 1, 1]}).to_csv(folder / "loans.csv", index=False)
    pd.DataFrame({"loan": [11, 11, 12, 14], "month": [-2, -1, -1, -1], "late": [0.0, 30.0, 0.0, 60.0]}).to_csv(
        folder / "balance.csv", index=False)
    pd.DataFrame({"cid": [1, 3], "id_01": [0.5, 0.7]}).to_csv(folder / "id_train.csv", index=False)
    pd.DataFrame({"cid": [5], "id-01": [0.9]}).to_csv(folder / "id_test.csv", index=False)
    return {
        "app": {"train": "app_train.csv", "test": "app_test.csv"},
        "loans": {"file": "loans.csv", "parent": "app", "key": "cid", "order_by": "day"},
        "balance": {"file": "balance.csv", "parent": "loans", "key": "loan", "order_by": "month",
                    "aggregate": {"numeric": ["max", "last"]}},
        "identity": {"train": "id_train.csv", "test": "id_test.csv", "parent": "app", "key": "cid", "relation": "one"},
    }


def test_tables_join_roll_up_and_cache(tmp_path):
    tables = _relational(tmp_path / "data")
    assert check_tables(tables) == []
    train, test, report = assemble(tables, tmp_path / "data", tmp_path / "cache")
    assert list(train["cid"]) == [1, 2, 3] and list(test["cid"]) == [4, 5]       # main file order kept
    assert list(train["loans__count"]) == [2, 1, 0]                               # no loans: 0, not missing
    assert train["loans__debt__mean"].iloc[0] == 2.0 and train["loans__debt__trend"].iloc[0] == 2.0
    assert train["loans__kind__last"].iloc[0] == "b"                              # last by day
    assert train["loans__balance__late__max__max"].iloc[0] == 30.0                # balance -> loan -> customer
    assert test["loans__balance__late__max__max"].iloc[0] == 60.0
    assert np.isnan(train["loans__debt__mean"].iloc[2])
    assert test["id_01"].tolist()[1] == pytest.approx(0.9)                        # 'id-01' aligned to 'id_01'
    assert train["loans__debt__mean"].dtype == np.float32
    assert report["tables"]["identity"]["relation"] == "one"
    _, _, again = assemble(tables, tmp_path / "data", tmp_path / "cache")
    assert again.get("cached") is True


def test_sliced_roll_up_matches_one_pass(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"cid": np.arange(200), "y": rng.integers(0, 2, 200)}).to_csv(data / "tr.csv", index=False)
    pd.DataFrame({"cid": np.arange(200, 260)}).to_csv(data / "te.csv", index=False)
    pd.DataFrame({"cid": rng.integers(0, 260, 3000), "amt": rng.normal(size=3000), "t": rng.permutation(3000),
                  "kind": rng.choice(list("xyz"), 3000)}).to_csv(data / "events.csv", index=False)
    base = {"main": {"train": "tr.csv", "test": "te.csv"},
            "events": {"file": "events.csv", "parent": "main", "key": "cid", "order_by": "t"}}
    sliced = {**base, "events": {**base["events"], "aggregate": {"partitions": 4}}}
    one, one_test, _ = assemble(base, data, tmp_path / "c1")
    four, four_test, report = assemble(sliced, data, tmp_path / "c4")
    assert report["tables"]["events"]["partitions"] == 4
    pd.testing.assert_frame_equal(one, four[one.columns])
    pd.testing.assert_frame_equal(one_test, four_test[one_test.columns])


def test_tables_are_checked_and_located(tmp_path):
    assert check_tables({"a": {"train": "x.csv"}})
    assert check_tables({"a": {"train": "x", "test": "y"}, "b": {"file": "z", "parent": "missing", "key": "k"}})
    assert check_tables({"a": {"train": "x", "test": "y"}, "b": {"file": "z", "parent": "a", "key": "k",
                                                                  "aggregate": {"numeric": ["last"]}}})  # no order_by
    tables = _relational(tmp_path / "data")
    spec = ProblemSpec(name="rel", source={"type": "csv", "path": str(tmp_path / "data")}, target="y",
                       id_column="cid", tables=tables)
    assert locate_data_dir(spec) == tmp_path / "data"
    with pytest.raises(ValueError):
        ProblemSpec.from_dict({"name": "bad", "tables": {"a": {"file": "only.csv"}}})


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def test_group_and_time_folds():
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    groups = rng.integers(0, 300, n)
    for method in ("group_kfold", "stratified_group_kfold"):
        folds = make_folds(y, {"method": method, "n_splits": 5, "group": "g", "random_state": 1}, groups)
        assert set(folds) == set(range(5))
        assert pd.Series(folds).groupby(groups).nunique().max() == 1            # a group never spans folds
    times = rng.permutation(n)
    folds = make_folds(y, {"method": "time_blocks", "n_splits": 4, "time_column": "t"}, times)
    blocks = pd.Series(times).groupby(folds).agg(["min", "max"])
    assert all(blocks["max"].iloc[i] < blocks["min"].iloc[i + 1] for i in range(3))  # contiguous, in order
    assert tuning_fold_indices({"method": "time_blocks"}, 4, 2) == [2, 3]       # tune on the latest periods
    assert tuning_fold_indices({"method": "stratified_kfold"}, 5, 2) == [0, 1]
    cv = {"method": "group_kfold", "n_splits": 5, "group": "g"}
    assert fold_signature(y, cv, groups) != fold_signature(y, cv, groups[::-1])
    with pytest.raises(ValueError):
        make_folds(y, cv)                                                          # needs the column's values


def test_run_folds_records_the_signature(tmp_path):
    y = np.tile([0, 1], 50)
    frame = pd.DataFrame({"month": np.repeat(np.arange(10), 10)})
    cv = {"method": "group_kfold", "n_splits": 5, "group": "month"}
    folds, signature = run_folds(y, cv, tmp_path / "problem", tmp_path / "features", frame)
    again, same = run_folds(y, cv, tmp_path / "problem", tmp_path / "features")  # later stage: no frame
    assert same == signature and np.array_equal(folds, again)
    with pytest.raises(ValueError):
        ProblemSpec.from_dict({"name": "x", "cv": {"method": "time_blocks"}})   # needs time_column


# --------------------------------------------------------------------------- #
# Label-free features
# --------------------------------------------------------------------------- #

def test_expressions_are_safe_and_correct():
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 0.0, 1.0], "odd name": [1, 2, 3]})
    assert evaluate("a + 2 * b", frame).tolist()[:2] == [7.0, 2.0]
    assert np.isnan(evaluate("a / b", frame).iloc[1])                           # division by zero -> missing
    assert evaluate("floor(b / 2) + (a > 1)", frame).tolist()[:2] == [1.0, 1.0]  # 1 + 0, 0 + 1
    assert evaluate("col('odd name') * 10", frame).tolist() == [10.0, 20.0, 30.0]
    assert evaluate("fillna(a, 0)", frame).tolist() == [1.0, 2.0, 0.0]
    assert expression_columns("log1p(a) - b") == ["a", "b"]
    for bad in ("__import__('os')", "a.real", "(lambda: 1)()", "open('x')", "a if b else 0"):
        with pytest.raises(ValueError):
            evaluate(bad, frame)
    assert check_settings({"derived": {"x": "os.system('ls')"}})
    assert check_settings({"keys": {"k": ["only_one"]}})
    assert check_settings({"group_stats": [{"by": "k", "cols": ["a"], "stats": ["median"]}]})


def test_group_stats_are_past_only_by_default():
    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame({"card": rng.integers(0, 20, n), "amt": rng.normal(size=n), "t": rng.permutation(n)})
    train.loc[::13, "amt"] = np.nan
    test = pd.DataFrame({"card": rng.integers(0, 20, 50), "amt": rng.normal(size=50), "t": n + np.arange(50)})
    added, report = add_group_stats(train, test, [{"by": "card", "cols": ["amt"],
                                                   "stats": ["count", "mean", "std", "max", "nunique"]}],
                                    time_column="t")
    assert report[0]["past_only"] is True
    combined = pd.concat([train, test], ignore_index=True)
    for i in rng.choice(len(combined), 40, replace=False):                       # brute force: rows up to this one
        row = combined.iloc[i]
        past = combined[(combined["card"] == row["card"]) & (combined["t"] <= row["t"])]
        assert row["gs__card__count"] == len(past)
        values = past["amt"].dropna()
        if len(values):
            assert row["gs__card__amt__mean"] == pytest.approx(values.mean(), rel=1e-4, abs=1e-5)
            assert row["gs__card__amt__nunique"] == values.nunique()
        if len(values) > 1:
            assert row["gs__card__amt__std"] == pytest.approx(values.std(), rel=1e-3, abs=1e-4)
    # A train row never sees test rows (they are all later in time)
    assert (train["gs__card__count"] <= train.groupby("card")["card"].transform("size")).all()


def test_drift_filter_drops_columns_that_separate_train_from_test():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"stable": rng.normal(size=3000), "time": np.arange(3000.0), "id": np.arange(3000),
                          "y": rng.integers(0, 2, 3000)})
    test = pd.DataFrame({"stable": rng.normal(size=1000), "time": 3000 + np.arange(1000.0),
                         "id": 3000 + np.arange(1000)})
    report = apply(train, test, {"drift_filter": {"threshold": 0.8, "sample": 1000}}, protected=["id", "y"])
    assert report["drift_filter"]["dropped"] == ["time"]
    assert "stable" in train.columns and "id" in train.columns and "time" not in test.columns


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _amex_reference(y_true, y_pred):
    """The competition's published pandas implementation."""
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})

    def top_four(df):
        df = df.sort_values("prediction", ascending=False)
        weight = np.where(df["target"] == 0, 20, 1)
        cut = weight.cumsum() <= int(0.04 * weight.sum())
        return df["target"][cut].sum() / df["target"].sum()

    def gini(df, by):
        df = df.sort_values(by, ascending=False)
        weight = np.where(df["target"] == 0, 20, 1)
        random = (weight / weight.sum()).cumsum()
        lorentz = (df["target"] * weight).cumsum() / (df["target"] * weight).sum()
        return ((lorentz - random) * weight).sum()

    return 0.5 * (gini(df, "prediction") / gini(df, "target") + top_four(df))


def test_amex_and_gini_metrics():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.25).astype(int)
    pred = 0.6 * y + rng.random(5000)
    assert _amex_metric(y, pred) == pytest.approx(_amex_reference(y, pred), abs=1e-9)
    assert _amex_metric(y, y + 0.01 * rng.random(5000)) == pytest.approx(1.0)
    assert _normalized_gini(y, pred) == pytest.approx(2 * roc_auc_score(y, pred) - 1)
    assert get_metric("amex").output == "probability"
    assert get_metric("normalized_gini").score(y, np.column_stack([1 - pred, pred]), "binary") == pytest.approx(
        _normalized_gini(y, pred))


def test_experiments_accept_the_new_feature_settings():
    from src.genML.experiments import check_patch

    good = {"feature_engineering": {"derived": {"day": "floor(t / 86400)"}, "keys": {"pair": ["a", "b"]},
                                    "group_stats": [{"by": "pair", "cols": ["x"], "stats": ["mean", "deviation"]}],
                                    "time_column": "t", "drift_filter": {"threshold": 0.85}}}
    assert check_patch(good) == []
    assert check_patch({"feature_engineering": {"derived": {"x": "__import__('os')"}}})


# --------------------------------------------------------------------------- #
# End to end: several tables, time-ordered folds, group statistics, drift filter
# --------------------------------------------------------------------------- #

def test_multi_table_time_split_end_to_end(tmp_path):
    from src.genML.flow import run_pipeline

    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    n_train, n_test = 600, 150
    n = n_train + n_test
    card = rng.integers(0, 60, n)
    risk = rng.normal(size=60)
    amount = rng.gamma(2.0, 50.0, n)
    t = np.sort(rng.random(n)) * 1e6
    y = (rng.random(n) < 1 / (1 + np.exp(-(risk[card] + 0.01 * (amount - 100))))).astype(int)
    main = pd.DataFrame({"tid": np.arange(n), "t": t, "card": card, "amount": amount})
    main.iloc[:n_train].assign(fraud=y[:n_train]).to_csv(data / "tx_train.csv", index=False)
    main.iloc[n_train:].to_csv(data / "tx_test.csv", index=False)
    device = pd.DataFrame({"tid": np.arange(0, n, 2), "screen": rng.choice(["s", "m", "l"], n // 2 + n % 2)})
    device[device.tid < n_train].to_csv(data / "dev_train.csv", index=False)
    device[device.tid >= n_train].to_csv(data / "dev_test.csv", index=False)
    pd.DataFrame({"tid": main.tid.iloc[n_train:], "fraud": 0.5}).to_csv(data / "sample_submission.csv", index=False)

    spec = ProblemSpec(
        name="toytx", source={"type": "csv", "path": str(data)}, target="fraud", id_column="tid", task="binary",
        metric="roc_auc", models=["LightGBM"], budget={"tuning": False}, ensemble={"method": "none"},
        tables={"tx": {"train": "tx_train.csv", "test": "tx_test.csv"},
                "device": {"train": "dev_train.csv", "test": "dev_test.csv", "parent": "tx", "key": "tid",
                           "relation": "one"}},
        cv={"method": "time_blocks", "time_column": "t", "n_splits": 4},
        columns={"drop": ["t"]},
        feature_engineering={"base": "raw", "time_column": "t",
                             "derived": {"log_amount": "log1p(amount)"},
                             "group_stats": [{"by": "card", "cols": ["amount"], "stats": ["count", "mean", "deviation"]}],
                             "drift_filter": {"threshold": 0.9, "sample": 2000},
                             "encodings": {"target_encode": ["card"]}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    reports = tmp_path / "out" / "toytx" / summary["run_id"] / "reports"
    final = json.loads((reports / "final_predictions.json").read_text())
    assert final["forward_holdout"]["fold"] == 3 and final["forward_holdout"]["score"] > 0.55
    features = json.loads((reports / "feature_engineering_report.json").read_text())["features_used"]
    assert {"log_amount", "gs__card__count", "gs__card__amount__deviation", "screen", "card_te"} <= set(features)
    assert "t" not in features
    folds = np.load(tmp_path / "out" / "toytx" / f"folds_{final['fold_signature']}.npy")
    assert list(np.flatnonzero(np.diff(folds) < 0)) == []                          # rows are in time order
