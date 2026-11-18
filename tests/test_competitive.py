"""Tests for OOF storage, blending, leak-safe encodings and the submission log."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.genML.features.encoding import build_encodings, nested_target_encode, target_encode
from src.genML.flow import run_pipeline
from src.genML.pipeline.ensemble import hill_climb, stack
from src.genML.pipeline.oof import ExperimentStore, FoldFeatures, fit_oof, make_folds
from src.genML.pipeline.problem import ProblemSpec
from src.genML.run_context import list_runs


def _write_problem(folder, task="binary", n_train=400, n_test=80, seed=0):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    shop = rng.choice(list("ABCDEF"), size=n)
    shop_effect = pd.Series(shop).map({"A": 1.0, "B": -1.0, "C": 0.5, "D": 0, "E": -0.5, "F": 0.2}).to_numpy()
    signal = x1 + 0.5 * x2 + shop_effect + rng.normal(scale=0.5, size=n)
    if task == "binary":
        target = (signal > 0).astype(int)
        placeholder = 0.5
    else:  # rmsle regression on a positive target
        target = np.exp(1 + 0.3 * signal)
        placeholder = 1.0
    frame = pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "shop": shop})
    folder.mkdir(parents=True, exist_ok=True)
    frame.iloc[:n_train].assign(y=target[:n_train]).to_csv(folder / "train.csv", index=False)
    frame.iloc[n_train:].to_csv(folder / "test.csv", index=False)
    pd.DataFrame({"id": frame["id"].iloc[n_train:], "y": placeholder}).to_csv(
        folder / "sample_submission.csv", index=False)
    return folder


def _spec(data_dir, **overrides):
    base = dict(name="toy", source={"type": "csv", "path": str(data_dir)}, target="y", id_column="id",
                task="binary", metric="roc_auc", models=["XGBoost", "Logistic Regression"],
                budget={"tuning": False})
    base.update(overrides)
    return ProblemSpec(**base)


# --------------------------------------------------------------------------- #
# Encodings
# --------------------------------------------------------------------------- #

def test_target_encoding_is_out_of_fold():
    """A unique-per-row category carries no signal out of fold (in-fold it would be perfect)."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=2000)
    keys = pd.Series([f"row{i}" for i in range(2000)])
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 5, "random_state": 0})
    train_enc, _ = target_encode(keys, keys.iloc[:10], y, folds, "binary", 0, smoothing=1.0)
    assert abs(roc_auc_score(y, train_enc[:, 0]) - 0.5) < 0.05


def test_encodings_capture_category_signal():
    rng = np.random.default_rng(2)
    cat = rng.choice(list("ABCD"), size=3000)
    y = (pd.Series(cat).map({"A": 0.1, "B": 0.4, "C": 0.6, "D": 0.9}) > rng.random(3000)).astype(int).to_numpy()
    train = pd.DataFrame({"cat": cat, "num": rng.integers(0, 5, size=3000)})
    test = train.head(100)
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 5, "random_state": 0})
    tr, te, report = build_encodings(train, test, y, folds, "binary", 0, ["cat"], ["num"],
                                     {"target_encode": "auto", "count_encode": "auto", "pairs": "auto",
                                      "numeric_as_categorical": "auto"})
    assert {"cat_te", "num_te", "cat_count", "cat__num_te", "cat__num_count"} <= set(tr.columns)
    assert report["numeric_as_categorical"] == ["num"]
    assert roc_auc_score(y, tr["cat_te"]) > 0.7
    assert len(te) == 100 and not te.isna().any().any()


def test_multiclass_target_encoding_has_one_column_per_class():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=600)
    keys = pd.Series(rng.choice(list("XYZ"), size=600))
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 3, "random_state": 0})
    tr, te = target_encode(keys, keys.head(5), y, folds, "multiclass", 3, smoothing=5.0)
    assert tr.shape == (600, 3) and te.shape == (5, 3)
    assert np.allclose(tr.sum(axis=1), 1.0)


def test_nested_target_encoding_ignores_the_outer_folds_labels():
    """Fold k's block must not change when fold k's labels change (the cross-fold leak)."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=1500)
    keys = pd.Series(rng.choice(list("ABCDEFGH"), size=1500))
    test_keys = pd.Series(list("ABCDEFGHZ"))
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 5, "random_state": 0})
    blocks = nested_target_encode(keys, test_keys, y, folds, "binary", 0, smoothing=5.0)
    assert len(blocks) == 5
    for outer, (train_block, test_block) in enumerate(blocks):
        flipped = y.copy()
        flipped[folds == outer] = 1 - flipped[folds == outer]
        again_train, again_test = nested_target_encode(keys, test_keys, flipped, folds, "binary", 0,
                                                       smoothing=5.0)[outer]
        np.testing.assert_allclose(train_block, again_train)
        np.testing.assert_allclose(test_block, again_test)
        assert train_block.shape == (1500, 1) and test_block.shape == (9, 1)


def test_single_matrix_target_encoding_does_leak_across_folds():
    """Documents why nesting exists: the single OOF matrix depends on other folds' labels."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=1500)
    keys = pd.Series(rng.choice(list("ABCDEFGH"), size=1500))
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 5, "random_state": 0})
    single, _ = target_encode(keys, keys.head(3), y, folds, "binary", 0, smoothing=5.0)
    flipped = y.copy()
    flipped[folds == 0] = 1 - flipped[folds == 0]
    again, _ = target_encode(keys, keys.head(3), flipped, folds, "binary", 0, smoothing=5.0)
    outside = folds != 0
    assert not np.allclose(single[outside], again[outside])


def test_fit_oof_uses_each_folds_block(tmp_path):
    """A column that is pure label noise in X but replaced per fold must reach the models."""
    rng = np.random.default_rng(6)
    n = 600
    y = rng.integers(0, 2, size=n)
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 3, "random_state": 0})
    X = np.column_stack([rng.normal(size=n), np.zeros(n)]).astype(np.float32)
    X_test = np.zeros((10, 2), dtype=np.float32)
    signal = (y + rng.normal(scale=0.3, size=n)).astype(np.float32).reshape(-1, 1)
    blocks = [signal for _ in range(3)]
    fold_features = FoldFeatures.save(tmp_path, [1], ["sig"], blocks, [np.ones((10, 1))] * 3)
    loaded = FoldFeatures.load(tmp_path)
    assert loaded.columns == ["sig"] and list(loaded.indices) == [1]

    from sklearn.linear_model import LogisticRegression

    def factory(params=None):
        return LogisticRegression()

    plain, _ = fit_oof(factory, None, X, y, X_test, folds, "binary", 2)
    nested, test_pred = fit_oof(factory, None, X, y, X_test, folds, "binary", 2, fold_features=loaded)
    assert roc_auc_score(y, plain[:, 1]) < 0.6
    assert roc_auc_score(y, nested[:, 1]) > 0.9
    assert test_pred.shape == (10, 2)
    assert fold_features.indices.tolist() == [1]


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #

def test_stack_and_hill_climb_beat_or_match_members():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=1000)
    noisy = [np.clip(y + rng.normal(scale=s, size=1000), 0.01, 0.99) for s in (0.6, 0.7, 0.8)]
    oofs = [np.column_stack([1 - p, p]) for p in noisy]
    tests = [o[:50] for o in oofs]
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 5, "random_state": 0})
    member_best = max(roc_auc_score(y, o[:, 1]) for o in oofs)

    stacked, stacked_test, info = stack(y, oofs, tests, folds, "binary", 2)
    assert roc_auc_score(y, stacked[:, 1]) > member_best
    assert stacked_test.shape == (50, 2) and info["method"] == "stack"

    blended, _, info = hill_climb(oofs, tests, lambda p: roc_auc_score(y, p[:, 1]))
    assert roc_auc_score(y, blended[:, 1]) >= member_best
    assert abs(sum(info["weights"]) - 1) < 1e-3  # weights are rounded to 4 decimals


# --------------------------------------------------------------------------- #
# OOF stage through the pipeline
# --------------------------------------------------------------------------- #

def test_oof_stage_stores_experiments_and_picks_final(tmp_path):
    data = _write_problem(tmp_path / "data")
    root = tmp_path / "out"
    summary = run_pipeline(spec=_spec(data), setup_log=False, outputs_root=root)
    assert summary["status"] == "completed", summary.get("error")

    run_dir = root / "toy" / summary["run_id"]
    final = json.loads((run_dir / "reports" / "final_predictions.json").read_text())
    assert set(final["single_scores"]) == {"XGBoost", "Logistic Regression"}
    assert final["ensemble"]["method"] == "stack"
    assert summary["cv_score"] == pytest.approx(final["final_score"])
    assert 0.7 < summary["cv_score"] <= 1.0  # native OOF ROC AUC

    experiments = ExperimentStore(root / "toy").list(final["fold_signature"])
    assert {e.meta["model"] for e in experiments} >= {"XGBoost", "Logistic Regression", "Ensemble (stack)"}
    assert all(e.oof.shape == (400, 2) and e.test.shape == (80, 2) for e in experiments)


def test_runs_share_folds_and_store_ensemble_blends_across_runs(tmp_path):
    from src.genML.pipeline.store_ensemble import ensemble_from_store

    data = _write_problem(tmp_path / "data")
    root = tmp_path / "out"
    first = run_pipeline(spec=_spec(data, models=["XGBoost"]), setup_log=False, outputs_root=root)
    second = run_pipeline(spec=_spec(data, models=["Logistic Regression"]), setup_log=False, outputs_root=root)
    assert first["status"] == second["status"] == "completed"
    assert len(list((root / "toy").glob("folds_*.npy"))) == 1

    summary = ensemble_from_store("toy", method="stack", top=5, outputs_root=root)
    assert summary["status"] == "completed", summary.get("error")
    submission = pd.read_csv(summary["submission"])
    assert list(submission.columns) == ["id", "y"] and len(submission) == 80
    assert submission["y"].between(0, 1).all()
    assert len(list_runs("toy", outputs_root=root)) == 3


def test_rmsle_problem_trains_on_log_scale(tmp_path):
    data = _write_problem(tmp_path / "data", task="regression")
    spec = _spec(data, task="regression", metric="rmsle", models=["XGBoost", "Linear Regression"])
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    assert 0 < summary["cv_score"] < 0.5  # RMSLE on the original scale
    submission = pd.read_csv(summary["submission"])
    assert (submission["y"] > 0).all()


def test_spec_encodings_reach_the_feature_matrix(tmp_path):
    data = _write_problem(tmp_path / "data")
    spec = _spec(data, models=["XGBoost"], ensemble={"method": "none"},
                 feature_engineering={"encodings": {"target_encode": ["shop"], "count_encode": ["shop"]}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    report = json.loads((tmp_path / "out" / "toy" / summary["run_id"] / "reports" /
                         "feature_engineering_report.json").read_text())
    assert report["encodings"]["target_encoded"] == ["shop"]
    assert {"shop_te", "shop_count"} <= set(report["features_used"])
    assert report["nested_target_encoding"] is True
    run_dir = tmp_path / "out" / "toy" / summary["run_id"]
    blocks = json.loads((run_dir / "features" / "fold_blocks.json").read_text())
    assert blocks["columns"] == ["shop_te"] and blocks["n_folds"] == 5
    assert report["features_used"][blocks["indices"][0]] == "shop_te"
    experiments = ExperimentStore(tmp_path / "out" / "toy").list()
    assert all(e.meta.get("nested_encoding") for e in experiments if e.meta.get("model") == "XGBoost")


def test_raw_base_keeps_original_columns_plus_encodings(tmp_path):
    data = _write_problem(tmp_path / "data")
    spec = _spec(data, models=["XGBoost"], ensemble={"method": "none"},
                 feature_engineering={"base": "raw", "encodings": {"target_encode": ["shop"]}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    report = json.loads((tmp_path / "out" / "toy" / summary["run_id"] / "reports" /
                         "feature_engineering_report.json").read_text())
    assert report["features_used"] == ["x1", "x2", "shop", "shop_te"]
    assert report["feature_engineering_method"] == "raw"


# --------------------------------------------------------------------------- #
# Submission log (Kaggle calls mocked)
# --------------------------------------------------------------------------- #

def test_submit_and_refresh_scores(tmp_path, monkeypatch):
    from src.genML import submissions
    from src.genML.pipeline import kaggle

    data = _write_problem(tmp_path / "data")
    root = tmp_path / "out"
    summary = run_pipeline(spec=_spec(data, models=["XGBoost"]), setup_log=False, outputs_root=root)

    sent = {}
    monkeypatch.setattr(submissions, "_competition", lambda name: "toy-competition")
    monkeypatch.setattr(kaggle, "submit_file",
                        lambda comp, path, message: sent.update(comp=comp, path=path, message=message) or "ok")
    row = submissions.submit_run("toy", message="first try", outputs_root=root)
    assert sent["comp"] == "toy-competition"
    assert f"[genml {summary['run_id']}" in sent["message"]
    assert row["run_id"] == summary["run_id"]

    monkeypatch.setattr(kaggle, "list_submissions", lambda comp: [{
        "ref": "123", "date": "now", "description": sent["message"], "status": "COMPLETE",
        "public_score": "0.91234", "private_score": None, "file_name": "submission.csv",
    }])
    rows = submissions.refresh_scores("toy", outputs_root=root)
    assert rows[0]["public_score"] == "0.91234" and rows[0]["status"] == "COMPLETE"


def test_tuned_xgboost_runs(tmp_path):
    """Regression: tuning used tree_method='gpu_hist', which XGBoost 3.x rejects."""
    data = _write_problem(tmp_path / "data")
    spec = _spec(data, models=["XGBoost"], budget={"n_trials": 2, "timeout_per_model": 120})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    report = json.loads((tmp_path / "out" / "toy" / summary["run_id"] / "reports" /
                         "model_training_report.json").read_text())
    assert report["all_results"]["XGBoost"]["tuned"] is True


def test_gbdt_tuning_early_stops_and_fixes_tree_count(tmp_path):
    """XGBoost and LightGBM tune on the shared folds; the early-stopped tree count is kept."""
    data = _write_problem(tmp_path / "data")
    spec = _spec(data, models=["XGBoost", "LightGBM"], ensemble={"method": "none"},
                 budget={"n_trials": 2, "timeout_per_model": 120, "tuning_folds": 2},
                 feature_engineering={"base": "raw", "encodings": {"target_encode": ["shop"]}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=tmp_path / "out")
    assert summary["status"] == "completed", summary.get("error")
    report = json.loads((tmp_path / "out" / "toy" / summary["run_id"] / "reports" /
                         "model_training_report.json").read_text())
    for name in ("XGBoost", "LightGBM"):
        result = report["all_results"][name]
        assert result["tuned"] is True
        assert 1 <= result["best_params"]["n_estimators"] < 6000
    assert report["all_results"]["LightGBM"]["best_params"]["subsample_freq"] == 1


# --------------------------------------------------------------------------- #
# Diagnostics: paired comparison and seed noise
# --------------------------------------------------------------------------- #

def test_paired_compare_and_verdict():
    from src.genML.pipeline.diagnostics import paired_compare, verdict

    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=3000)
    folds = make_folds(y, {"method": "stratified_kfold", "n_splits": 5, "random_state": 0})
    weak = y + rng.normal(scale=1.5, size=3000)
    strong = y + rng.normal(scale=0.5, size=3000)

    def score(rows, pred):
        return roc_auc_score(y[rows], pred[rows])

    better = paired_compare(score, weak, strong, folds, n_boot=50)
    assert better["diff"] > 0 and better["folds_won"] == 5
    assert verdict(better)["accept"] is True
    same = paired_compare(score, weak, weak, folds, n_boot=20)
    assert same["diff"] == 0 and verdict(same)["accept"] is False
    # Large seed noise raises the bar enough to reject a real but small gain
    assert verdict(better, sigma_seed=1.0)["accept"] is False


def test_noise_test_and_compare_on_a_run(tmp_path):
    from src.genML.pipeline.diagnostics import compare_experiments, seed_noise_test

    data = _write_problem(tmp_path / "data")
    root = tmp_path / "out"
    spec = _spec(data, models=["XGBoost", "Logistic Regression"],
                 feature_engineering={"base": "raw", "encodings": {"target_encode": ["shop"]}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=root)
    assert summary["status"] == "completed", summary.get("error")
    run_id = summary["run_id"]

    report = seed_noise_test("toy", run_id, "XGBoost", n_seeds=3, outputs_root=root, n_boot=20)
    assert len(report["seeds"]) == 3 and len(report["scores"]) == 3
    assert report["sigma_seed"] >= 0 and report["threshold"] >= 0
    assert (root / "toy" / run_id / "reports" / "noise_test.json").exists()

    result = compare_experiments("toy", f"{run_id}__logistic_regression", f"{run_id}__xgboost",
                                 outputs_root=root, n_boot=20)
    assert result["seed_noise_measured"] is True
    assert result["n_folds"] == 5 and "accept" in result
