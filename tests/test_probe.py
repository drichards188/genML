"""The probe harness (autopilot/probe.py): cheap screens that never wear out the judge's rows."""

import json

import numpy as np
import pandas as pd
import pytest

from src.genML.autopilot import probe
from src.genML.autopilot.judge import HOLDOUT, ROWS_FILE, choose_rows
from src.genML.pipeline.oof import ExperimentStore, FoldFeatures

FAST = {"n_estimators": 40, "num_leaves": 7, "min_child_samples": 20, "n_jobs": 2}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _toy(tmp_path, cv_method="time_blocks", n=2500, blocks=True):
    """A finished best run: features, folds, targets, raw table, the judge's row codes and anchors."""
    rng = np.random.default_rng(0)
    signal, hidden = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < _sigmoid(1.5 * signal + 1.5 * hidden)).astype(float)
    folds = np.repeat(np.arange(5), n // 5) if cv_method == "time_blocks" else rng.permutation(n) % 5
    outputs = tmp_path / "outputs"
    problem_dir = outputs / "toy"
    features = problem_dir / "R1" / "features"
    features.mkdir(parents=True)
    X = np.column_stack([signal + rng.normal(scale=0.5, size=n), rng.normal(size=n)]).astype(np.float32)
    np.save(features / "X_train.npy", X)
    (features / "feature_names.txt").write_text("signal_proxy\nencoded\n")
    if blocks:                                       # a nested encoding in column 1, different per fold
        FoldFeatures.save(features, [1], ["encoded"], [np.full((n, 1), k, dtype=np.float32) for k in range(5)],
                          [np.zeros((3, 1), dtype=np.float32) for _ in range(5)])
    np.save(problem_dir / "folds_sig1.npy", folds)
    ExperimentStore(problem_dir).save_targets("sig1", y, {})
    reports, data = problem_dir / "R1" / "reports", problem_dir / "R1" / "data"
    reports.mkdir()
    data.mkdir()
    (reports / "resolved_spec.json").write_text(json.dumps({
        "cv": {"method": cv_method, "time_column": "t"}, "feature_engineering": {}, "id_column": "id"}))
    pd.DataFrame({"id": np.arange(n), "t": np.arange(n, dtype=float), "hidden": hidden, "y": y}).to_parquet(
        data / "train_data.parquet")
    codes, scheme = choose_rows(folds, y, "binary", cv_method)
    home = problem_dir / "autopilot"
    home.mkdir()
    np.save(home / ROWS_FILE, codes)
    anchors = {"fold_signature": "sig1", "task": "binary", "metric": "roc_auc", "target": "y",
               "cv_method": cv_method, "rows": {"file": str(home / ROWS_FILE), "scheme": scheme}}
    return {"outputs": outputs, "anchors": anchors, "folds": folds, "codes": codes, "hidden": hidden, "y": y}


def _probe(tmp_path, toy, bar=0.01):
    folder = tmp_path / "wt" / ".autopilot"
    probe.write_context(folder, problem="toy", outputs_root=toy["outputs"], run_id="R1", anchors=toy["anchors"],
                        judge_bar={"threshold": bar, "source": "test"})
    return probe.load(folder / probe.CONTEXT_FILE)


def test_time_probes_do_not_use_the_judges_block():
    folds = np.repeat(np.arange(6), 10)
    codes, _ = choose_rows(folds, np.zeros(60), "binary", "time_blocks")
    rows, info = probe.probe_rows(folds, codes, "time_blocks")
    assert np.all(rows[folds == 5] == 0)                                   # the judge's block: never used
    assert np.all(rows[folds == 4] == probe.PROBE_EVAL) and np.all(rows[folds < 4] == probe.PROBE_TRAIN)
    assert info["kind"] == "time_forward" and info["fold"] == 4
    with pytest.raises(ValueError):
        probe.probe_rows(np.repeat(np.arange(2), 10), np.zeros(20), "time_blocks")


def test_other_probes_never_score_a_held_back_row_and_do_not_point_at_them():
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.3).astype(float)
    folds = rng.permutation(1000) % 5
    codes, _ = choose_rows(folds, y, "binary", "stratified_kfold")
    rows, info = probe.probe_rows(folds, codes, "stratified_kfold")
    held = codes == HOLDOUT
    assert not np.any((rows == probe.PROBE_EVAL) & held)                   # never scored
    assert np.all(folds[rows == probe.PROBE_EVAL] == 0) and np.all(rows[folds != 0] == probe.PROBE_TRAIN)
    # the unused rows are not the held-back rows: the file must not tell a session which ones they are
    unused = rows == 0
    assert unused.sum() > 0 and not np.array_equal(unused, held) and held[unused].mean() < 0.6
    assert info["kind"] == "one_fold"


def test_a_probe_finds_a_real_feature_and_logs_each_call(tmp_path):
    toy = _toy(tmp_path)
    p = _probe(tmp_path, toy, bar=0.01)
    assert set(p.eval_rows) == set(np.where(toy["folds"] == 3)[0])
    hidden = p.raw(["hidden"])["hidden"].to_numpy()
    real = p.compare("the hidden signal", add={"hidden": hidden}, params=FAST)
    assert real["gain"] > 0.03 and real["z"] > 3 and real["gain_vs_bar"] == pytest.approx(real["gain"] / 0.01)
    noise = p.compare("pure noise", add={"noise": np.random.default_rng(5).normal(size=len(hidden))}, params=FAST)
    assert abs(noise["gain"]) < max(3 * noise["noise"], 0.01)
    log = probe.read_log(p.folder / probe.LOG_FILE)
    assert [r["id"] for r in log] == ["P01", "P02"] and log[0]["label"] == "the hidden signal"
    assert log[0]["added"] == ["hidden"] and log[0]["eval"] == "time_forward"
    assert log[0]["eval_rows"] == len(p.eval_rows) and log[0]["train_rows"] == len(p.train_rows)


def test_the_base_is_fitted_once_and_cached_on_disk(tmp_path, monkeypatch):
    toy = _toy(tmp_path)
    p = _probe(tmp_path, toy)
    calls = []
    fit = probe.Probe._fit_predict

    def counted(self, *args, **kwargs):
        calls.append(1)
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(probe.Probe, "_fit_predict", counted)
    p.compare("one", add={"a": toy["hidden"]}, params=FAST)
    p.compare("two", variant_params={"num_leaves": 3}, params=FAST)
    assert len(calls) == 2 + 2 + 2                                         # base once, then two variants
    again = probe.load(p.context_path)
    again.compare("three", drop=["signal_proxy"], params=FAST)
    assert len(calls) == 8                                                 # the base came from the cache


def test_probes_use_each_folds_own_encodings(tmp_path):
    toy = _toy(tmp_path)
    p = _probe(tmp_path, toy)
    X_train, X_eval, _ = p._base_matrices(None)
    assert np.all(X_eval[:, 1] == 3) and np.all(X_train[:, 1] == 3)       # fold 3's block, not the stored column
    plain = _probe(tmp_path / "plain", _toy(tmp_path / "plain", blocks=False))
    assert plain.fold_features is None and plain._base_matrices(None)[1].shape[1] == 2


def test_probe_inputs_are_checked(tmp_path):
    toy = _toy(tmp_path)
    p = _probe(tmp_path, toy)
    with pytest.raises(ValueError, match="target"):
        p.raw(["y"])
    with pytest.raises(ValueError, match="one value per training row"):
        p.compare("short", add={"a": np.zeros(5)}, params=FAST)
    with pytest.raises(ValueError, match="already exists"):
        p.compare("clash", add={"encoded": toy["hidden"]}, params=FAST)
    with pytest.raises(TypeError, match="numeric"):
        p.compare("text", add={"a": np.array(["x"] * len(toy["y"]))}, params=FAST)
    with pytest.raises(ValueError, match="do not exist"):
        p.compare("drop", drop=["nope"], params=FAST)
    with pytest.raises(ValueError, match="nothing to compare"):
        p.compare("empty", params=FAST)
    with pytest.raises(ValueError, match="label"):
        p.compare(" ", drop=["encoded"], params=FAST)
    with pytest.raises(ValueError, match="non-negative"):
        p.compare("bad weights", weights=-np.ones(len(toy["y"])), params=FAST)
    seen = []
    p.compare("weights see the probe's training rows", weights=lambda rows: seen.append(rows) or np.ones(len(rows)),
              params=FAST)
    assert np.array_equal(seen[0], p.train_rows)
    assert len(probe.read_log(p.folder / probe.LOG_FILE)) == 1             # refused calls are not logged


def test_a_pruned_best_run_has_no_probe_context(tmp_path):
    toy = _toy(tmp_path)
    (toy["outputs"] / "toy" / "R1" / "features" / "X_train.npy").unlink()
    with pytest.raises(FileNotFoundError, match="X_train"):
        probe.write_context(tmp_path / "ctx", problem="toy", outputs_root=toy["outputs"], run_id="R1",
                            anchors=toy["anchors"])
