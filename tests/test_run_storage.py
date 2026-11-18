"""Run storage: Parquet between stages, pruning of unneeded runs, the job worker's memory cap."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.genML import jobs, pruning
from src.genML.pipeline import dataset


# --------------------------------------------------------------------------- #
# Stage tables
# --------------------------------------------------------------------------- #

def _frame(n=6):
    return pd.DataFrame({
        "id": np.arange(n),
        "text": ["a", np.nan, "b", "a", np.nan, "c"][:n],
        "num": np.array([1.5, np.nan, 3.0, 4.0, 5.0, 6.0][:n], dtype=np.float32),
        "cat": pd.Categorical(["x", None, "y", "x", "y", None][:n]),
        "flag": [True, False, True, True, False, False][:n],
    })


def test_stage_tables_round_trip_as_parquet(tmp_path):
    tmp_path = tmp_path / "data"
    tmp_path.mkdir()
    train, test = _frame(), _frame(4)
    assert dataset.save_stage_tables(train, test, tmp_path) == "parquet"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test_data.parquet", "train_data.parquet"]
    back = dataset.load_stage_table("train", tmp_path)
    pd.testing.assert_frame_equal(back, train)
    assert back["text"].isna().sum() == 2 and all(isinstance(v, float) for v in back["text"][back["text"].isna()])
    pd.testing.assert_frame_equal(dataset.load_stage_table("test", tmp_path), test)


def test_stage_tables_fall_back_to_pickle_together(tmp_path):
    tmp_path = tmp_path / "data"
    tmp_path.mkdir()
    train = _frame()
    train["mixed"] = [1, "a", 2.5, None, "b", 3]      # Parquet refuses mixed-type columns
    assert dataset.save_stage_tables(train, _frame(4), tmp_path) == "pickle"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test_data.pkl", "train_data.pkl"]
    pd.testing.assert_frame_equal(dataset.load_stage_table("train", tmp_path), train)


def test_old_runs_pickles_still_load(tmp_path):
    _frame().to_pickle(tmp_path / "train_data.pkl")
    pd.testing.assert_frame_equal(dataset.load_stage_table("train", tmp_path), _frame())


def test_pruned_run_gives_ids_only_when_asked(tmp_path):
    data = tmp_path / "run" / "data"
    data.mkdir(parents=True)
    pd.DataFrame({"id": [7, 8]}).to_parquet(data / dataset.TEST_IDS_FILE)
    (tmp_path / "run" / pruning.PRUNED_FILE).write_text("{}")
    assert dataset.load_stage_table("test", data, ids_only_ok=True)["id"].tolist() == [7, 8]
    with pytest.raises(FileNotFoundError, match="pruned"):
        dataset.load_stage_table("test", data)
    with pytest.raises(FileNotFoundError):
        dataset.load_stage_table("train", data, ids_only_ok=True)


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #

def _run(root, run_id, score=None, finished=True, old=False):
    run_dir = root / "toy" / run_id
    for sub in ("data", "features", "reports", "predictions"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    dataset.save_stage_tables(_frame(), _frame(4), run_dir / "data")
    for name in ("X_train.npy", "X_test.npy", "fold_train_0.npy", "fold_test_0.npy"):
        np.save(run_dir / "features" / name, np.zeros((50, 4), dtype=np.float32))
    np.save(run_dir / "features" / "final_test_predictions.npy", np.zeros(4))
    (run_dir / "features" / "feature_names.txt").write_text("a\nb\n")
    (run_dir / "reports" / "resolved_spec.json").write_text(json.dumps({"id_column": "id"}))
    if finished:
        (run_dir / "run.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}))
        if score is not None:
            (run_dir / "reports" / "final_predictions.json").write_text(json.dumps(
                {"final_experiment_id": f"exp_{run_id}", "final_score": score, "greater_is_better": True}))
    if old:
        stamp = time.time() - 2 * 86400
        for path in [run_dir, *run_dir.rglob("*")]:
            os.utime(path, (stamp, stamp))
    return run_dir


def _big_files_left(run_dir):
    return sorted(p.name for p in (run_dir / "features").glob("*.npy") if p.name != "final_test_predictions.npy") \
        + sorted(p.name for p in (run_dir / "data").iterdir() if p.name != dataset.TEST_IDS_FILE)


def test_prune_keeps_what_is_needed_and_frees_the_rest(tmp_path):
    root = tmp_path / "outputs"
    best = _run(root, "20261101_000000_001", score=0.9)
    worse = _run(root, "20261101_000000_002", score=0.8)
    crashed = _run(root, "20261101_000000_003", finished=False, old=True)
    running = _run(root, "20261101_000000_004", finished=False)
    noise_target = _run(root, "20261101_000000_005", score=0.7)
    anchor = _run(root, "20261101_000000_006", score=0.6)
    jobs.enqueue("toy", kind="noise", options={"run_id": noise_target.name}, outputs_root=root)
    (root / "toy" / "autopilot").mkdir()
    (root / "toy" / "autopilot" / "state.json").write_text(json.dumps(
        {"anchors": {"noise_runs": [anchor.name]}, "cycles": [{"run_id": worse.name}]}))

    kept = pruning.protected_runs("toy", root)
    assert set(kept) >= {best.name, running.name, noise_target.name, anchor.name}   # + the noise job's own id
    assert worse.name not in kept and crashed.name not in kept

    dry = pruning.prune_problem("toy", root, dry_run=True)
    assert {r["run_id"] for r in dry["pruned"]} == {worse.name, crashed.name} and dry["freed_bytes"] > 0
    assert _big_files_left(worse)                                        # a dry run deletes nothing

    report = pruning.prune_problem("toy", root)
    assert {r["run_id"] for r in report["pruned"]} == {worse.name, crashed.name}
    for run_dir in (worse, crashed):
        assert _big_files_left(run_dir) == []
        assert (run_dir / pruning.PRUNED_FILE).exists()
        assert (run_dir / "features" / "final_test_predictions.npy").exists()
        assert (run_dir / "features" / "feature_names.txt").exists()
        assert pd.read_parquet(run_dir / "data" / dataset.TEST_IDS_FILE)["id"].tolist() == [0, 1, 2, 3]
    for run_dir in (best, running, noise_target, anchor):
        assert len(_big_files_left(run_dir)) == 6
    assert pruning.prune_problem("toy", root)["pruned"] == []            # nothing left to do


def test_unreadable_autopilot_state_keeps_every_run(tmp_path):
    root = tmp_path / "outputs"
    _run(root, "20261101_000000_001", score=0.9)
    worse = _run(root, "20261101_000000_002", score=0.8)
    (root / "toy" / "autopilot").mkdir()
    (root / "toy" / "autopilot" / "state.json").write_text("{not json")
    assert pruning.prune_problem("toy", root)["pruned"] == []
    assert _big_files_left(worse)


def test_worker_prunes_after_a_job(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    _run(root, "20261101_000000_001", score=0.9)
    worse = _run(root, "20261101_000000_002", score=0.8)
    job = jobs.enqueue("toy", outputs_root=root)
    monkeypatch.setattr(jobs, "_launch", lambda job, outputs_root=None: (
        [sys.executable, "-c", "print('done')"], str(tmp_path), None, None))
    finished = jobs.run_one(root)
    assert finished["status"] == "failed"                  # the fake job wrote no run.json
    assert _big_files_left(worse) == []
    assert "pruned after this job" in (root / "jobs" / f"job_{job['id']}.log").read_text()


# --------------------------------------------------------------------------- #
# Memory cap
# --------------------------------------------------------------------------- #

def test_memory_cap_follows_the_machine_unless_set(monkeypatch):
    monkeypatch.setattr(jobs, "_machine_memory", lambda: 24 * 1024 ** 3)
    monkeypatch.delenv(jobs.MEMORY_CAP_ENV, raising=False)
    assert jobs.memory_cap_bytes() == 24 * 1024 ** 3
    monkeypatch.setenv(jobs.MEMORY_CAP_ENV, "16G")
    assert jobs.memory_cap_bytes() == 16 * 1024 ** 3
    monkeypatch.setenv(jobs.MEMORY_CAP_ENV, "512MiB")
    assert jobs.memory_cap_bytes() == 512 * 1024 ** 2
    monkeypatch.setenv(jobs.MEMORY_CAP_ENV, "off")
    assert jobs.memory_cap_bytes() is None


def test_machine_memory_reads_meminfo():
    assert jobs._machine_memory() > 256 * 1024 ** 2


def test_capped_command_uses_a_scope_without_swap(monkeypatch):
    monkeypatch.setenv(jobs.MEMORY_CAP_ENV, "8G")
    monkeypatch.setitem(jobs._SCOPE_CHECK, "result", (True, "ok"))
    command, unit, note = jobs._capped(["python", "-m", "x"], {"id": 3, "run_id": "20261116_101010_001"})
    assert unit == "genml-job-3-20261116_101010_001"
    assert command[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert f"MemoryMax={8 * 1024 ** 3}" in command and "MemorySwapMax=0" in command
    assert command[-3:] == ["python", "-m", "x"] and "8.0 GiB" in note

    monkeypatch.setitem(jobs._SCOPE_CHECK, "result", (False, "no user bus"))
    command, unit, note = jobs._capped(["python"], {"id": 3, "run_id": "r"})
    assert command == ["python"] and unit is None and "no user bus" in note
    monkeypatch.setitem(jobs._SCOPE_CHECK, "result", (True, "ok"))
    assert "--collect" not in jobs._capped(["python"], {"id": 3})[0]       # the worker reads a job scope's result
    command, unit, _ = jobs.memory_scope(["claude", "-p"], "genml-session plan/x", collect=True)
    assert "--collect" in command and unit == "genml-session_plan_x" and command[-2:] == ["claude", "-p"]


def test_out_of_memory_job_is_reported(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    job = jobs.enqueue("toy", outputs_root=root)
    monkeypatch.setattr(jobs, "_launch", lambda job, outputs_root=None: (
        [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"], str(tmp_path), None, None))
    monkeypatch.setattr(jobs, "_capped", lambda command, job: (command, "genml-job-test", "memory cap: 1.0 GiB"))
    monkeypatch.setattr(jobs, "_scope_result", lambda unit: "oom-kill")
    finished = jobs.run_one(root)
    assert finished["status"] == "failed" and "out of memory" in finished["error"]
    assert "killed: the job needed more memory" in (root / "jobs" / f"job_{job['id']}.log").read_text()


def _scope_works():
    if os.environ.get("CI"):
        return False
    try:
        return jobs._scope_available()[0]
    except Exception:
        return False


@pytest.mark.skipif(not _scope_works(), reason="no systemd user scope on this machine")
def test_real_scope_kills_a_job_over_its_cap(monkeypatch):
    monkeypatch.setenv(jobs.MEMORY_CAP_ENV, "150M")
    hog = [sys.executable, "-c", "x = [bytearray(10_000_000) for _ in range(60)]; print('survived')"]
    command, unit, _ = jobs._capped(hog, {"id": 0, "run_id": f"test{os.getpid()}"})
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "survived" not in result.stdout
    assert jobs._scope_result(unit) == "oom-kill"
