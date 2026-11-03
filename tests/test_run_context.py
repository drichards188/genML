"""Tests for per-run output folders, run summaries and the run index."""

import csv

import numpy as np
import pandas as pd
import pytest

from src.genML.flow import run_pipeline
from src.genML.pipeline.problem import ProblemSpec
from src.genML.run_context import RunContext, list_runs, load_run


@pytest.fixture
def toy_problem(tmp_path):
    """A small binary problem with a probability (ROC AUC) submission."""
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 60
    x1 = rng.normal(size=n_train + n_test)
    x2 = rng.normal(size=n_train + n_test)
    color = rng.choice(["red", "green", "blue"], size=n_train + n_test)
    target = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n_train + n_test) > 0).astype(int)
    frame = pd.DataFrame({"id": np.arange(n_train + n_test), "x1": x1, "x2": x2, "color": color})

    data_dir = tmp_path / "toy_data"
    data_dir.mkdir()
    train = frame.iloc[:n_train].assign(label=target[:n_train])
    test = frame.iloc[n_train:]
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "label": 0.5}).to_csv(data_dir / "sample_submission.csv", index=False)

    def make_spec():
        return ProblemSpec(
            name="toy",
            source={"type": "csv", "path": str(data_dir)},
            target="label",
            id_column="id",
            task="binary",
            metric="roc_auc",
            models=["XGBoost"],
            budget={"tuning": False},
        )

    return make_spec, tmp_path / "runs_root"


def test_run_writes_isolated_folder(toy_problem):
    make_spec, root = toy_problem
    summary = run_pipeline(spec=make_spec(), setup_log=False, outputs_root=root)

    assert summary["status"] == "completed", summary.get("error")
    run_dir = root / "toy" / summary["run_id"]
    for name in ("data", "features", "models", "reports", "submission.csv", "run.json"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "reports" / "resolved_spec.json").exists()

    assert summary["metric"] == "roc_auc"
    assert summary["best_model"] == "XGBoost"
    assert 0.5 < summary["cv_score"] <= 1.0

    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id", "label"]
    assert submission["label"].between(0, 1).all() and submission["label"].nunique() > 2


def test_runs_do_not_overwrite_each_other(toy_problem):
    make_spec, root = toy_problem
    first = run_pipeline(spec=make_spec(), notes="first", setup_log=False, outputs_root=root)
    first_submission = (root / "toy" / first["run_id"] / "submission.csv").read_text()
    second = run_pipeline(spec=make_spec(), notes="second", setup_log=False, outputs_root=root)

    assert first["run_id"] != second["run_id"]
    assert (root / "toy" / first["run_id"] / "submission.csv").read_text() == first_submission

    runs = list_runs("toy", outputs_root=root)
    assert [r["notes"] for r in runs] == ["second", "first"]
    assert load_run("toy", first["run_id"], outputs_root=root)["notes"] == "first"

    with open(root / "toy" / "runs.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["notes"] for row in rows] == ["first", "second"]
    assert all(row["status"] == "completed" for row in rows)


def test_failed_run_is_recorded(toy_problem):
    make_spec, root = toy_problem
    spec = make_spec()
    spec.target = "does_not_exist"
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=root)

    assert summary["status"] == "failed"
    assert "does_not_exist" in summary["error"]
    assert list_runs("toy", outputs_root=root)[0]["status"] == "failed"


def test_run_context_paths(tmp_path):
    run = RunContext("demo", run_id="r1", outputs_root=tmp_path)
    assert run.run_dir == tmp_path / "demo" / "r1"
    assert run.submission_file == tmp_path / "demo" / "r1" / "submission.csv"
