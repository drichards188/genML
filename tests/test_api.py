"""Tests for the run-aware dashboard API endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from src.genML.api import create_app


@pytest.fixture
def outputs(isolate_outputs):
    """A fake outputs/ tree with two runs of one problem (cwd is a temp dir)."""
    root = isolate_outputs / "outputs"
    for run_id, score in (("20261101_100000_000", 0.91), ("20261102_100000_000", 0.93)):
        run_dir = root / "fraud" / run_id
        (run_dir / "reports").mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id, "problem": "fraud", "status": "completed",
            "metric": "roc_auc", "cv_score": score,
        }))
        (run_dir / "reports" / "model_training_report.json").write_text(json.dumps({
            "best_model": "XGBoost", "best_score": score, "scoring_metric": "roc_auc", "all_results": {},
        }))
        (run_dir / "pipeline.log").write_text(f"log for {run_id}\n")
    return root


@pytest.fixture
def client(outputs):
    with TestClient(create_app()) as test_client:
        yield test_client


def test_problems_and_runs(client):
    problems = client.get("/api/problems").json()
    assert problems == [{
        "problem": "fraud", "runs": 2, "latest_run": "20261102_100000_000",
        "latest_status": "completed", "metric": "roc_auc",
    }]
    runs = client.get("/api/problems/fraud/runs").json()
    assert [r["run_id"] for r in runs] == ["20261102_100000_000", "20261101_100000_000"]
    assert client.get("/api/problems/fraud/runs/20261101_100000_000").json()["cv_score"] == 0.91


def test_models_reads_requested_run(client):
    assert client.get("/api/models", params={"problem": "fraud", "run_id": "20261101_100000_000"}).json()[
        "best_score"] == 0.91
    # Without parameters: the newest run
    assert client.get("/api/models").json()["best_score"] == 0.93


def test_per_run_log(client):
    response = client.get("/api/logs/20261101_100000_000")
    assert response.status_code == 200
    assert "log for 20261101_100000_000" in response.json()["content"]


def test_rejects_path_traversal(client):
    assert client.get("/api/problems/..%2F..%2Fetc/runs").status_code in (400, 404)
    assert client.get("/api/reports", params={"problem": "..", "run_id": "x"}).status_code == 400
    assert client.get("/api/reports/..%2Fsecret", params={"problem": "fraud",
                                                          "run_id": "20261101_100000_000"}).status_code in (403, 404)
