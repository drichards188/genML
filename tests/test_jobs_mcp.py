"""Tests for the job queue, the service layer and the MCP server."""

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.genML import jobs, service
from src.genML.pipeline import problem as problem_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy_env(tmp_path, monkeypatch):
    """A declared toy problem in a temp problems/ dir, visible to worker subprocesses too."""
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 60
    x1, x2 = rng.normal(size=n_train + n_test), rng.normal(size=n_train + n_test)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n_train + n_test) > 0).astype(int)
    frame = pd.DataFrame({"id": np.arange(n_train + n_test), "x1": x1, "x2": x2})
    data = tmp_path / "data"
    data.mkdir()
    frame.iloc[:n_train].assign(label=y[:n_train]).to_csv(data / "train.csv", index=False)
    frame.iloc[n_train:].to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame["id"].iloc[n_train:], "label": 0.5}).to_csv(data / "sample_submission.csv",
                                                                          index=False)

    problems = tmp_path / "problems"
    (problems / "toy").mkdir(parents=True)
    (problems / "toy" / "problem.yaml").write_text(yaml.safe_dump({
        "name": "toy", "source": {"type": "csv", "path": str(data)}, "target": "label", "id_column": "id",
        "task": "binary", "metric": "roc_auc", "models": ["XGBoost"], "budget": {"tuning": False},
    }))
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(problem_module, "PROBLEMS_DIR", problems)
    monkeypatch.setattr("src.genML.run_context.OUTPUTS_ROOT", outputs)
    monkeypatch.setenv("GENML_PROBLEMS", str(problems))
    monkeypatch.setenv("GENML_OUTPUTS", str(outputs))
    return outputs


def test_queue_lifecycle(toy_env):
    first = service.start_run("toy", notes="a", start_worker=False)
    second = service.start_run("toy", n_trials=3, notes="b", start_worker=False)
    assert first["status"] == "queued" and first["queue_position"] == 0
    assert second["queue_position"] == 1
    assert second["options"] == {"budget": {"n_trials": 3}}
    assert [j["notes"] for j in service.list_jobs()] == ["b", "a"]

    assert service.cancel_run(second["job_id"])["status"] == "cancelled"
    assert service.get_run_status(second["job_id"])["status"] == "cancelled"


def test_unknown_problem_is_rejected(toy_env):
    with pytest.raises(FileNotFoundError):
        service.start_run("missing", start_worker=False)
    with pytest.raises(service.ServiceError):
        service.start_run("../etc", start_worker=False)


def test_worker_runs_job_in_subprocess(toy_env):
    queued = service.start_run("toy", notes="via worker", start_worker=False)
    finished = jobs.run_one()
    assert finished["id"] == queued["job_id"]
    assert finished["status"] == "completed", (toy_env / "jobs" / f"job_{queued['job_id']}.log").read_text()[-3000:]

    status = service.get_run_status(queued["job_id"])
    assert status["result"]["metric"] == "roc_auc"
    assert 0.5 < status["result"]["cv_score"] <= 1.0
    assert Path(status["result"]["submission"]).exists()

    runs = service.list_runs("toy")
    assert runs[0]["run_id"] == queued["run_id"] and runs[0]["notes"] == "via worker"
    report = service.get_run_report("toy", queued["run_id"], "models")
    assert "XGBoost" in report["models"]
    diagnostics = service.get_feature_diagnostics("toy", queued["run_id"])
    assert diagnostics["columns"][0]["column"] == "x1"  # strongest signal

    preview = service.submit_to_kaggle("toy")
    assert preview["submitted"] is False and preview["run_id"] == queued["run_id"]


def test_failed_job_is_marked_failed(toy_env):
    spec_file = problem_module.PROBLEMS_DIR / "toy" / "problem.yaml"
    spec = yaml.safe_load(spec_file.read_text())
    spec["target"] = "nope"
    spec_file.write_text(yaml.safe_dump(spec))
    queued = service.start_run("toy", start_worker=False)
    finished = jobs.run_one()
    assert finished["status"] == "failed" and "nope" in finished["error"]


def test_mcp_tools_in_process(toy_env):
    from mcp import Client

    from src.genML.mcp_server import build_server

    async def scenario():
        async with Client(build_server()) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert {"start_run", "get_run_status", "compare_runs", "submit_to_kaggle",
                    "get_feature_diagnostics", "cancel_run"} <= set(tools)
            assert tools["list_runs"].annotations.read_only_hint is True
            assert tools["cancel_run"].annotations.destructive_hint is True

            result = await client.call_tool("list_problems", {})
            assert not result.is_error
            assert json.loads(result.content[0].text)["name"] == "toy"

            result = await client.call_tool("start_run", {"problem": "toy", "notes": "from mcp"})
            payload = json.loads(result.content[0].text)
            assert payload["status"] == "queued"
            jobs.cancel_job(payload["job_id"])  # do not leave a worker behind

            result = await client.call_tool("get_run_report", {"problem": "toy", "run_id": "nope"})
            assert result.is_error

    # start_run would spawn a worker; keep the test self-contained
    import src.genML.service as svc
    original = svc.jobs.ensure_worker
    svc.jobs.ensure_worker = lambda root=None: False
    try:
        asyncio.run(scenario())
    finally:
        svc.jobs.ensure_worker = original


@pytest.mark.slow
def test_mcp_stdio_protocol_is_not_corrupted():
    """Native libraries print to stdout at import; the server must keep the protocol stream clean."""
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "src.genML.mcp_server"],
                                   cwd=str(PROJECT_ROOT), env={**os.environ})

    async def scenario():
        async with Client(params, read_timeout_seconds=120) as client:
            result = await client.call_tool("list_problems", {})
            assert not result.is_error
            names = {json.loads(item.text)["name"] for item in result.content}
            assert "fraud-s6e11" in names

    asyncio.run(scenario())
