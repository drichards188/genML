"""Autopilot: git guards, the anchored judge, the agent command and a full cycle with a fake Claude CLI."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.genML.autopilot import judge as judging
from src.genML.autopilot import workspace as ws
from src.genML.autopilot.agent import AgentSettings, build_command, parse_reset
from src.genML.autopilot.loop import (MODES_DIR, Autopilot, AutopilotConfig, check_patch_structure, format_status,
                                      load_state, validate_explore, validate_plan)
from src.genML.pipeline.oof import ExperimentStore, fold_signature, make_folds

PROBLEM = "toy"
CV = {"method": "stratified_kfold", "n_splits": 5, "shuffle": True, "random_state": 42}


# --------------------------------------------------------------------------- #
# Fixtures: a git repository and a measured problem
# --------------------------------------------------------------------------- #

def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "work")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    files = {
        ".gitignore": "outputs/\n.venv/\n",
        "src/genML/pipeline/diagnostics.py": "JUDGE = 1\n",
        "src/genML/jobs.py": "# jobs\n",
        "src/pkg.py": "VALUE = 1\n",
        "tests/test_old.py": "def test_old():\n    assert True\n",
    }
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "initial")
    return path


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _save_run(problem_dir: Path, run_id: str, oof: np.ndarray, signature: str, score: float) -> str:
    store = ExperimentStore(problem_dir)
    proba = np.column_stack([1 - oof, oof])
    experiment = store.save("LightGBM", run_id, proba, proba[:10], {"fold_signature": signature, "model": "LightGBM",
                                                                    "oof_score": score})
    reports = problem_dir / run_id / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "final_predictions.json").write_text(json.dumps({
        "final_experiment_id": experiment.experiment_id, "final_score": score, "greater_is_better": True,
        "fold_signature": signature}))
    (reports / "resolved_spec.json").write_text(json.dumps({
        "name": PROBLEM, "target": "y", "task": "binary", "metric": "roc_auc", "classes": [0, 1],
        "models": ["LightGBM"], "budget": {"n_trials": 2}, "feature_engineering": {"base": "raw"},
        "ensemble": {"method": "none"}, "cv": CV}))
    (problem_dir / run_id / "run.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}))
    return experiment.experiment_id


@pytest.fixture
def measured(tmp_path, monkeypatch):
    """Outputs with one completed run (the anchor) and a toy problem spec."""
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)
    y = (rng.random(n) < _sigmoid(2 * signal)).astype(int)
    outputs = tmp_path / "outputs"
    problem_dir = outputs / PROBLEM
    signature = fold_signature(y, CV)
    problem_dir.mkdir(parents=True, exist_ok=True)
    np.save(problem_dir / f"folds_{signature}.npy", make_folds(y, CV))
    ExperimentStore(problem_dir).save_targets(signature, y, {})
    base_oof = _sigmoid(2 * signal + rng.normal(scale=1.5, size=n))
    from sklearn.metrics import roc_auc_score

    _save_run(problem_dir, "20260101_000000_000", base_oof, signature, roc_auc_score(y, base_oof))
    # The noise measurement (--baseline): the same settings with two more seeds
    runs = ["20260101_000000_000"]
    for k in (1, 2):
        rerun = _sigmoid(2 * signal + rng.normal(scale=1.5, size=n))
        run_id = f"20260101_00000{k}_000"
        _save_run(problem_dir, run_id, rerun, signature, roc_auc_score(y, rerun) - 0.01)   # never the best
        runs.append(run_id)
    (problem_dir / "noise_runs.json").write_text(json.dumps([{"repeats": 3, "runs": [
        {"run_id": r, "seed": k} for k, r in enumerate(runs)]}]))
    problems = tmp_path / "problems"
    (problems / PROBLEM).mkdir(parents=True)
    (problems / PROBLEM / "problem.yaml").write_text(
        "name: toy\ntarget: y\ntask: binary\nmetric: roc_auc\nmodels: [LightGBM]\n")
    (problems / PROBLEM / "notes.md").write_text("# toy notes\n")
    monkeypatch.setattr("src.genML.pipeline.problem.PROBLEMS_DIR", problems)
    return {"outputs": outputs, "problem_dir": problem_dir, "y": y, "signal": signal, "signature": signature,
            "problems": problems, "rng": rng}


# --------------------------------------------------------------------------- #
# Git guards
# --------------------------------------------------------------------------- #

def test_snapshot_worktree_and_guarded_commit(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    (repo / "src/pkg.py").write_text("VALUE = 2  # uncommitted\n")
    (repo / "src/untracked.py").write_text("NEW = 1\n")
    (repo / "outputs").mkdir()
    (repo / "outputs/big.npy").write_text("ignored data")
    status_before = _git(repo, "status", "--porcelain")

    base = ws.snapshot_working_tree(repo)
    assert _git(repo, "status", "--porcelain") == status_before          # index and files untouched
    assert _git(repo, "rev-parse", "HEAD") != base
    assert "VALUE = 2" in _git(repo, "show", f"{base}:src/pkg.py")
    assert _git(repo, "show", f"{base}:src/untracked.py") == "NEW = 1"
    assert not ws.exists_at(repo, base, "outputs/big.npy")               # ignored files stay out

    ws.create_branch(repo, "autopilot/toy/main", base)
    wt = ws.add_worktree(repo, tmp_path / "wt" / "C001", "autopilot/toy/C001", base)
    (wt / "src/genML/pipeline/diagnostics.py").write_text("JUDGE = 0  # bend the judge\n")
    (wt / "tests/test_old.py").write_text("def test_old():\n    pass\n")
    (wt / "tests/test_new.py").write_text("def test_new():\n    assert 1\n")
    (wt / "src/pkg.py").write_text("VALUE = 3\n")
    (wt / ".autopilot" / "plan.json").write_text("{}")
    for leftover in (".bashrc", ".mcp.json"):                            # a killed sandbox's placeholders
        (wt / leftover).touch()
        (wt / leftover).chmod(0o444)

    info = ws.commit_cycle(wt, base, "C001: try")
    assert info["sandbox_leftovers"] == [".bashrc", ".mcp.json"] and not (wt / ".mcp.json").exists()
    assert info["reverted_protected"] == ["src/genML/pipeline/diagnostics.py"]
    assert info["reverted_tests"] == ["tests/test_old.py"]
    assert sorted(info["files"]) == ["src/pkg.py", "tests/test_new.py"]
    assert (wt / "src/genML/pipeline/diagnostics.py").read_text() == "JUDGE = 1\n"
    assert ws.added_test_functions(wt, base) == 1
    assert "Test User" in _git(wt, "log", "-1", "--format=%an")
    assert "Co-Authored" not in _git(wt, "log", "-1", "--format=%B")

    ws.move_branch(repo, "autopilot/toy/main", info["commit"])
    assert ws.branch_head(repo, "autopilot/toy/main") == info["commit"]
    with pytest.raises(ws.GitError):
        ws.move_branch(repo, "autopilot/toy/main", base)                 # not a fast-forward
    ws.remove_worktree(repo, wt)
    assert not wt.exists() and ws.branch_head(repo, "autopilot/toy/C001") == info["commit"]


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

def test_judge_uses_anchors_and_flags_suspicious_gains(measured):
    outputs, problem_dir, y, signal, rng = (measured[k] for k in ("outputs", "problem_dir", "y", "signal", "rng"))
    anchors = judging.make_anchors(PROBLEM, outputs)
    best = anchors["best"]
    assert anchors["metric"] == "roc_auc" and anchors["n_rows"] == len(y)

    better = _sigmoid(2 * signal + rng.normal(scale=0.7, size=len(y)))
    _save_run(problem_dir, "20260102_000000_000", better, measured["signature"], 0.0)
    result = judging.judge_run(anchors, best, PROBLEM, "20260102_000000_000", outputs, suspicious_ratio=1e6)
    assert result["verdict"] == "accepted" and result["gain"] > result["threshold"]
    assert set(result["holdout"]) == {"candidate", "best"} and result["rows"] == "random_holdout"
    assert anchors["sigma_run"] > 0 and anchors["rows"]["n_holdout"] == round(0.2 * len(y))
    assert result["candidate_score"] > best["score"]          # rescored, not the run's own claim of 0.0
    held = judging.judge_run(anchors, best, PROBLEM, "20260102_000000_000", outputs, suspicious_ratio=1.0)
    assert held["verdict"] == "suspicious"

    same = ExperimentStore(problem_dir).get(best["experiment_id"]).oof[:, 1]
    _save_run(problem_dir, "20260103_000000_000", same, measured["signature"], 0.0)
    assert judging.judge_run(anchors, best, PROBLEM, "20260103_000000_000", outputs)["verdict"] == "rejected"

    _save_run(problem_dir, "20260104_000000_000", better, "otherfolds", 0.0)
    assert judging.judge_run(anchors, best, PROBLEM, "20260104_000000_000", outputs)["verdict"] == "failed"

    targets = problem_dir / "oof" / "_targets" / f"y_{measured['signature']}.npy"
    np.save(targets, 1 - y)                                    # someone rewrote the labels
    tampered = judging.judge_run(anchors, best, PROBLEM, "20260102_000000_000", outputs)
    assert tampered["verdict"] == "integrity_failure"


# --------------------------------------------------------------------------- #
# Agent command, parsing and validation
# --------------------------------------------------------------------------- #

def test_agent_command_limits_tools_and_settings(tmp_path):
    from src.genML.autopilot.policy import load_mode

    header, _ = load_mode(MODES_DIR / "build.md")
    cmd = build_command(AgentSettings(model="opus"), policy=header, settings_path=tmp_path / "s.json",
                        read_dirs=[Path("/data")], system_append="rules", session_id="abc", resume=False)
    joined = " ".join(cmd)
    assert cmd[:2] == ["claude", "-p"] and "--strict-mcp-config" in cmd and "--restricted" in cmd
    assert cmd[cmd.index("--permission-prompts") + 1] == "none"
    assert "WebFetch" not in cmd[cmd.index("--tools") + 1] and "--session-id" in cmd
    assert "--add-dir /data" in joined and cmd[cmd.index("--model") + 1] == "opus"
    resumed = build_command(AgentSettings(), policy=header, settings_path=tmp_path / "s.json", read_dirs=[],
                            system_append="", session_id="abc", resume=True)
    assert resumed[resumed.index("--resume") + 1] == "abc" and "--session-id" not in resumed


def test_reset_time_parsing():
    from datetime import datetime

    now = datetime(2026, 11, 7, 13, 0)
    assert parse_reset("Claude AI usage limit reached|1794000000", now) == 1794000000.0
    assert datetime.fromtimestamp(parse_reset("limit reached, resets 3pm", now)).hour == 15
    assert datetime.fromtimestamp(parse_reset("resets 9am", now)).day == 8      # tomorrow
    assert parse_reset("no time given", now) is None


def test_plan_and_patch_validation():
    idea = {"title": "t", "kind": "settings", "hypothesis": "h", "evidence": "e", "novelty": "adapted",
            "predicted_gain": 0.0003, "spec_patch_yaml": "budget:\n  n_trials: 30\n"}
    assert validate_plan({"decision": "continue", "reason": "r", "idea": idea}) == []
    assert validate_plan({"decision": "conclude", "reason": "done"}) == []
    assert validate_plan({"decision": "continue", "reason": "r", "idea": {**idea, "kind": "code"}})  # no brief
    assert validate_plan({"decision": "maybe", "reason": "r"})
    assert validate_explore({"decision": "continue", "reason": "r", "idea": idea})                 # no probes
    probes = [{"script": "a.py", "question": "q", "finding": "f"}] * 2
    assert validate_explore({"decision": "continue", "reason": "r", "idea": {**idea, "probes": probes}}) == []
    assert validate_explore({"decision": "continue", "reason": "r",
                             "idea": {**idea, "probes": probes, "novelty": "general"}})
    assert check_patch_structure({"budget": {"n_trials": 3}}) == []
    assert check_patch_structure({"cv": {"n_splits": 3}}) and check_patch_structure({"metric": "accuracy"})
    assert check_patch_structure({}) and check_patch_structure({"models": []})


def test_plans_cite_only_probes_their_cycle_ran():
    idea = {"title": "t", "kind": "settings", "hypothesis": "h", "evidence": "e", "novelty": "adapted",
            "predicted_gain": 0.0003, "spec_patch_yaml": "budget:\n  n_trials: 30\n"}

    def plan(cited):
        return {"decision": "continue", "reason": "r", "idea": {**idea, "probe_ids": cited}}

    assert validate_plan(plan(["P01"]), probe_ids=["P01", "P02"]) == []
    assert validate_plan(plan(["P09"]), probe_ids=["P01"])                  # not in this cycle's log
    assert validate_plan(plan(["P01"]), probe_ids=[])                       # nothing was probed
    assert validate_plan(plan("P01"), probe_ids=["P01"])                    # must be a list
    probes = [{"script": "a.py", "question": "q", "finding": "f"}] * 2
    explore = {"decision": "continue", "reason": "r", "idea": {**idea, "probes": probes, "probe_ids": ["P07"]}}
    assert validate_explore(explore, probe_ids=["P01"]) and validate_explore(explore, probe_ids=["P07"]) == []


def test_the_loop_keeps_each_sessions_probes_with_their_outcome(tmp_path):
    from types import SimpleNamespace

    from src.genML.autopilot import probe

    wt = tmp_path / "wt"
    (wt / ws.SCRATCH).mkdir(parents=True)
    log = wt / ws.SCRATCH / probe.LOG_FILE

    def logged(n):                                    # what the harness appends during a session
        for _ in range(n):
            i = len(probe.read_log(log)) + 1
            with open(log, "a") as handle:
                handle.write(json.dumps({"id": f"P{i:02d}", "label": f"idea {i}", "gain": 0.001 * i,
                                         "noise": 0.0005}) + "\n")

    fake = SimpleNamespace(home=tmp_path / "home", save=lambda: None, state={"cycles": []})
    fake._probe_records = lambda cycle_id=None: Autopilot._probe_records(fake, cycle_id)
    cycle = {"id": "C003", "worktree": str(wt), "idea": {"probe_ids": ["P02"]}, "judge": {"gain": 0.0011},
             "status": "rejected"}
    logged(2)
    Autopilot._collect_probes(fake, cycle, "plan")
    logged(1)
    Autopilot._collect_probes(fake, cycle, "revise")
    Autopilot._collect_probes(fake, cycle, "build")                          # nothing new: nothing copied
    kept = probe.read_log(fake.home / probe.LOG_FILE)
    assert [(r["cycle"], r["step"], r["id"]) for r in kept] == [("C003", "plan", "P01"), ("C003", "plan", "P02"),
                                                                ("C003", "revise", "P03")]
    assert [r["id"] for r in Autopilot._cited_probes(fake, cycle)] == ["P02"]
    fake.state["cycles"] = [cycle]
    history = Autopilot._probe_history(fake, {"id": "C004"})
    assert [(h["id"], h["run_measured_gain"], h["cycle_status"]) for h in history if h["cited_by_plan"]] == [
        ("P02", 0.0011, "rejected")]
    assert all(h["run_measured_gain"] is None for h in history if not h["cited_by_plan"])


def test_plans_see_the_bar_the_judge_applies():
    # The dossier's seed-noise threshold is over all rows; on a time split the judge's own rows give a
    # far higher bar (ieee-fraud: 0.0022 against 0.0070), and C001 planned against the lower one.
    from types import SimpleNamespace

    anchors = {"sigma_seed": 0.0015, "sigma_run": None, "z": 3.0,
               "rows": {"scheme": {"kind": "time_forward"}, "n_accept": 100}}
    fake = SimpleNamespace(state={"anchors": anchors, "cycles": []})
    bar = Autopilot._judge_bar(fake)
    assert bar["threshold"] == pytest.approx(3 * np.sqrt(2) * 0.0015) and "lower bound" in bar["source"]
    fake.state["cycles"] = [{"id": "C002", "judge": {"threshold": 0.0070}}, {"id": "C003", "judge": {}}]
    bar = Autopilot._judge_bar(fake)
    assert bar["threshold"] == 0.0070 and "C002" in bar["source"] and bar["rows"] == "time_forward"
    fake._judge_bar = lambda: Autopilot._judge_bar(fake)
    assert Autopilot._judge_bar_text(fake).startswith("0.0070 (")


def test_sessions_are_memory_capped_like_jobs(tmp_path, monkeypatch):
    # Probes run inside the session: a large one must be killed, never push the machine into swap
    from src.genML import jobs
    from src.genML.autopilot import agent, policy

    monkeypatch.setenv(jobs.MEMORY_CAP_ENV, "8G")
    monkeypatch.setitem(jobs._SCOPE_CHECK, "result", (True, "ok"))
    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    seen = {}

    class FakeProcess:
        returncode, pid = 0, 1

        def __init__(self, cmd, **kwargs):
            seen["cmd"] = cmd

        def communicate(self, prompt, timeout=None):
            return None

    monkeypatch.setattr(agent.subprocess, "Popen", FakeProcess)
    header = policy.load_mode(MODES_DIR / "plan.md")[0]
    with pytest.raises(agent.AgentFailed):                                  # the fake leaves no result
        agent.run_agent("p", cwd=tmp_path, log_path=tmp_path / "plan.jsonl", settings=AgentSettings(),
                        system_append="", policy=header, session_id="00000000-aaaa-bbbb-cccc-dddddddddddd")
    cmd = seen["cmd"]
    assert cmd[:3] == ["systemd-run", "--user", "--scope"] and "--collect" in cmd and "MemorySwapMax=0" in cmd
    assert f"MemoryMax={8 * 1024 ** 3}" in cmd
    first = json.loads((tmp_path / "plan.jsonl").read_text().splitlines()[0])
    assert "8.0 GiB, no swap" in first["memory"]


def test_every_mode_file_exists_and_names_its_result_file():
    for mode in ("plan", "explore", "build", "fix", "conclude"):
        text = (MODES_DIR / f"{mode}.md").read_text()
        assert "{{RESULT_FILE}}" in text, mode
        assert "—" not in text, mode
    shared = (MODES_DIR / "_shared.md").read_text()
    assert "Never submit to Kaggle" in shared and "{{PROTECTED}}" in shared


def test_jobs_run_worktree_code_with_shared_folders(tmp_path):
    from src.genML import jobs

    assert jobs._job_code({"id": 1, "options": {}})[1] is None
    code = tmp_path / "wt"
    (code / "src" / "genML").mkdir(parents=True)
    (code / "src" / "genML" / "jobs.py").write_text("")
    cwd, env = jobs._job_code({"id": 2, "options": {"code_dir": str(code)}}, tmp_path / "out")
    assert cwd == code and env["GENML_OUTPUTS"] == str((tmp_path / "out").resolve())
    with pytest.raises(FileNotFoundError):
        jobs._job_code({"id": 3, "options": {"code_dir": str(tmp_path / "missing")}})


# --------------------------------------------------------------------------- #
# A full cycle with a fake Claude CLI
# --------------------------------------------------------------------------- #

FAKE_CLAUDE = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, re, sys
    from pathlib import Path
    argv = sys.argv[1:]
    log = Path(os.environ["FAKE_LOG"])
    with open(log, "a") as handle:
        handle.write(json.dumps(argv) + "\\n")
    session = argv[argv.index("--resume") + 1] if "--resume" in argv else argv[argv.index("--session-id") + 1]
    prompt = sys.stdin.read()
    limit = Path(os.environ["FAKE_LIMIT_FLAG"])
    if not limit.exists():
        limit.write_text("hit")
        print(json.dumps({"type": "result", "subtype": "success", "is_error": True, "session_id": session,
                          "result": "Claude AI usage limit reached|1700000000"}))
        sys.exit(1)
    match = re.search(r"(/\\S+/\\.autopilot/(?:plan|build)\\.json|/\\S+/\\.autopilot/final_report\\.md)", prompt)
    target = Path(match.group(1))
    cwd = Path.cwd()
    if target.name == "plan.json":
        target.write_text(json.dumps({"decision": "continue", "reason": "directive", "journal": "fake plan notes",
            "idea": {"title": "Add a feature", "kind": "code", "hypothesis": "h", "evidence": "e",
                     "novelty": "adapted", "predicted_gain": 0.01, "build_brief": "add src/feature.py"}}))
    elif target.name == "build.json":
        fixing = "Fix the build" in prompt
        (cwd / "src" / "feature.py").write_text("def feature():\\n    return 1\\n")
        (cwd / "tests" / "test_feature.py").write_text("def test_feature():\\n    assert True\\n")
        if not fixing:  # the first attempt also tries to edit the judge
            (cwd / "src" / "genML" / "pipeline" / "diagnostics.py").write_text("JUDGE = 0\\n")
        target.write_text(json.dumps({"summary": "built", "spec_patch_yaml": "budget:\\n  n_trials: 5\\n",
                                      "tests": ["tests/test_feature.py::test_feature"], "leak_test": None}))
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session,
                      "result": "done", "num_turns": 2, "total_cost_usd": 0.1}))
    ''')


class FakeRunner:
    def __init__(self, measured, interrupt_first_wait=False):
        self.measured = measured
        self.launches = []
        self.interrupt = interrupt_first_wait

    def launch(self, problem, options, notes):
        rng = np.random.default_rng(1)
        m = self.measured
        better = _sigmoid(2 * m["signal"] + rng.normal(scale=0.7, size=len(m["y"])))
        run_id = f"20260105_00000{len(self.launches)}_000"
        _save_run(m["problem_dir"], run_id, better, m["signature"], 0.0)
        self.launches.append({"options": options, "notes": notes})
        return {"id": len(self.launches), "run_id": run_id}

    def wait(self, job_id, timeout):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt("simulated kill while the run is going")
        return {"id": job_id, "status": "completed"}


def test_full_cycle_with_fake_agent(tmp_path, measured, monkeypatch):
    repo = _make_repo(tmp_path / "repo")
    fake = tmp_path / "fake_claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_LIMIT_FLAG", str(tmp_path / "limit_hit"))
    from src.genML.autopilot import policy, second_opinion

    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    monkeypatch.setattr(second_opinion, "ask_second_opinion", lambda *a, **k: {
        "verdict": "ok", "summary": "fine", "concerns": [], "suggestion": "", "backend": "fake-codex"})
    waits = []
    config = AutopilotConfig(problem=PROBLEM, cycles=1, directive="add a feature", suspicious_ratio=1e6,
                             workspace_dir=str(tmp_path / "ws"), verify_command=[sys.executable, "-c", "pass"],
                             agent=AgentSettings(claude_bin=str(fake)), explore_every=0, canary=False)
    runner = FakeRunner(measured, interrupt_first_wait=True)
    pilot = Autopilot(config, repo=repo, outputs_root=measured["outputs"], problems_dir=measured["problems"],
                      runner=runner, sleep=waits.append, data_dir=tmp_path)

    with pytest.raises(KeyboardInterrupt):                       # killed while the run is going
        pilot.run()
    state = load_state(PROBLEM, measured["outputs"])
    assert state["cycles"][0]["step"] == "run" and state["cycles"][0]["job_id"] == 1

    resumed = Autopilot(config, repo=repo, outputs_root=measured["outputs"], problems_dir=measured["problems"],
                        runner=runner, sleep=waits.append, data_dir=tmp_path)
    state = resumed.run()
    cycle = state["cycles"][0]
    assert len(runner.launches) == 1                             # waited for the queued run, no relaunch
    assert cycle["status"] == "accepted", cycle.get("error")
    assert cycle["fix_attempts"] == 1                            # the judge edit was reverted and fixed
    assert any("protected files" in w for w in cycle["warnings"])
    assert len(waits) == 1                                       # one usage-limit wait
    assert [o["verdict"] for o in cycle["second_opinions"]] == ["ok"] and not cycle.get("revised")
    assert runner.launches[0]["options"]["code_dir"] == cycle["worktree"]
    assert runner.launches[0]["options"]["budget"]["n_trials"] == 5

    main = state["main_branch"]
    assert ws.branch_head(repo, main) == cycle["commit"]
    assert _git(repo, "show", f"{main}:src/genML/pipeline/diagnostics.py") == "JUDGE = 1"
    assert "def feature" in _git(repo, "show", f"{main}:src/feature.py")
    assert not Path(cycle["worktree"]).exists()
    assert state["best"]["experiment_id"] == cycle["judge"]["candidate_experiment_id"]

    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    for call in calls:
        settings = json.loads(Path(call[call.index("--settings") + 1]).read_text())
        assert "Edit(src/genML/pipeline/diagnostics.py)" in settings["permissions"]["deny"]
        assert settings["sandbox"]["failIfUnavailable"] and settings["sandbox"]["network"]["deniedDomains"] == ["*"]
    assert any("--resume" in call for call in calls)             # the limit hit was resumed, not restarted

    ledger = json.loads((measured["problem_dir"] / "experiments.json").read_text())
    assert ledger[-1]["source"]["autopilot"] == "C001" and ledger[-1]["status"] == "accepted"
    journal = (measured["outputs"] / PROBLEM / "autopilot" / "journal.md").read_text()
    assert "C001 [accepted] Add a feature" in journal and "fake plan notes" in journal
    assert "accepted" in format_status(state)
    assert os.path.exists(measured["outputs"] / PROBLEM / "autopilot" / "cycles" / "C001" / "judge.json")


def _reviewed_pilot(tmp_path, measured, monkeypatch, answers, suspicious_gain=None):
    """A pilot with the fake Claude CLI and a scripted second opinion (no usage-limit hit)."""
    from src.genML.autopilot import policy, second_opinion

    repo = _make_repo(tmp_path / "repo")
    fake = tmp_path / "fake_claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    (tmp_path / "limit_hit").write_text("hit")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_LIMIT_FLAG", str(tmp_path / "limit_hit"))
    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    asked = []

    def fake_opinion(problem, question, material, **kwargs):
        asked.append(question)
        verdict = answers.pop(0) if answers else "ok"
        if isinstance(verdict, Exception):
            raise verdict
        concerns = [] if verdict == "ok" else [{"kind": "weak_evidence", "detail": "no probe supports it"}]
        return {"verdict": verdict, "summary": f"{verdict} summary", "concerns": concerns, "suggestion": "",
                "backend": "fake-codex", "model": "m"}

    monkeypatch.setattr(second_opinion, "ask_second_opinion", fake_opinion)
    config = AutopilotConfig(problem=PROBLEM, cycles=1, directive="add a feature", suspicious_ratio=1e6,
                             suspicious_gain=suspicious_gain, workspace_dir=str(tmp_path / "ws"),
                             verify_command=[sys.executable, "-c", "pass"],
                             agent=AgentSettings(claude_bin=str(fake)), explore_every=0, canary=False)
    runner = FakeRunner(measured)
    pilot = Autopilot(config, repo=repo, outputs_root=measured["outputs"], problems_dir=measured["problems"],
                      runner=runner, sleep=lambda seconds: None, data_dir=tmp_path)
    return pilot, runner, asked


def test_concerns_get_one_revision_then_the_cycle_goes_on(tmp_path, measured, monkeypatch):
    from src.genML.autopilot import second_opinion

    pilot, runner, asked = _reviewed_pilot(tmp_path, measured, monkeypatch, ["concerns"])
    cycle = pilot.run()["cycles"][0]
    assert asked == [second_opinion.PLAN_QUESTION]                     # no second review after "concerns"
    assert cycle["revised"] and cycle["status"] == "accepted" and len(runner.launches) == 1
    folder = measured["outputs"] / PROBLEM / "autopilot" / "cycles" / "C001"
    assert (folder / "plan_revised.json").exists() and (folder / "second_opinion_plan.json").exists()
    assert "OPINION_JSON" not in (folder / "revise_prompt.md").read_text()
    assert "no probe supports it" in (folder / "revise_prompt.md").read_text()


def test_a_revised_plan_rejected_again_fails_the_cycle(tmp_path, measured, monkeypatch):
    pilot, runner, asked = _reviewed_pilot(tmp_path, measured, monkeypatch, ["reject", "reject"])
    cycle = pilot.run()["cycles"][0]
    assert len(asked) == 2 and cycle["status"] == "failed"
    assert "rejected the revised plan" in cycle["error"] and runner.launches == []


def test_no_reviewer_does_not_stop_the_loop(tmp_path, measured, monkeypatch):
    from src.genML.llm import LLMError

    pilot, runner, _ = _reviewed_pilot(tmp_path, measured, monkeypatch, [LLMError("no backend answered")])
    cycle = pilot.run()["cycles"][0]
    assert cycle["second_opinions"][0]["verdict"] == "unavailable" and not cycle.get("revised")
    assert cycle["status"] == "accepted" and len(runner.launches) == 1


def test_suspicious_gains_get_a_triage_for_the_person(tmp_path, measured, monkeypatch):
    from src.genML.autopilot import second_opinion

    pilot, _, asked = _reviewed_pilot(tmp_path, measured, monkeypatch, ["ok", "reject"], suspicious_gain=0.0)
    state = pilot.run()
    cycle = state["cycles"][0]
    assert cycle["status"] == "suspicious" and state["status"] == "paused"
    assert asked == [second_opinion.PLAN_QUESTION, second_opinion.GAIN_QUESTION]
    assert "Second opinion (fake-codex): reject" in state["stop_reason"]
    assert (measured["outputs"] / PROBLEM / "autopilot" / "cycles" / "C001" / "second_opinion_gain.json").exists()


def test_settings_idea_runs_clean_start_code(tmp_path, measured, monkeypatch):
    from src.genML.autopilot import policy

    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    repo = _make_repo(tmp_path / "repo")
    runner = FakeRunner(measured)
    config = AutopilotConfig(problem=PROBLEM, workspace_dir=str(tmp_path / "ws"), explore_every=0)
    pilot = Autopilot(config, repo=repo, outputs_root=measured["outputs"], problems_dir=measured["problems"],
                      runner=runner, data_dir=tmp_path)
    pilot.load_or_init()
    cycle = pilot.new_cycle()
    pilot._prepare_worktree(cycle)
    wt = Path(cycle["worktree"])
    (wt / "src" / "genML" / "pipeline" / "diagnostics.py").write_text("JUDGE = 0\n")   # a plan session's edit
    cycle.update(step="run", idea={"title": "More trials", "kind": "settings", "hypothesis": "h", "evidence": "e",
                                   "novelty": "general", "predicted_gain": 0.001,
                                   "spec_patch_yaml": "budget:\n  n_trials: 7\n"})
    pilot.run_cycle(cycle)
    assert len(runner.launches) == 1 and runner.launches[0]["options"]["budget"]["n_trials"] == 7
    assert any("discarded unverified worktree changes" in w for w in cycle["warnings"])
    assert cycle["status"] in ("accepted", "rejected", "suspicious"), cycle.get("error")


def test_loop_refuses_to_start_without_a_sandbox(tmp_path, measured, monkeypatch):
    from src.genML.autopilot import policy

    monkeypatch.setattr(policy, "sandbox_available", lambda: (False, "bwrap not installed"))
    config = AutopilotConfig(problem=PROBLEM, workspace_dir=str(tmp_path / "ws"))
    pilot = Autopilot(config, repo=_make_repo(tmp_path / "repo"), outputs_root=measured["outputs"],
                      problems_dir=measured["problems"], runner=FakeRunner(measured), data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="no sandbox"):
        pilot.run()
    assert load_state(PROBLEM, measured["outputs"]) is None               # nothing was started
    config.allow_unsandboxed = True
    assert pilot.preflight() == []


def test_directive_inbox_feeds_one_cycle_at_a_time(tmp_path, measured):
    from src.genML.autopilot.loop import queue_directive

    first = queue_directive(PROBLEM, "build RealMLP", measured["outputs"])
    queue_directive(PROBLEM, "build TabM", measured["outputs"])
    assert first.exists() and first.read_text() == "build RealMLP\n"
    pilot = Autopilot(AutopilotConfig(problem=PROBLEM, explore_every=1, workspace_dir=str(tmp_path / "ws")),
                      repo=tmp_path, outputs_root=measured["outputs"], problems_dir=measured["problems"])
    pilot.state = {"cycles": [], "pending_directive": None, "problem": PROBLEM}
    pilot.save = lambda: None
    cycle = pilot.new_cycle()
    assert cycle["directive"] == "build RealMLP" and cycle["mode"] == "plan"   # a directive overrides explore
    assert pilot.new_cycle()["directive"] == "build TabM"
    third = pilot.new_cycle()
    assert third["directive"] is None and third["mode"] == "explore"
    assert len(list((measured["outputs"] / PROBLEM / "autopilot" / "inbox" / "done").glob("*.md"))) == 2


def test_sessions_find_the_gate_copy_already_made(tmp_path, monkeypatch):
    # A session's sandbox cannot write outside its worktree, and a worktree's gate script has CRLF
    # line endings (git autocrlf), so its copy differs from the owner's: the loop makes it first.
    from datetime import date
    from types import SimpleNamespace

    from src.genML.autopilot import research

    wt = tmp_path / "worktree"
    script = wt / "src" / "genML" / "autopilot" / research.GATE_SCRIPT.name
    script.parent.mkdir(parents=True)
    script.write_bytes(research.GATE_SCRIPT.read_bytes().replace(b"\n", b"\r\n"))
    root = tmp_path / "gate"
    monkeypatch.setattr(research, "GATE_COPY_ROOT", root)
    logged = []
    fake = SimpleNamespace(log=logged.append)
    Autopilot._prepare_gate_copy(fake, wt)
    folders = list(root.iterdir())
    assert len(folders) == 1 and (folders[0] / script.name).read_bytes() == script.read_bytes()
    # inside the session nothing outside the worktree is writable: the command must still build
    monkeypatch.setattr(research, "GATE_SCRIPT", script)
    for folder in (root, folders[0]):
        folder.chmod(0o555)
    try:
        config = research.ResearchConfig(domain="card payment fraud detection " * 3, sources_before=date(2019, 7, 15))
        assert str(folders[0]) in research.gate_command(config)
    finally:
        for folder in (root, folders[0]):
            folder.chmod(0o755)
    Autopilot._prepare_gate_copy(fake, tmp_path / "no-gate-here")          # a toy checkout: nothing to do
    assert logged == []


def test_a_failing_suite_is_reported_failures_first(tmp_path):
    # pytest prints its warnings after the tracebacks; the fix prompt and the log lead with what failed
    from types import SimpleNamespace

    noisy = ("print('FAILED tests/test_a.py::test_b - AssertionError'); print('warning ' * 2000); "
             "print('1 failed, 3 passed in 0.1s'); raise SystemExit(1)")
    fake = SimpleNamespace(config=SimpleNamespace(verify_command=[sys.executable, "-c", noisy], verify_timeout=60),
                           cycle_dir=lambda cycle: tmp_path)
    fake._verify_command = lambda wt: Autopilot._verify_command(fake, wt)
    cycle = {"fix_attempts": 0}
    report = Autopilot._run_tests(fake, cycle, tmp_path)
    assert report.splitlines()[:2] == ["FAILED tests/test_a.py::test_b - AssertionError", "1 failed, 3 passed in 0.1s"]
    assert cycle["verify"]["returncode"] == 1 and (tmp_path / "verify_0.log").exists()
    fake.config.verify_command = [sys.executable, "-c", "pass"]
    assert Autopilot._run_tests(fake, cycle, tmp_path) is None
