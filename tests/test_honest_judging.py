"""Honest judging: leak rules, the shuffled-label canary, run-to-run noise and the untouched holdout.

Protected: the autopilot's agents cannot change these tests.
"""

import json
import sys

import numpy as np
import pytest

from src.genML import experiments, jobs
from src.genML.autopilot import judge as judging
from src.genML.autopilot import policy
from src.genML.autopilot.agent import AgentSettings
from src.genML.autopilot.loop import Autopilot, AutopilotConfig
from src.genML.pipeline.oof import ExperimentStore, fold_signature, make_folds
from tests.test_autopilot import (CV, FAKE_CLAUDE, PROBLEM, FakeRunner, _make_repo, _save_run,  # noqa: F401
                                  _sigmoid, measured)


# --------------------------------------------------------------------------- #
# Leak rules
# --------------------------------------------------------------------------- #

def test_settings_that_read_the_target_are_refused_even_through_a_chain():
    from src.genML.features.derived import check_settings, label_references

    chain = {"derived": {"t2": "y * 2", "ok": "a / b"}, "keys": {"k": ["t2", "shop"]},
             "group_stats": [{"by": "k", "cols": ["ok"]}, {"by": "shop", "cols": ["a"]}]}
    assert label_references(chain, "y") == ["derived.t2", "keys.k", "group_stats[0]"]
    assert any("reads the target" in e for e in check_settings(chain, "y"))
    assert check_settings({"group_stats": [{"by": "shop", "cols": ["a"]}]}, "y") == []


@pytest.mark.parametrize("patch, fragment", [
    ({"feature_engineering": {"encodings": {"nested": False}}}, "nested: false"),
    ({"feature_engineering": {"base": "auto"}}, "base: auto"),
    ({"feature_engineering": {"group_stats": [{"by": "shop", "cols": ["y"]}]}}, "reads the target"),
])
def test_patches_that_open_a_leak_are_refused(patch, fragment):
    errors = experiments.check_patch(patch, target="y")
    assert any(fragment in e for e in errors), errors
    assert any(fragment in e for e in experiments.check_patch(patch, target="y", known_only=False))


def test_code_ideas_may_name_new_settings_but_not_leaks():
    new = {"feature_engineering": {"my_new_encoding": True}, "models": ["MyNewModel"]}
    assert experiments.check_patch(new, target="y")                         # unknown to this checkout
    assert experiments.check_patch(new, target="y", known_only=False) == []
    assert experiments.leak_problems({"feature_engineering": {}}, "y")      # base defaults to auto
    assert experiments.leak_problems({"feature_engineering": {"base": "raw"}}, "y") == []


def test_experiments_cannot_set_the_seed():
    assert experiments.check_patch({"budget": {"seed": 3}})                 # only --baseline sets it
    from src.genML.pipeline.training import run_seed

    assert run_seed({}) == 42 and run_seed({"seed": 2}) == 44


# --------------------------------------------------------------------------- #
# Rows that decide, and anchoring
# --------------------------------------------------------------------------- #

def test_time_blocks_are_judged_forward_on_the_latest_block():
    fold_ids = np.repeat(np.arange(6), 100)
    y = np.random.default_rng(0).integers(0, 2, 600)
    codes, scheme = judging.choose_rows(fold_ids, y, "binary", "time_blocks")
    assert scheme["kind"] == "time_forward" and scheme["block"] == 5
    assert set(np.unique(codes[fold_ids < 5])) == {0}                       # earlier blocks never score
    assert (codes == judging.ACCEPT).sum() == 50 and (codes == judging.HOLDOUT).sum() == 50
    keep, chunks = judging._acceptance(codes, fold_ids, scheme)
    assert len(keep) == 50 and set(chunks) == set(range(judging.TIME_CHUNKS))


def test_random_holdout_is_stratified():
    y = np.array([0] * 900 + [1] * 100)
    codes, scheme = judging.choose_rows(np.arange(1000) % 5, y, "binary", "stratified_kfold")
    held = codes == judging.HOLDOUT
    assert scheme["kind"] == "random_holdout" and held.sum() == 200 and y[held].sum() == 20


def test_anchoring_needs_a_finished_noise_measurement(measured):
    (measured["problem_dir"] / "noise_runs.json").unlink()
    with pytest.raises(ValueError, match="--baseline"):
        judging.make_anchors(PROBLEM, measured["outputs"])
    (measured["problem_dir"] / "noise_runs.json").write_text(json.dumps(
        [{"runs": [{"run_id": "20260101_000000_000"}, {"run_id": "20990101_000000_000"}]}]))
    with pytest.raises(ValueError, match="still running"):
        judging.make_anchors(PROBLEM, measured["outputs"])


def test_anchoring_refuses_a_leaky_baseline(measured):
    for run in measured["problem_dir"].glob("2026*/reports/resolved_spec.json"):
        spec = json.loads(run.read_text())
        spec["feature_engineering"] = {}                                    # base: auto
        run.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="leak"):
        judging.make_anchors(PROBLEM, measured["outputs"])


def test_noise_runs_are_queued_and_measured(tmp_path):
    from src.genML.pipeline.diagnostics import queue_baseline_runs, run_noise

    outputs = tmp_path / "out"
    record = queue_baseline_runs("toy", repeats=3, outputs_root=outputs, start_worker=False)
    queued = jobs.list_jobs(outputs_root=outputs)
    assert sorted(j["options"]["budget"]["seed"] for j in queued) == [0, 1, 2]
    noise = run_noise("toy", outputs)
    assert not noise["complete"] and len(noise["missing"]) == 3
    for k, run in enumerate(record["runs"]):
        reports = outputs / "toy" / run["run_id"] / "reports"
        reports.mkdir(parents=True)
        (reports / "final_predictions.json").write_text(json.dumps({"final_score": 0.9 + k * 0.001}))
    noise = run_noise("toy", outputs)
    assert noise["complete"] and noise["sigma_final"] == pytest.approx(0.001)


# --------------------------------------------------------------------------- #
# Shuffled-label canary
# --------------------------------------------------------------------------- #

def _canary_run(root, y_shuffled, oof, run_id="20260201_000000_000"):
    problem_dir = root / PROBLEM
    signature = fold_signature(y_shuffled, CV)
    problem_dir.mkdir(parents=True, exist_ok=True)
    np.save(problem_dir / f"folds_{signature}.npy", make_folds(y_shuffled, CV))
    ExperimentStore(problem_dir).save_targets(signature, y_shuffled, {})
    _save_run(problem_dir, run_id, oof, signature, 0.0)
    return run_id


def test_canary_verdict_tells_leaks_from_chance(measured, tmp_path):
    anchors = judging.make_anchors(PROBLEM, measured["outputs"])
    rng = np.random.default_rng(5)
    y = rng.permutation(measured["y"])
    clean = _canary_run(tmp_path / "clean", y, rng.random(len(y)))
    assert judging.canary_verdict(anchors, tmp_path / "clean", PROBLEM, clean)["verdict"] == "clean"
    leaky = _canary_run(tmp_path / "leaky", y, np.clip(0.6 * y + 0.4 * rng.random(len(y)), 0, 1))
    result = judging.canary_verdict(anchors, tmp_path / "leaky", PROBLEM, leaky)
    assert result["verdict"] == "leak" and result["gain_over_constant"] > result["bar"]


def test_canary_data_shuffles_only_the_labels(tmp_path, monkeypatch):
    from src.genML.autopilot.canary import prepare_canary_data

    data = tmp_path / "data"
    data.mkdir()
    rows = ["id,amount,city,y"] + [f"{i},{i * 1.5:.2f},c{i % 3},{i % 2}" for i in range(50)]
    (data / "train.csv").write_text("\n".join(rows) + "\n")
    (data / "test.csv").write_text("id,amount,city\n100,1.00,c1\n")
    (data / "sample_submission.csv").write_text("id,y\n100,0.5\n")
    problems = tmp_path / "problems"
    (problems / "toy").mkdir(parents=True)
    (problems / "toy" / "problem.yaml").write_text(
        f"name: toy\nsource: {{type: csv, path: {data}}}\ntarget: y\nid_column: id\ntask: binary\nmetric: roc_auc\n")
    monkeypatch.setattr("src.genML.pipeline.problem.PROBLEMS_DIR", problems)
    out = prepare_canary_data("toy", tmp_path / "canary", seed=1)
    before = [line.split(",") for line in (data / "train.csv").read_text().splitlines()]
    after = [line.split(",") for line in (out / "train.csv").read_text().splitlines()]
    assert [r[:3] for r in before] == [r[:3] for r in after]                # every other column untouched
    assert sorted(r[3] for r in before[1:]) == sorted(r[3] for r in after[1:])
    assert [r[3] for r in before[1:]] != [r[3] for r in after[1:]]          # labels moved
    assert (out / "test.csv").is_symlink() and (out / "sample_submission.csv").is_symlink()


class CanaryRunner(FakeRunner):
    """Writes the canary's run into its own outputs root, with chance-level or leaky predictions."""

    def __init__(self, measured, leak):
        super().__init__(measured)
        self.leak = leak

    def launch(self, problem, options, notes):
        if "outputs_root" not in options:
            return super().launch(problem, options, notes)
        rng = np.random.default_rng(9)
        y = rng.permutation(self.measured["y"])
        oof = np.clip(0.6 * y + 0.4 * rng.random(len(y)), 0, 1) if self.leak else rng.random(len(y))
        from pathlib import Path

        run_id = _canary_run(Path(options["outputs_root"]), y, oof, run_id="20260301_000000_000")
        self.launches.append({"options": options, "notes": notes})
        return {"id": len(self.launches), "run_id": run_id}


@pytest.mark.parametrize("leak, status", [(True, "rejected"), (False, "accepted")])
def test_code_cycles_must_pass_the_canary(tmp_path, measured, monkeypatch, leak, status):
    from src.genML.autopilot import canary, second_opinion

    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    monkeypatch.setattr(canary, "prepare_canary_data", lambda problem, dest, seed: dest)
    monkeypatch.setattr(second_opinion, "ask_second_opinion", lambda *a, **k: {
        "verdict": "ok", "summary": "", "concerns": [], "suggestion": "", "backend": "fake"})
    fake = tmp_path / "fake_claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    (tmp_path / "limit_hit").write_text("hit")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_LIMIT_FLAG", str(tmp_path / "limit_hit"))
    runner = CanaryRunner(measured, leak)
    config = AutopilotConfig(problem=PROBLEM, cycles=1, directive="add a feature", suspicious_ratio=1e6,
                             workspace_dir=str(tmp_path / "ws"), verify_command=[sys.executable, "-c", "pass"],
                             agent=AgentSettings(claude_bin=str(fake)), explore_every=0)
    pilot = Autopilot(config, repo=_make_repo(tmp_path / "repo"), outputs_root=measured["outputs"],
                      problems_dir=measured["problems"], runner=runner, sleep=lambda s: None, data_dir=tmp_path)
    state = pilot.run()
    cycle = state["cycles"][0]
    assert cycle["status"] == status and len(runner.launches) == 2
    canary_options = runner.launches[1]["options"]
    assert canary_options["source"]["type"] == "csv" and "canary" in canary_options["outputs_root"]
    assert cycle["judge"]["canary"] == ("leak" if leak else "clean")
    if leak:
        assert any("shuffled labels" in w for w in cycle["warnings"])
        assert state["best"]["run_id"] == state["anchors"]["best"]["run_id"]   # the leak did not become best
    assert "holdout" not in cycle["judge"] and cycle["id"] in state["holdout_log"]


# --------------------------------------------------------------------------- #
# Jobs route canary runs to their own outputs and data
# --------------------------------------------------------------------------- #

def test_canary_jobs_write_to_their_own_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    code = tmp_path / "wt"
    (code / "src" / "genML").mkdir(parents=True)
    (code / "src" / "genML" / "jobs.py").write_text("# execute-json\n")
    outputs, canary_root = tmp_path / "outputs", tmp_path / "canary_out"
    job = jobs.enqueue("toy", "run", {"code_dir": str(code), "outputs_root": str(canary_root),
                                      "source": {"type": "csv", "path": "/data/shuffled"}}, "", outputs_root=outputs)
    command, _, env, tmp = jobs._launch(jobs.get_job(job["id"], outputs), outputs)
    joined = " ".join(command)
    assert env["GENML_OUTPUTS"] == str(canary_root.resolve())
    assert f"--bind {canary_root.resolve() / 'toy'}" in joined and f"--outputs-root {canary_root.resolve()}" in joined
    assert str(tmp).startswith(str(outputs.resolve()))                     # job temp files stay with the queue


# --------------------------------------------------------------------------- #
# Prompts carry this problem's facts, not another problem's
# --------------------------------------------------------------------------- #

def test_every_prompt_is_filled_for_this_problem(tmp_path, measured):
    from src.genML.autopilot.loop import LOOP_MODES

    pilot = Autopilot(AutopilotConfig(problem=PROBLEM, workspace_dir=str(tmp_path / "ws")), repo=tmp_path,
                      outputs_root=measured["outputs"], problems_dir=measured["problems"], data_dir=tmp_path)
    pilot.state = {"anchors": judging.make_anchors(PROBLEM, measured["outputs"]), "cycles": [],
                   "best": {"run_id": "20260101_000000_000", "experiment_id": "x", "score": 0.8}}
    cycle = {"id": "C001", "worktree": str(tmp_path / "wt")}
    extra = {k: "x" for k in ("RESULT_FILE", "IDEA_JSON", "PROBLEMS", "ATTEMPT", "OPINION_JSON", "REVIEWER")}
    for mode in ("_shared",) + LOOP_MODES:
        text = pilot._render(mode, cycle, extra)
        assert "{{" not in text, (mode, text[text.find("{{"):text.find("{{") + 40])
        assert "700k" not in text and "11 GB" not in text and "train.csv order" not in text, mode
    facts = pilot._facts()
    assert facts["TRAIN_ROWS"] == f"{len(measured['y']):,}" and facts["N_FOLDS"] == "5"
    assert facts["FOLDS_NEEDED"] == "4" and "80%" in facts["JUDGED_ON"]


# --------------------------------------------------------------------------- #
# One baseline run plus a seed test (the default)
# --------------------------------------------------------------------------- #

from tests.test_jobs_mcp import toy_env  # noqa: E402,F401  (fixture)


def test_one_run_baseline_with_a_seed_test_through_the_queue(toy_env):
    from src.genML.pipeline.diagnostics import NOISE_OOFS, queue_baseline_runs, run_noise

    record = queue_baseline_runs("toy", start_worker=False)
    assert record["repeats"] == 1 and record["seed_test"]["run_id"] == record["runs"][0]["run_id"]
    assert not run_noise("toy")["complete"]
    assert jobs.run_one()["status"] == "completed"                         # the baseline run
    assert not run_noise("toy")["complete"]                                 # the seed test is still queued
    noise_job = jobs.run_one()
    assert noise_job["kind"] == "noise" and noise_job["status"] == "completed", noise_job.get("error")
    noise = run_noise("toy")
    assert noise["complete"] and noise["sigma_seed"] is not None and noise["sigma_final"] is None
    run_dir = toy_env / "toy" / record["runs"][0]["run_id"] / "reports"
    assert np.load(run_dir / NOISE_OOFS).shape[0] == 3                      # the run's seed plus two more
    from src.genML.run_context import list_runs

    assert [r["run_id"] for r in list_runs("toy")] == [record["runs"][0]["run_id"]]   # the seed test is no run
    assert experiments.current_best("toy")["run_id"] == record["runs"][0]["run_id"]


def test_the_judge_rescores_the_seed_test_on_its_own_rows(measured):
    from src.genML.pipeline.diagnostics import NOISE_OOFS, NOISE_REPORT

    problem_dir, signal, rng = measured["problem_dir"], measured["signal"], measured["rng"]
    run_id = "20260101_000000_000"
    seeds = [_sigmoid(2 * signal + rng.normal(scale=1.5, size=len(signal))) for _ in range(3)]
    np.save(problem_dir / run_id / "reports" / NOISE_OOFS, np.stack([np.column_stack([1 - s, s]) for s in seeds]))
    (problem_dir / run_id / "reports" / NOISE_REPORT).write_text(json.dumps({"sigma_seed": 123.0}))
    (problem_dir / "noise_runs.json").write_text(json.dumps([{"repeats": 1, "runs": [{"run_id": run_id}],
                                                              "seed_test": {"run_id": run_id}}]))
    anchors = judging.make_anchors(PROBLEM, measured["outputs"])
    assert anchors["sigma_run"] is None
    assert 0 < anchors["sigma_seed"] < 1                                    # rescored, not the report's 123
    assert judging.noise_sigma(anchors) == anchors["sigma_seed"]
