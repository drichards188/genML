"""The referee (src/genML/referee): contests, sandboxed contestants, graded verdicts and leak checks."""

import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from src.genML.referee import arena, checks, contest, verdict

CONTESTANT = '''
import argparse, json, socket
import numpy as np, pandas as pd
p = argparse.ArgumentParser()
for name in ("--contest", "--out", "--workdir", "--time-limit", "--column"):
    p.add_argument(name)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--noise", type=float, default=0.05)
p.add_argument("--probe", default=None)
a = p.parse_args()
hidden = pd.read_parquet(a.contest + "/hidden.parquet")
rng = np.random.default_rng(a.seed)
np.save(a.out, hidden[a.column].to_numpy(float) + rng.normal(scale=a.noise, size=len(hidden)))
json.dump({"validation_score": 0.99}, open(a.workdir + "/self_report.json", "w"))
if a.probe:
    seen = {}
    try:
        open(a.probe).read(); seen["labels"] = "read"
    except OSError as exc:
        seen["labels"] = type(exc).__name__
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3); seen["network"] = "reached"
    except OSError as exc:
        seen["network"] = type(exc).__name__
    json.dump(seen, open(a.workdir + "/probe.json", "w"))
'''


def _frame(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    s, h = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * s + 1.5 * h)))).astype(int)
    return pd.DataFrame({"id": np.arange(n), "t": rng.permutation(n).astype(float), "weak": s + 0.5 * h,
                         "strong": s + h, "y": y})


@pytest.fixture
def made(tmp_path):
    folder = contest.create_contest(_frame(), "toy", target="y", metric="roc_auc", id_column="id",
                                    time_column="t", outputs_root=tmp_path)
    script = tmp_path / "contestant.py"
    script.write_text(CONTESTANT)
    return {"root": tmp_path, "folder": folder, "command": [sys.executable, str(script)]}


def _enter(made, entry, column, seeds=(0, 1), noise=0.05, sandbox=False, extra=()):
    for seed in seeds:
        result = arena.run_entry("toy", entry, [*made["command"], "--column", column, "--noise", str(noise), *extra],
                                 seed=seed, sandbox=sandbox, outputs_root=made["root"])
        assert result["status"] == "ok", result
    return result


def test_a_time_contest_hides_the_latest_block_and_keeps_its_labels_private(made):
    public = made["folder"] / "public"
    train, hidden = pd.read_parquet(public / "train.parquet"), pd.read_parquet(public / "hidden.parquet")
    labels, codes = contest.labels_and_rows("toy", made["root"])
    assert "y" not in hidden.columns and "y" in train.columns
    assert hidden["t"].min() > train["t"].max()                       # contestants predict forward in time
    assert len(hidden) == len(labels) == len(codes) and abs(len(hidden) - 20000 / 6) < 2
    assert set(np.unique(codes)) == {contest.DECIDE, contest.HOLDOUT}
    assert abs((codes == contest.DECIDE).mean() - 0.5) < 0.01
    facts = json.loads((public / "contest.json").read_text())
    assert facts["classes"] == [0, 1] and "hashes" not in facts and not list(public.glob("*.npy"))
    record = contest.load_contest("toy", made["root"])
    assert set(record["hashes"]) == {"train.parquet", "hidden.parquet", "contest.json"}


def test_a_random_contest_hides_a_stratified_share(tmp_path):
    frame = _frame(5000)
    contest.create_contest(frame, "rand", target="y", metric="roc_auc", outputs_root=tmp_path)
    labels, _ = contest.labels_and_rows("rand", tmp_path)
    assert len(labels) == 1000 and abs(labels.mean() - frame["y"].mean()) < 0.01


def test_a_changed_public_file_or_prediction_is_not_counted(made):
    _enter(made, "base", "weak", seeds=(0,))
    entry_dir = next((made["folder"] / "entries").iterdir())
    np.save(entry_dir / "predictions.npy", np.zeros(len(contest.labels_and_rows("toy", made["root"])[0])))
    assert arena.entries("toy", made["root"]) == {}                    # predictions edited after the run
    (made["folder"] / "public" / "contest.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        contest.load_contest("toy", made["root"])


def test_graded_verdicts(made):
    _enter(made, "weak", "weak")
    _enter(made, "strong", "strong")
    _enter(made, "weak-again", "weak", seeds=(5, 6))
    real = verdict.compare("toy", "weak", "strong", outputs_root=made["root"])
    assert real["grade"] == "proven" and real["gain"] > real["smallest_provable_gain"]
    assert real["holdout_gain"] > 0 and not real["flags"]
    assert verdict.compare("toy", "weak", "weak-again", outputs_root=made["root"])["grade"] == "no_evidence"
    assert verdict.compare("toy", "strong", "weak", outputs_root=made["root"])["grade"] == "worse"
    summary = verdict.entry_summary("toy", made["root"])
    assert summary["strong"]["seeds"] == [0, 1] and summary["strong"]["seed_sd"] is not None
    assert summary["weak"]["optimism"] > 0.05                         # it claimed 0.99 about itself


def test_the_seed_study_runs_delong_and_the_referee_on_groups_of_runs(made):
    from src.genML.referee import seed_study

    _enter(made, "same", "weak", seeds=tuple(range(6)))
    result = seed_study.study("toy", "same", size=3, n_groups=5, n_boot=50, outputs_root=made["root"])
    assert result["single_runs"]["pairs"] == 15 and 0 <= result["single_runs"]["delong_p05_rate"] <= 1
    assert result["groups"]["comparisons"] == 5 and len(result["per_run"]) == 6
    assert sum(result["groups"]["referee_grades"].values()) == 5
    y = np.random.default_rng(0).random(4000) < 0.1
    signal = y + np.random.default_rng(1).normal(0, 1, 4000)
    same = seed_study.delong(y.astype(int), signal, signal)
    assert same["z"] == 0 and same["p"] == 1
    better = seed_study.delong(y.astype(int), signal, np.random.default_rng(2).random(4000))
    assert better["auc_a"] > 0.7 and better["z"] > 5


def test_the_power_study_judges_each_comparison_five_ways(made):
    from src.genML.referee import power_study

    _enter(made, "weak", "weak", seeds=tuple(range(6)))
    _enter(made, "weak-b", "weak", seeds=(20, 21, 22))
    _enter(made, "strong", "strong", seeds=(10, 11, 12))
    design = {"same": {"weak": ["weak"]}, "equal": ["weak", "weak-b"], "planted": {"weak": ["strong"]}}
    result = power_study.study("toy", design, n=3, n_boot=50, outputs_root=made["root"])
    summary = result["summary"]
    assert summary["same_settings"]["referee"]["n"] == 3 and summary["planted"]["referee"]["n"] == 3
    assert len(result["detail"]["different_but_equal"]) == 1
    for test in power_study.TESTS:                      # a clear gain: every test finds it, none the wrong way
        assert summary["planted"][test]["right"] == 1 and summary["planted"][test]["wrong_way"] == 0
        assert summary["planted"][test]["holdout_agrees"] == 1
    assert result["conclusion"] and result["formula_vs_referee_max_gap"] is not None


def test_without_seed_noise_a_gain_can_only_be_likely(made):
    _enter(made, "weak", "weak", seeds=(0,))
    _enter(made, "strong", "strong", seeds=(0,))
    result = verdict.compare("toy", "weak", "strong", outputs_root=made["root"])
    assert result["grade"] == "likely" and "seed noise not measured" in result["flags"][0]
    measured = verdict.compare("toy", "weak", "strong", outputs_root=made["root"], prior_sd=0.001)
    assert measured["grade"] == "proven"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
def test_a_sandboxed_contestant_sees_no_labels_and_no_network(made):
    labels_path = made["folder"] / "private" / "labels.npy"
    _enter(made, "probe", "weak", seeds=(0,), sandbox=True, extra=("--probe", str(labels_path)))
    entry_dir = made["folder"] / "entries" / "probe__s0"
    seen = json.loads((entry_dir / "work" / "probe.json").read_text())
    assert seen["labels"] == "FileNotFoundError" and seen["network"] != "reached"
    assert arena.entries("toy", made["root"])["probe"][0]["sandboxed"] is True


def test_leak_checks_flag_duplicates_a_leaky_column_and_an_id(tmp_path):
    rng = np.random.default_rng(3)
    frame = _frame(6000, seed=3)
    frame["chargeback"] = frame["y"] + rng.normal(scale=0.05, size=len(frame))   # recorded after the outcome
    frame = pd.concat([frame, frame.sample(600, random_state=1)], ignore_index=True)  # duplicated rows
    frame["id"] = np.where(frame["y"] == 1, rng.integers(0, 1000, len(frame)), rng.integers(500, 3000, len(frame)))
    contest.create_contest(frame.drop(columns=["t"]), "leaky", target="y", metric="roc_auc", id_column="id",
                           outputs_root=tmp_path)
    found = checks.check_contest("leaky", tmp_path)
    text = " ".join(found["flags"])
    assert "chargeback" in text and "hidden rows also appear" in text and "id column" in text
    clean = _frame(6000, seed=4).drop(columns=["t"])
    contest.create_contest(clean, "clean", target="y", metric="roc_auc", id_column="id", outputs_root=tmp_path)
    assert checks.check_contest("clean", tmp_path)["flags"] == []
