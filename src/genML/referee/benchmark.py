"""The referee's first benchmark, on a contest made from real data (default ``ieee-100k``).

1. Noise only: a fixed LightGBM run with 15 seeds, grouped into five 3-seed entries, plus one entry with
   ten pure-noise columns. Every comparison among them is a change that cannot help: how often does the
   referee call one a gain, and are the differences the size its noise estimate says?
2. Planted real gains: the same LightGBM with a group of real columns withheld (a baseline missing
   something useful) against the full one. Which gains does it prove, and do the holdout rows agree?
3. Planted leaks: contests made from the same rows with a column recorded after the outcome, duplicated
   rows, or an id that tracks the label, plus clean controls. What do the referee's checks and
   Deepchecks' train-test suite flag on each?
4. AutoGluon: three seeds as people use it, and one validated on the latest training rows without the
   time column: how much does it overstate itself, and how much of that is its validation?

Resumable: finished entries are kept and skipped. Writes ``outputs/referee/benchmark/<contest>.json``.

    python -m src.genML.referee.benchmark [--contest ieee-100k] [--skip-autogluon]
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.genML.referee import arena, checks, contest, verdict

ROOT = Path(__file__).resolve().parents[3]
LGB = [sys.executable, str(ROOT / "contestants" / "lightgbm_contestant.py")]
AUTOGLUON = [str(Path.home() / ".venvs" / "genml-autogluon" / "bin" / "python"),
             str(ROOT / "contestants" / "autogluon_contestant.py"), "--presets", "medium_quality"]
DEEPCHECKS = [str(Path.home() / ".venvs" / "genml-deepchecks" / "bin" / "python"),
              str(ROOT / "external" / "deepchecks_check.py")]
NULL_GROUPS = {f"lgb-full-{k}": list(range(3 * i, 3 * i + 3)) for i, k in enumerate("abcde")}
WITHHELD = {"lgb-minus-C": ["--drop-prefix", "C"], "lgb-minus-card": ["--drop-prefix", "card"],
            "lgb-minus-M": ["--drop-prefix", "M"], "lgb-minus-D": ["--drop", *[f"D{k}" for k in range(1, 16)]],
            "lgb-minus-V": ["--drop-prefix", "V"], "lgb-minus-amount": ["--drop", "TransactionAmt"]}
AG_LIMIT = 600


def _enter(name: str, entry: str, command: List[str], seeds, time_limit: Optional[float] = None,
           outputs_root: Optional[Path] = None) -> None:
    folder = contest.contest_dir(name, outputs_root) / "entries"
    for seed in seeds:
        done = folder / f"{entry}__s{seed}" / "entry.json"
        if done.exists() and json.loads(done.read_text()).get("status") == "ok":
            continue
        if done.parent.exists():                       # a failed or interrupted attempt: try again
            import shutil
            shutil.rmtree(done.parent)
        result = arena.run_entry(name, entry, command, seed=seed, time_limit=time_limit, outputs_root=outputs_root)
        print(f"  {entry} seed {seed}: {result['status']} in {result['seconds']} s", flush=True)


def _brief(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("base", "candidate", "grade", "gain", "se_total", "smallest_provable_gain", "chunks_won",
            "holdout_gain", "flags")
    out = {k: result[k] for k in keys}
    out["z"] = result["gain"] / result["se_total"] if result["se_total"] else None
    return out


def noise_only(name: str, outputs_root=None) -> Dict[str, Any]:
    for entry, seeds in NULL_GROUPS.items():
        _enter(name, entry, LGB, seeds, outputs_root=outputs_root)
    _enter(name, "lgb-noise10", [*LGB, "--noise-columns", "10"], [20, 21, 22], outputs_root=outputs_root)
    pairs = list(itertools.combinations(NULL_GROUPS, 2)) + [(g, "lgb-noise10") for g in NULL_GROUPS]
    results = [_brief(verdict.compare(name, a, b, outputs_root=outputs_root)) for a, b in pairs]
    grades = [r["grade"] for r in results]
    z = np.array([r["z"] for r in results])
    return {"comparisons": results, "grades": {g: grades.count(g) for g in sorted(set(grades))},
            "abs_z_mean": float(np.mean(np.abs(z))), "abs_z_max": float(np.max(np.abs(z)))}


def planted_gains(name: str, outputs_root=None) -> Dict[str, Any]:
    for entry, extra in WITHHELD.items():
        _enter(name, entry, [*LGB, *extra], [30, 31, 32], outputs_root=outputs_root)
    results = [_brief(verdict.compare(name, entry, "lgb-full-a", outputs_root=outputs_root)) for entry in WITHHELD]
    return {"comparisons": sorted(results, key=lambda r: -r["gain"])}


def _frame(name: str, outputs_root=None) -> pd.DataFrame:
    """The contest's rows back in one table (train rows, then hidden rows with their labels)."""
    record = contest.load_contest(name, outputs_root)
    public = contest.contest_dir(name, outputs_root) / "public"
    hidden = pd.read_parquet(public / "hidden.parquet")
    hidden[record["target"]] = contest.labels_and_rows(name, outputs_root)[0]
    return pd.concat([pd.read_parquet(public / "train.parquet"), hidden], ignore_index=True)


def _deepchecks(name: str, outputs_root=None) -> Dict[str, Any]:
    folder = contest.contest_dir(name, outputs_root)
    out = folder / "deepchecks.json"
    try:
        subprocess.run([*DEEPCHECKS, "--contest-dir", str(folder), "--out", str(out)], check=True, timeout=1800,
                       capture_output=True, text=True)
        return json.loads(out.read_text())
    except subprocess.CalledProcessError as exc:
        return {"error": (exc.stderr or "")[-600:]}
    except subprocess.TimeoutExpired:
        return {"error": "timed out after 30 minutes"}


def planted_leaks(name: str, outputs_root=None) -> Dict[str, Any]:
    record = contest.load_contest(name, outputs_root)
    target, id_column, time_column = record["target"], record["id_column"], record["time_column"]
    frame = _frame(name, outputs_root)
    rng = np.random.default_rng(20261121)
    y = (frame[target] == record["classes"][1]).to_numpy(int)
    leaky = frame.copy()
    flip = rng.random(len(leaky)) < 0.05
    leaky["chargeback_flag"] = np.where(flip, 1 - y, y)          # known only after the outcome
    dupes = pd.concat([frame, frame.sample(frac=0.05, random_state=7)], ignore_index=True)
    telling = frame.copy()
    telling[id_column] = np.where(y == 1, rng.integers(0, 4_000_000, len(y)), rng.integers(2_000_000, 10_000_000,
                                                                                           len(y)))
    plants = {  # name: (table, time column for the split, what is planted)
        "bench-clean-time": (frame, time_column, None),
        "bench-clean-random": (frame, None, None),
        "bench-leaky-column": (leaky, time_column, "chargeback_flag: the label with 5% flipped"),
        "bench-duplicates": (dupes, None, "5% of rows duplicated, split at random"),
        "bench-telling-id": (telling, time_column, "ids drawn from a lower range for positives"),
    }
    out = {}
    for plant, (table, time_col, planted) in plants.items():
        contest.create_contest(table, plant, target=target, metric=record["metric"], id_column=id_column,
                               time_column=time_col, task=record["task"], outputs_root=outputs_root,
                               overwrite=True, source={"from_contest": name, "planted": planted})
        print(f"  {plant}: checks and Deepchecks", flush=True)
        out[plant] = {"planted": planted, "referee_flags": checks.check_contest(plant, outputs_root)["flags"],
                      "deepchecks": _deepchecks(plant, outputs_root)}
    # what the leaky column does to a model's score: with it against without it
    _enter("bench-leaky-column", "lgb-without", [*LGB, "--drop", "chargeback_flag"], [0, 1, 2],
           outputs_root=outputs_root)
    _enter("bench-leaky-column", "lgb-with", LGB, [0, 1, 2], outputs_root=outputs_root)
    out["bench-leaky-column"]["verdict_with_leak"] = _brief(
        verdict.compare("bench-leaky-column", "lgb-without", "lgb-with", outputs_root=outputs_root))
    return out


def autogluon(name: str, outputs_root=None) -> Dict[str, Any]:
    _enter(name, "ag-medium", AUTOGLUON, [0, 1, 2], time_limit=AG_LIMIT, outputs_root=outputs_root)
    _enter(name, "ag-timeval", [*AUTOGLUON, "--time-validation", "0.2"], [0], time_limit=AG_LIMIT,
           outputs_root=outputs_root)
    summary = verdict.entry_summary(name, outputs_root)
    return {k: summary[k] for k in ("ag-medium", "ag-timeval", "lgb-full-a") if k in summary}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contest", default="ieee-100k")
    parser.add_argument("--skip-autogluon", action="store_true")
    args = parser.parse_args(argv)
    report_path = contest.referee_root() / "benchmark" / f"{args.contest}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"contest": args.contest, "started_at": datetime.now().isoformat(timespec="seconds")}
    steps = [("noise_only", noise_only), ("planted_gains", planted_gains), ("planted_leaks", planted_leaks)]
    if not args.skip_autogluon:
        steps.append(("autogluon", autogluon))
    for key, step in steps:
        print(f"[{datetime.now():%H:%M:%S}] {key}", flush=True)
        try:
            report[key] = step(args.contest)
        except Exception as exc:                          # keep what the other steps found
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}
        report["updated_at"] = datetime.now().isoformat(timespec="seconds")
        report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[{datetime.now():%H:%M:%S}] done: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
