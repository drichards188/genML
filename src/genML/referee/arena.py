"""Run a contestant: any command that reads a contest's public files and predicts its hidden rows.

The command gets ``--contest <public folder> --out <predictions.npy> --seed <n> --workdir <folder>``
and may write ``self_report.json`` into its working folder ({"validation_score": ...}: what the tool
believes its own score is). It runs under bubblewrap with no network, the contest's private folder
(labels, row codes) replaced by an empty one, credential files masked, the whole machine read-only
except its own entry folder, and inside a memory-capped systemd scope like every job.

Predictions: binary tasks give the probability of the contest's second class (``classes[1]``),
regression tasks the value, multiclass tasks one probability column per class in ``classes`` order.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.genML.referee.contest import contest_dir, load_contest, sha256

ENTRY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def sandbox_command(command: Sequence[str], *, private_dir: Path, entry_dir: Path,
                    home: Optional[Path] = None) -> List[str]:
    """``command`` wrapped in bubblewrap: no network, labels and other entries hidden, writes only in
    ``entry_dir``."""
    from src.genML.autopilot.policy import CREDENTIAL_PATHS, _home_path

    home = Path(home or Path.home())
    entry_dir = Path(entry_dir).resolve()
    args = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev",
            "--proc", "/proc", "--unshare-net",
            "--tmpfs", str(Path(private_dir).resolve()),         # the labels and row codes
            "--tmpfs", str(entry_dir.parent),                     # other entries' predictions
            "--bind", str(entry_dir), str(entry_dir)]
    for raw in CREDENTIAL_PATHS:
        path = _home_path(raw, home)
        if path.is_dir():
            args += ["--tmpfs", str(path)]
        elif path.exists():
            args += ["--ro-bind", "/dev/null", str(path)]
    return args + ["--chdir", str(entry_dir), "--", *command]


def _env(entry_dir: Path) -> Dict[str, str]:
    from src.genML.autopilot.policy import clean_env

    work = entry_dir / "work"
    env = clean_env(os.environ)
    env.update(HOME=str(work / "home"), TMPDIR=str(work / "tmp"), XDG_CACHE_HOME=str(work / "cache"),
               MPLCONFIGDIR=str(work / "tmp" / "matplotlib"), PYTHONHASHSEED="0")
    for name in ("home", "tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    return env


def _check_predictions(pred: np.ndarray, record: Dict[str, Any]) -> List[str]:
    n = record["hidden_rows"]
    errors = []
    if len(pred) != n:
        errors.append(f"{len(pred)} predictions for {n} hidden rows")
    elif not np.all(np.isfinite(np.asarray(pred, dtype=float))):
        errors.append("predictions contain NaN or infinite values")
    elif record["task"] == "multiclass" and (pred.ndim != 2 or pred.shape[1] != len(record.get("classes") or [])):
        errors.append("multiclass predictions need one column per class")
    elif record["task"] != "multiclass" and pred.ndim != 1:
        errors.append("predictions must be one value per hidden row")
    return errors


def run_entry(contest: str, entry: str, command: Sequence[str], *, seed: int = 0,
              time_limit: Optional[float] = None, sandbox: bool = True,
              outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run one contestant command for one seed; returns the entry record (also saved as entry.json)."""
    from src.genML.jobs import memory_scope

    if not ENTRY_NAME.match(entry):
        raise ValueError(f"bad entry name {entry!r}")
    record = load_contest(contest, outputs_root)
    folder = contest_dir(contest, outputs_root).resolve()      # absolute: the sandbox changes directory
    entry_dir = folder / "entries" / f"{entry}__s{seed}"
    if entry_dir.exists():
        raise FileExistsError(f"entry {entry_dir.name} exists in contest {contest!r}")
    entry_dir.mkdir(parents=True)
    out, work = entry_dir / "predictions.npy", entry_dir / "work"
    full = [*command, "--contest", str(folder / "public"), "--out", str(out), "--seed", str(seed),
            "--workdir", str(work)]
    if time_limit:
        full += ["--time-limit", str(int(time_limit))]
    env = _env(entry_dir)
    full = ["nice", "-n", "15", *full]                   # the owner's own work on this machine comes first
    if sandbox:
        full = sandbox_command(full, private_dir=folder / "private", entry_dir=entry_dir)
    full, unit, memory_note = memory_scope(full, f"genml-referee-{contest}-{entry_dir.name}"[:200], collect=True)
    started = time.time()
    with open(entry_dir / "log.txt", "w") as log:
        log.write(f"# {datetime.now().isoformat(timespec='seconds')} {' '.join(full)}\n# memory: {memory_note}\n")
        log.flush()
        try:
            exit_code = subprocess.run(full, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=entry_dir,
                                       timeout=(time_limit * 2 + 600) if time_limit else None).returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"
    result: Dict[str, Any] = {"contest": contest, "entry": entry, "seed": seed, "command": list(command),
                              "sandboxed": sandbox, "exit_code": exit_code,
                              "seconds": round(time.time() - started, 1),
                              "contest_hashes": record["hashes"], "finished_at": datetime.now().isoformat(
                                  timespec="seconds")}
    errors = [] if exit_code == 0 else [f"the command ended with {exit_code} (see log.txt)"]
    if not errors:
        if out.exists():
            pred = np.load(out, allow_pickle=False)
            errors = _check_predictions(pred, record)
            result["predictions_sha"] = sha256(out)
        else:
            errors = ["the command wrote no predictions"]
    report = work / "self_report.json"
    if report.exists():
        try:
            result["self_report"] = json.loads(report.read_text())
        except json.JSONDecodeError:
            result["self_report"] = {"error": "unreadable self_report.json"}
    result["status"] = "failed" if errors else "ok"
    result["errors"] = errors
    (entry_dir / "entry.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def entries(contest: str, outputs_root: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Finished, valid entries of a contest grouped by entry name (one record per seed)."""
    folder = contest_dir(contest, outputs_root) / "entries"
    record = load_contest(contest, outputs_root)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(folder.glob("*/entry.json")) if folder.exists() else []:
        item = json.loads(path.read_text())
        pred_path = path.parent / "predictions.npy"
        if item.get("status") != "ok" or item.get("contest_hashes") != record["hashes"]:
            continue
        if not pred_path.exists() or sha256(pred_path) != item.get("predictions_sha"):
            continue                                       # predictions changed after the run: not counted
        item["predictions_path"] = str(pred_path)
        grouped.setdefault(item["entry"], []).append(item)
    return grouped
