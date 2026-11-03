"""Submit runs to Kaggle and keep a CV-versus-leaderboard log.

Each submission is tagged in its Kaggle description with ``[genml <run_id>]``
and recorded in ``outputs/<problem>/submissions.csv`` together with the run's
out-of-fold CV score. ``refresh_scores`` pulls public/private scores back from
Kaggle so you can check that CV tracks the leaderboard before spending more of
the daily submission budget.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.genML.pipeline.problem import load_problem
from src.genML import run_context
from src.genML.run_context import list_runs, load_run

LOG_FIELDS = ("submitted_at", "run_id", "metric", "cv_score", "best_model", "message", "file",
              "status", "public_score", "private_score", "kaggle_ref")
TAG = re.compile(r"\[genml ([^\]\s]+)")


def _log_path(problem_name: str, outputs_root: Optional[Path] = None) -> Path:
    return Path(outputs_root or run_context.OUTPUTS_ROOT) / problem_name / "submissions.csv"


def read_log(problem_name: str, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = _log_path(problem_name, outputs_root)
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_log(problem_name: str, rows: List[Dict[str, Any]], outputs_root: Optional[Path] = None) -> None:
    path = _log_path(problem_name, outputs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _competition(problem_name: str) -> str:
    spec = load_problem(problem_name)
    if spec.source.get("type") != "kaggle":
        raise ValueError(f"Problem '{problem_name}' is not a Kaggle competition (source.type != kaggle)")
    return spec.source["competition"]


def submit_run(problem_name: str, run_id: Optional[str] = None, message: str = "",
               outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Submit a run's submission.csv to Kaggle (default: the newest completed run)."""
    from src.genML.pipeline.kaggle import submit_file

    if run_id is None:
        completed = [r for r in list_runs(problem_name, outputs_root) if r.get("status") == "completed"]
        if not completed:
            raise FileNotFoundError(f"No completed runs for '{problem_name}'")
        run_id = completed[0]["run_id"]
    run = load_run(problem_name, run_id, outputs_root)
    if run.get("status") != "completed" or not run.get("submission"):
        raise ValueError(f"Run {run_id} has no submission to send")

    score = run.get("cv_score")
    score_text = f"{score:.5f}" if isinstance(score, (int, float)) else "?"
    description = f"{message} [genml {run_id} cv={score_text}]".strip()
    response = submit_file(_competition(problem_name), Path(run["submission"]), description)

    row = {
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "metric": run.get("metric"),
        "cv_score": score,
        "best_model": run.get("best_model"),
        "message": description,
        "file": run["submission"],
        "status": "submitted",
    }
    rows = read_log(problem_name, outputs_root) + [row]
    _write_log(problem_name, rows, outputs_root)
    return {**row, "kaggle_response": response}


def refresh_scores(problem_name: str, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fill public/private scores in the log from Kaggle; returns the updated log."""
    from src.genML.pipeline.kaggle import list_submissions

    remote = list_submissions(_competition(problem_name))
    by_run: Dict[str, Dict[str, Any]] = {}
    for item in remote:
        match = TAG.search(item["description"])
        if match and match.group(1) not in by_run:  # newest first
            by_run[match.group(1)] = item

    rows = read_log(problem_name, outputs_root)
    for row in rows:
        item = by_run.get(row["run_id"])
        if item:
            row.update({"status": item["status"], "public_score": item["public_score"],
                        "private_score": item["private_score"], "kaggle_ref": item["ref"]})
    _write_log(problem_name, rows, outputs_root)
    return rows
