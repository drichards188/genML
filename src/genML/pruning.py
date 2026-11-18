"""Free the disk space of runs nothing needs any more.

A run folder holds its input tables (``data/``) and feature matrices (``features/X_*.npy`` and the
per-fold encoding blocks): gigabytes for a large problem, and every autopilot cycle makes a new run.
What later steps use is small and lives elsewhere or stays: the OOF store (``outputs/<problem>/oof/``),
reports, predictions, the submission, ``final_test_predictions.npy`` and the small feature files.

Pruning a run deletes its tables (keeping the test ids, which a cross-run ensemble's submission needs)
and its big matrices, and writes ``pruned.json``. Runs that keep everything:

* the current best run (``experiments.current_best``),
* every run the autopilot state names (anchors, the best it tracks),
* runs a queued or running job uses (its own run, or the run a noise job refits),
* runs that have not finished and changed recently (they may still be running).

The job worker prunes a problem after each of its jobs; ``main.py --prune-runs`` does it by hand.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

PRUNED_FILE = "pruned.json"
BIG_FEATURE_PATTERNS = ("X_train.npy", "X_test.npy", "fold_train_*.npy", "fold_test_*.npy")
RECENT_HOURS = 12.0          # an unfinished run touched this recently may still be running
RUN_ID = re.compile(r"^\d{8}_\d{6}(_\d+)?$")


def _root(outputs_root: Optional[Path]) -> Path:
    from src.genML.run_context import OUTPUTS_ROOT

    return Path(outputs_root) if outputs_root else Path(OUTPUTS_ROOT)


def _run_ids_in(value: Any) -> Iterable[str]:
    """Every run id mentioned anywhere in a JSON value."""
    if isinstance(value, str):
        if RUN_ID.match(value):
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _run_ids_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _run_ids_in(item)


def _newest_change(folder: Path) -> float:
    newest = folder.stat().st_mtime
    for path in folder.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def protected_runs(problem: str, outputs_root: Optional[Path] = None) -> Dict[str, str]:
    """Run id -> why that run keeps all its files."""
    from src.genML.experiments import current_best

    root = _root(outputs_root)
    problem_dir = root / problem
    keep: Dict[str, str] = {}
    best = current_best(problem, root)
    if best:
        keep[best["run_id"]] = "current best run"

    state_file = problem_dir / "autopilot" / "state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except (OSError, ValueError):
            state = None
        if state is None:       # unreadable state: keep everything rather than guess
            return {p.name: "autopilot state unreadable" for p in problem_dir.iterdir() if RUN_ID.match(p.name)}
        for section in ("anchors", "best"):
            for run_id in _run_ids_in(state.get(section)):
                keep.setdefault(run_id, f"autopilot {section}")

    if (root / "jobs.db").exists():
        from src.genML import jobs

        for job in jobs.list_jobs(limit=100_000, problem=problem, outputs_root=root):
            if job["status"] not in jobs.ACTIVE_STATES:
                continue
            for run_id in (job.get("run_id"), (job.get("options") or {}).get("run_id")):
                if run_id:
                    keep.setdefault(run_id, f"used by {job['status']} job {job['id']}")

    if problem_dir.exists():
        cutoff = time.time() - RECENT_HOURS * 3600
        for folder in problem_dir.iterdir():
            if RUN_ID.match(folder.name) and not (folder / "run.json").exists() \
                    and _newest_change(folder) > cutoff:
                keep.setdefault(folder.name, "unfinished and changed recently")
    return keep


def _big_files(run_dir: Path) -> List[Path]:
    files: List[Path] = []
    features = run_dir / "features"
    if features.is_dir():
        for pattern in BIG_FEATURE_PATTERNS:
            files.extend(sorted(features.glob(pattern)))
    data = run_dir / "data"
    if data.is_dir():
        from src.genML.pipeline.dataset import TEST_IDS_FILE

        files.extend(sorted(p for p in data.iterdir() if p.is_file() and p.name != TEST_IDS_FILE))
    return files


def _keep_test_ids(run_dir: Path) -> None:
    """Save the test table's id column as ``data/test_ids.parquet`` before the table goes."""
    import pandas as pd

    from src.genML.pipeline.dataset import TEST_IDS_FILE, load_stage_table, stage_table_files

    data = run_dir / "data"
    target = data / TEST_IDS_FILE
    if target.exists() or not stage_table_files(data, "test"):
        return
    spec = run_dir / "reports" / "resolved_spec.json"
    id_column = json.loads(spec.read_text()).get("id_column") if spec.exists() else None
    if not id_column:
        return
    files = stage_table_files(data, "test")
    if files[0].suffix == ".parquet":
        ids = pd.read_parquet(files[0], columns=[id_column])
    else:
        ids = load_stage_table("test", data)[[id_column]]
    ids.to_parquet(target, compression="zstd")


def prune_run(run_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Delete one run's tables and big matrices (no protection checks: callers decide)."""
    run_dir = Path(run_dir)
    files = _big_files(run_dir)
    freed = sum(p.stat().st_size for p in files)
    result = {"run_id": run_dir.name, "files": len(files), "freed_bytes": freed}
    if dry_run or not files:
        return result
    _keep_test_ids(run_dir)
    for path in files:
        path.unlink(missing_ok=True)
    record = {"pruned_at": datetime.now().isoformat(timespec="seconds"), "freed_bytes": freed,
              "removed": [str(p.relative_to(run_dir)) for p in files]}
    marker = run_dir / PRUNED_FILE
    if marker.exists():                      # pruned before (e.g. files written again later)
        try:
            earlier = json.loads(marker.read_text())
            record["earlier"] = earlier
        except ValueError:
            pass
    marker.write_text(json.dumps(record, indent=2))
    return result


def prune_problem(problem: str, outputs_root: Optional[Path] = None, dry_run: bool = False) -> Dict[str, Any]:
    """Prune every run of ``problem`` that ``protected_runs`` does not keep."""
    root = _root(outputs_root)
    problem_dir = root / problem
    report: Dict[str, Any] = {"problem": problem, "dry_run": dry_run, "kept": {}, "pruned": [], "freed_bytes": 0}
    if not problem_dir.is_dir():
        return report
    keep = protected_runs(problem, root)
    report["kept"] = keep
    for folder in sorted(problem_dir.iterdir()):
        if not folder.is_dir() or not RUN_ID.match(folder.name) or folder.name in keep:
            continue
        try:
            result = prune_run(folder, dry_run=dry_run)
        except Exception as exc:              # one bad folder must not stop the rest
            logger.warning("Could not prune %s: %s", folder, exc)
            report.setdefault("errors", []).append(f"{folder.name}: {exc}")
            continue
        if result["files"]:
            report["pruned"].append(result)
            report["freed_bytes"] += result["freed_bytes"]
    return report


def format_report(report: Dict[str, Any]) -> str:
    gb = report["freed_bytes"] / 1e9
    verb = "would free" if report["dry_run"] else "freed"
    lines = [f"{report['problem']}: {len(report['pruned'])} run(s) pruned, {verb} {gb:.2f} GB"]
    for item in report["pruned"]:
        lines.append(f"  {item['run_id']}: {item['files']} files, {item['freed_bytes'] / 1e9:.2f} GB")
    for run_id, reason in sorted(report["kept"].items()):
        lines.append(f"  kept {run_id}: {reason}")
    for error in report.get("errors", []):
        lines.append(f"  error {error}")
    return "\n".join(lines)


__all__ = ["PRUNED_FILE", "RUN_ID", "format_report", "protected_runs", "prune_problem", "prune_run"]
