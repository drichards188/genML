"""Job queue and worker for pipeline runs.

Jobs live in a SQLite table (``outputs/jobs.db``). One worker process takes
queued jobs in order and runs each in its own spawned subprocess, so:

- only one run uses the GPU at a time (no out-of-memory from parallel runs),
- a crashing run cannot take down the API or MCP server,
- cancelling a running job is a process kill,
- the run's stdout goes to its own log file, never to an MCP stdio channel.

Usage::

    python -m src.genML.jobs worker            # process the queue until stopped
    python -m src.genML.jobs enqueue fraud-s6e11 --notes "try more trials"
    python -m src.genML.jobs list
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOB_KINDS = ("run", "ensemble")
ACTIVE_STATES = ("queued", "running")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_root() -> Path:
    from src.genML import run_context

    return Path(run_context.OUTPUTS_ROOT)


def _db_path(outputs_root: Optional[Path] = None) -> Path:
    return Path(outputs_root or _default_root()) / "jobs.db"


@contextmanager
def _connect(outputs_root: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = _db_path(outputs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                problem TEXT NOT NULL,
                options TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                run_id TEXT,
                pid INTEGER,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            )""")
        yield connection
    finally:
        connection.close()


def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    job = dict(row)
    job["options"] = json.loads(job["options"] or "{}")
    return job


def enqueue(problem: str, kind: str = "run", options: Optional[Dict[str, Any]] = None, notes: str = "",
            outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    if kind not in JOB_KINDS:
        raise ValueError(f"kind must be one of {JOB_KINDS}")
    from src.genML.run_context import new_run_id

    run_id = new_run_id()
    with _connect(outputs_root) as db:
        cursor = db.execute(
            "INSERT INTO jobs (kind, problem, options, notes, status, run_id, created_at) VALUES (?,?,?,?,?,?,?)",
            (kind, problem, json.dumps(options or {}), notes, "queued", run_id, _now()),
        )
        job_id = cursor.lastrowid
    return get_job(job_id, outputs_root)


def get_job(job_id: int, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    with _connect(outputs_root) as db:
        job = _row(db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
        if job is None:
            raise KeyError(f"No job {job_id}")
        if job["status"] == "queued":
            ahead = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running') AND id < ?",
                               (job_id,)).fetchone()[0]
            job["queue_position"] = int(ahead)
    return job


def list_jobs(limit: int = 20, problem: Optional[str] = None,
              outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    query, params = "SELECT * FROM jobs", []
    if problem:
        query, params = query + " WHERE problem = ?", [problem]
    query += " ORDER BY id DESC LIMIT ?"
    with _connect(outputs_root) as db:
        return [_row(r) for r in db.execute(query, params + [limit]).fetchall()]


def _update(job_id: int, outputs_root: Optional[Path] = None, **fields) -> None:
    columns = ", ".join(f"{name} = ?" for name in fields)
    with _connect(outputs_root) as db:
        db.execute(f"UPDATE jobs SET {columns} WHERE id = ?", list(fields.values()) + [job_id])


def cancel_job(job_id: int, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Cancel a queued job, or stop a running one by killing its process group."""
    job = get_job(job_id, outputs_root)
    if job["status"] == "queued":
        _update(job_id, outputs_root, status="cancelled", finished_at=_now())
    elif job["status"] == "running" and job.get("pid"):
        try:
            os.killpg(os.getpgid(job["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass
        _update(job_id, outputs_root, status="cancelled", finished_at=_now(), error="cancelled by user")
    return get_job(job_id, outputs_root)


def _claim_next(outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _connect(outputs_root) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            db.execute("COMMIT")
            return None
        db.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?", (_now(), row["id"]))
        db.execute("COMMIT")
    return _row(row)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def execute_job(job_id: int, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run one job in the current process (called inside the job subprocess)."""
    job = get_job(job_id, outputs_root)
    options = job["options"]
    root = Path(outputs_root or _default_root())

    if job["kind"] == "ensemble":
        from src.genML.pipeline.store_ensemble import ensemble_from_store

        return ensemble_from_store(job["problem"], method=options.get("method", "stack"),
                                   top=int(options.get("top", 10)), outputs_root=root, notes=job["notes"],
                                   run_id=job["run_id"])

    from src.genML.flow import run_pipeline
    from src.genML.pipeline.problem import load_problem

    spec = load_problem(job["problem"])
    for key in ("models", "ensemble", "cv"):
        if options.get(key) is not None:
            setattr(spec, key, options[key])
    if options.get("budget"):
        spec.budget = {**spec.budget, **options["budget"]}
    if options.get("feature_engineering"):
        spec.feature_engineering = {**spec.feature_engineering, **options["feature_engineering"]}
    spec.validate()
    return run_pipeline(spec=spec, download=bool(options.get("download")), notes=job["notes"],
                        outputs_root=root, run_id=job["run_id"])


def _job_log(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> Path:
    path = Path(outputs_root or _default_root()) / "jobs" / f"job_{job['id']}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_one(outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Claim the next queued job and run it in a spawned subprocess; returns the job."""
    job = _claim_next(outputs_root)
    if job is None:
        return None
    command = [sys.executable, "-m", "src.genML.jobs", "execute", str(job["id"])]
    if outputs_root:
        command += ["--outputs-root", str(outputs_root)]
    with open(_job_log(job, outputs_root), "w") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        _update(job["id"], outputs_root, pid=process.pid)
        return_code = process.wait()

    current = get_job(job["id"], outputs_root)
    if current["status"] == "cancelled":
        return current
    summary = _read_summary(current, outputs_root)
    if return_code == 0 and summary and summary.get("status") == "completed":
        _update(job["id"], outputs_root, status="completed", finished_at=_now())
    else:
        error = (summary or {}).get("error") or f"job process exited with code {return_code}"
        _update(job["id"], outputs_root, status="failed", finished_at=_now(), error=str(error)[:2000])
    return get_job(job["id"], outputs_root)


def _read_summary(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = Path(outputs_root or _default_root()) / job["problem"] / job["run_id"] / "run.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def worker_loop(outputs_root: Optional[Path] = None, poll_seconds: float = 2.0, once: bool = False) -> None:
    pid_file = Path(outputs_root or _default_root()) / "worker.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    try:
        while True:
            job = run_one(outputs_root)
            if job is None:
                if once:
                    return
                time.sleep(poll_seconds)
    finally:
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            pid_file.unlink()


def worker_running(outputs_root: Optional[Path] = None) -> bool:
    pid_file = Path(outputs_root or _default_root()) / "worker.pid"
    if not pid_file.exists():
        return False
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def ensure_worker(outputs_root: Optional[Path] = None) -> bool:
    """Start a detached worker if none is running; returns True if one was started."""
    if worker_running(outputs_root):
        return False
    command = [sys.executable, "-m", "src.genML.jobs", "worker"]
    if outputs_root:
        command += ["--outputs-root", str(outputs_root)]
    log = Path(outputs_root or _default_root()) / "jobs" / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if worker_running(outputs_root):
            break
        time.sleep(0.1)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="genML job queue")
    parser.add_argument("--outputs-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="Process queued jobs")
    worker.add_argument("--once", action="store_true", help="Exit when the queue is empty")
    execute = sub.add_parser("execute", help="(internal) run one job in this process")
    execute.add_argument("job_id", type=int)
    enqueue_cmd = sub.add_parser("enqueue", help="Queue a run")
    enqueue_cmd.add_argument("problem")
    enqueue_cmd.add_argument("--notes", default="")
    sub.add_parser("list", help="List recent jobs")
    cancel = sub.add_parser("cancel", help="Cancel a job")
    cancel.add_argument("job_id", type=int)
    args = parser.parse_args(argv)
    root = args.outputs_root

    if args.command == "worker":
        worker_loop(root, once=args.once)
    elif args.command == "execute":
        summary = execute_job(args.job_id, root)
        return 0 if summary.get("status") == "completed" else 1
    elif args.command == "enqueue":
        print(json.dumps(enqueue(args.problem, notes=args.notes, outputs_root=root), indent=2))
    elif args.command == "list":
        for job in list_jobs(outputs_root=root):
            print(f"{job['id']:4d} {job['status']:10s} {job['kind']:8s} {job['problem']:20s} "
                  f"{job['run_id'] or '':22s} {job['notes']}")
    elif args.command == "cancel":
        print(json.dumps(cancel_job(args.job_id, root), indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.exit(main())
