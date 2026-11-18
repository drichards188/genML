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
import fcntl
import json
import os
import re
import shutil
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
JOB_KINDS = ("run", "ensemble", "noise")   # noise: seed test on a finished run
ACTIVE_STATES = ("queued", "running")
# Each job runs in a systemd scope that may use at most the machine's RAM (under WSL, the VM's) and no
# swap: a job that outgrows memory is killed and marked failed, instead of pushing the machine into swap
# (on 2026-11-13 swap growing on a full host disk took WSL down). "off" disables it; a size such as
# "16G" overrides the limit.
MEMORY_CAP_ENV = "GENML_JOB_MEMORY_CAP"
_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


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


def _cmdline(pid: Any) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError, TypeError):
        return ""


def _alive(pid: Any) -> bool:
    """Running (a zombie waiting to be reaped counts as gone)."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError, TypeError):
        return False
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def _is_job_process(pid: Any) -> bool:
    """A live process that runs a genML job (directly or in its sandbox); safe against PID reuse."""
    return _alive(pid) and "src.genML.jobs" in _cmdline(pid)


def _stop(pid: int, grace: float = 30.0) -> None:
    """SIGTERM the job's process group; SIGKILL whatever is left after ``grace`` seconds."""
    try:
        group = os.getpgid(pid)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + wait
        while time.time() < deadline:
            if not _alive(pid):
                return
            time.sleep(0.2)


def cancel_job(job_id: int, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Cancel a queued job, or stop a running one (its whole process group, forcefully if it hangs)."""
    job = get_job(job_id, outputs_root)
    if job["status"] == "queued":
        _update(job_id, outputs_root, status="cancelled", finished_at=_now())
    elif job["status"] == "running":
        if job.get("pid") and _is_job_process(job["pid"]):
            _stop(int(job["pid"]))
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
    return run_job(get_job(job_id, outputs_root), outputs_root)


def run_job(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run a job record in the current process. Sandboxed autopilot runs get the record as a JSON
    file (``execute-json``) because they cannot open the job database."""
    options = job["options"]
    root = Path(outputs_root or _default_root())

    if job["kind"] == "noise":
        from src.genML.pipeline.diagnostics import best_single_model, seed_noise_test

        run_id = options["run_id"]
        model = options.get("model") or best_single_model(job["problem"], run_id, root)
        report = seed_noise_test(job["problem"], run_id, model, n_seeds=int(options.get("seeds", 3)),
                                 outputs_root=root)
        # The report lands in the measured run's folder; this job makes no run of its own.
        return {"status": "completed", "kind": "noise", "measured_run": run_id, "model": model,
                "sigma_seed": report.get("sigma_seed")}

    if job["kind"] == "ensemble":
        from src.genML.pipeline.store_ensemble import ensemble_from_store

        return ensemble_from_store(job["problem"], method=options.get("method", "stack"),
                                   top=int(options.get("top", 10)), outputs_root=root, notes=job["notes"],
                                   run_id=job["run_id"])

    from src.genML.flow import run_pipeline
    from src.genML.pipeline.problem import load_problem

    spec = load_problem(job["problem"])
    if options.get("source"):        # trusted callers only (the autopilot's shuffled-label data)
        spec.source = {**(spec.source or {}), **options["source"]}
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


def _job_code(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> tuple:
    """Where a job's code comes from: this checkout, or ``options.code_dir`` (an autopilot worktree).

    A job with ``code_dir`` runs that folder's pipeline code but reads problems and
    writes runs, OOF predictions and folds in this checkout's folders, like any run.
    """
    code_dir = (job.get("options") or {}).get("code_dir")
    if not code_dir:
        return PROJECT_ROOT, None
    from src.genML.pipeline.problem import PROBLEMS_DIR

    if not (Path(code_dir) / "src" / "genML" / "jobs.py").exists():
        raise FileNotFoundError(f"job {job['id']}: code_dir {code_dir} has no genML checkout")
    from src.genML.autopilot.policy import clean_env

    env = {**clean_env(os.environ), "GENML_OUTPUTS": str(_run_root(job, outputs_root)),
           "GENML_PROBLEMS": str(Path(PROBLEMS_DIR).resolve())}
    return Path(code_dir), env


def _run_root(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> Path:
    """Where a job's run writes: the queue's outputs root, or ``options.outputs_root`` (the autopilot's
    shuffled-label canary, kept out of the problem's real runs and OOF store)."""
    override = (job.get("options") or {}).get("outputs_root")
    return Path(override or outputs_root or _default_root()).resolve()


def _launch(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> tuple:
    """(command, cwd, env, tmp folder or None) for a job.

    Jobs with ``options.code_dir`` run code written by autopilot agents: they get the job as a
    JSON file and run inside bubblewrap per ``autopilot/run_policy.yaml`` (no network, no
    credentials, writes only to the problem's outputs, the worktree and a private temp folder). Without a
    working sandbox they do not run, unless the owner set ``options.unsandboxed``.
    """
    base = [sys.executable, "-m", "src.genML.jobs"]
    if outputs_root or (job.get("options") or {}).get("outputs_root"):
        base += ["--outputs-root", str(_run_root(job, outputs_root))]
    cwd, env = _job_code(job, outputs_root)
    options = job.get("options") or {}
    if not options.get("code_dir"):
        return base + ["execute", str(job["id"])], cwd, env, None

    from src.genML.autopilot import policy

    if "execute-json" not in (Path(cwd) / "src" / "genML" / "jobs.py").read_text():
        raise RuntimeError(f"the code in {cwd} predates sandboxed runs (its jobs.py has no execute-json); "
                           "reset this problem's autopilot so it starts from current code")
    root = Path(outputs_root or _default_root()).resolve()
    tmp = root / "jobs" / "tmp" / f"job_{job['id']}"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "job.json").write_text(json.dumps(job, default=str))
    if options.get("unsandboxed"):
        return base + ["execute-json", str(tmp / "job.json")], cwd, env, tmp
    available, reason = policy.sandbox_available()
    if not available:
        raise RuntimeError(f"refusing to run autopilot code without a sandbox: {reason}")
    command = policy.run_sandbox_command(base + ["execute-json", str(tmp / "job.json")], worktree=cwd,
                                         outputs_root=_run_root(job, outputs_root), problem=job["problem"],
                                         tmp_dir=tmp)
    return command, cwd, policy.run_env(env, tmp), tmp


# --------------------------------------------------------------------------- #
# Memory cap
# --------------------------------------------------------------------------- #

def _parse_size(text: str) -> int:
    text = text.strip().upper().removesuffix("B").removesuffix("I")
    unit = text[-1] if text and text[-1] in _SIZE_UNITS else ""
    number = float(text[:-1] if unit else text)
    if number <= 0:
        raise ValueError(f"{MEMORY_CAP_ENV} must be positive, got {text!r}")
    return int(number * _SIZE_UNITS[unit])


def _machine_memory() -> int:
    """Total RAM as Linux sees it (under WSL: the VM's ``memory`` setting, minus what the kernel keeps)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no MemTotal in /proc/meminfo")


def memory_cap_bytes() -> Optional[int]:
    """The per-job memory limit, or None when capping is off."""
    setting = os.environ.get(MEMORY_CAP_ENV, "").strip()
    if setting.lower() in ("off", "none", "false", "0", "no"):
        return None
    return _parse_size(setting) if setting else _machine_memory()


_SCOPE_CHECK: Dict[str, Any] = {}


def _scope_available() -> tuple:
    """(usable, reason): can this user start a systemd scope with a memory limit and no swap?"""
    if "result" not in _SCOPE_CHECK:
        if not shutil.which("systemd-run"):
            _SCOPE_CHECK["result"] = (False, "systemd-run not found")
        else:
            try:
                probe = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect",
                                        "-p", "MemoryMax=1G", "-p", "MemorySwapMax=0", "true"],
                                       capture_output=True, text=True, timeout=30)
                _SCOPE_CHECK["result"] = (probe.returncode == 0,
                                          (probe.stderr or probe.stdout).strip()[:300] or "ok")
            except (OSError, subprocess.SubprocessError) as exc:
                _SCOPE_CHECK["result"] = (False, str(exc)[:300])
    return _SCOPE_CHECK["result"]


def memory_scope(command: List[str], unit: str, collect: bool = False) -> tuple:
    """(command, scope unit or None, note for the log): ``command`` in a systemd user scope capped at
    ``memory_cap_bytes()`` with no swap. The scope keeps the PID (systemd-run --scope execs the
    command), so cancelling and orphan checks work as before. ``collect``: drop the unit when it ends,
    even after an out-of-memory kill (for callers that do not read its result)."""
    cap = memory_cap_bytes()
    if cap is None:
        return command, None, f"memory cap: off ({MEMORY_CAP_ENV})"
    usable, reason = _scope_available()
    if not usable:
        return command, None, f"memory cap: none, no systemd user scope ({reason})"
    unit = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    wrapped = ["systemd-run", "--user", "--scope", "--quiet", *(["--collect"] if collect else []), "--unit", unit,
               "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0", "--", *command]
    return wrapped, unit, f"memory cap: {cap / 1024 ** 3:.1f} GiB, no swap (scope {unit}.scope)"


def _capped(command: List[str], job: Dict[str, Any]) -> tuple:
    """(command, scope unit or None, note for the job log)."""
    return memory_scope(command, f"genml-job-{job['id']}-{job.get('run_id') or 'x'}")


def _memory_peak(pid: int) -> Optional[int]:
    """Peak memory of the scope the process runs in, so far."""
    try:
        path = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[-1].split(":", 2)[-1]
        if not path.endswith(".scope"):
            return None
        return int(Path("/sys/fs/cgroup" + path, "memory.peak").read_text())
    except (OSError, ValueError, IndexError):
        return None


def _scope_result(unit: str) -> str:
    """How systemd says the scope ended ('success', 'oom-kill', ...); clears a failed unit."""
    try:
        shown = subprocess.run(["systemctl", "--user", "show", "-p", "Result", "--value", f"{unit}.scope"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        subprocess.run(["systemctl", "--user", "reset-failed", f"{unit}.scope"], capture_output=True, timeout=30)
        return shown
    except (OSError, subprocess.SubprocessError):
        return ""


def _prune_after(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> Optional[str]:
    """Free the disk of the problem's runs nothing needs any more (never fails the job)."""
    try:
        from src.genML.pruning import format_report, prune_problem

        report = prune_problem(job["problem"], _run_root(job, outputs_root))
        return format_report(report) if report["pruned"] or report.get("errors") else None
    except Exception as exc:
        return f"pruning failed: {exc}"


def run_one(outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Claim the next queued job and run it in a spawned subprocess; returns the job.

    A job that cannot start (missing worktree, no sandbox) is marked failed; the worker goes on.
    The job runs under the memory cap; afterwards its peak memory goes to the job log and the
    problem's unneeded runs are pruned."""
    job = _claim_next(outputs_root)
    if job is None:
        return None
    try:
        command, cwd, env, tmp = _launch(job, outputs_root)
        command, unit, cap_note = _capped(command, job)
    except Exception as exc:
        _update(job["id"], outputs_root, status="failed", finished_at=_now(),
                error=f"could not start: {exc}"[:2000])
        return get_job(job["id"], outputs_root)
    log_path = _job_log(job, outputs_root)
    peak = None
    try:
        with open(log_path, "w") as log:
            log.write(f"# command: {' '.join(command)}\n# {cap_note}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=env)
            _update(job["id"], outputs_root, pid=process.pid)
            while True:
                try:
                    return_code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if unit:
                        peak = max(peak or 0, _memory_peak(process.pid) or 0) or None
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    out_of_memory = bool(unit) and _scope_result(unit) == "oom-kill"
    with open(log_path, "a") as log:
        if peak:
            log.write(f"\n# peak memory: {peak / 1024 ** 3:.2f} GiB (sampled every 5 s)\n")
        if out_of_memory:
            log.write(f"# killed: the job needed more memory than its cap ({cap_note})\n")

    current = get_job(job["id"], outputs_root)
    if current["status"] != "cancelled":
        summary = _read_summary(current, outputs_root)
        if current["kind"] == "noise":            # no run folder: a clean exit means the report was written
            summary = {"status": "completed"} if return_code == 0 else None
        if return_code == 0 and summary and summary.get("status") == "completed":
            _update(job["id"], outputs_root, status="completed", finished_at=_now())
        else:
            error = (summary or {}).get("error") or f"job process exited with code {return_code}"
            if out_of_memory:
                error = f"killed: out of memory ({cap_note})"
            _update(job["id"], outputs_root, status="failed", finished_at=_now(), error=str(error)[:2000])
    pruned = _prune_after(current, outputs_root)
    if pruned:
        with open(log_path, "a") as log:
            log.write(f"# pruned after this job:\n{pruned}\n")
    return get_job(job["id"], outputs_root)


def _read_summary(job: Dict[str, Any], outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = _run_root(job, outputs_root) / job["problem"] / job["run_id"] / "run.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _reap_orphans(outputs_root: Optional[Path] = None, poll_seconds: float = 5.0) -> List[int]:
    """Jobs a dead worker left 'running': wait for any that still runs, then record how each ended."""
    reaped = []
    for job in list_jobs(limit=10_000, outputs_root=outputs_root):
        if job["status"] != "running":
            continue
        while job.get("pid") and _is_job_process(job["pid"]):
            time.sleep(poll_seconds)
        summary = _read_summary(job, outputs_root)
        if summary and summary.get("status") == "completed":
            _update(job["id"], outputs_root, status="completed", finished_at=_now())
        else:
            reason = "the job worker stopped while this job ran" + (
                f"; {summary.get('error')}" if summary and summary.get("error") else "")
            _update(job["id"], outputs_root, status="failed", finished_at=_now(), error=reason[:2000])
        reaped.append(job["id"])
    return reaped


def worker_loop(outputs_root: Optional[Path] = None, poll_seconds: float = 2.0, once: bool = False) -> None:
    """Run queued jobs one at a time. Only one worker per outputs folder (a lock file); on start it first
    settles jobs a previous worker left running."""
    root = Path(outputs_root or _default_root())
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / "worker.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return                                   # another worker holds the queue
    pid_file = root / "worker.pid"
    pid_file.write_text(str(os.getpid()))
    _reap_orphans(outputs_root)
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
        lock.close()


def worker_running(outputs_root: Optional[Path] = None) -> bool:
    pid_file = Path(outputs_root or _default_root()) / "worker.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return False
    line = _cmdline(pid)
    return _alive(pid) and "src.genML.jobs" in line and "worker" in line


def ensure_worker(outputs_root: Optional[Path] = None) -> bool:
    """Start a detached worker if none is running; returns True if one was started."""
    if worker_running(outputs_root):
        return False
    command = [sys.executable, "-m", "src.genML.jobs"]
    if outputs_root:
        command += ["--outputs-root", str(outputs_root)]
    command.append("worker")
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
    execute_json = sub.add_parser("execute-json", help="(internal) run a job record saved as JSON (sandboxed runs)")
    execute_json.add_argument("path", type=Path)
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
    elif args.command == "execute-json":
        summary = run_job(json.loads(args.path.read_text()), root)
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
