"""Operations under the autopilot: one shared code base, and a job worker that cleans up after itself.

Protected: the autopilot's agents cannot change these tests.
"""

import fcntl
import json
import subprocess
import sys
import time

import pytest

from src.genML import jobs
from src.genML.autopilot import workspace as ws


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "work")
    _git(path, "config", "user.name", "Owner")
    _git(path, "config", "user.email", "owner@example.com")
    (path / ".gitignore").write_text("outputs/\n")
    (path / "a.py").write_text("A = 1\n")
    (path / "b.py").write_text("B = 1\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "start")
    return path


def _accept_cycle(repo, tmp_path, name, content):
    """What an accepted code cycle does: a commit on top of the shared branch, fast-forwarded onto it."""
    start = ws.branch_head(repo, ws.SHARED_BRANCH)
    wt = ws.add_worktree(repo, tmp_path / f"wt_{name}", f"autopilot/toy/{name}", start)
    (wt / name).write_text(content)
    commit = ws.commit_cycle(wt, start, f"{name}: accepted idea")["commit"]
    ws.remove_worktree(repo, wt)
    ws.move_branch(repo, ws.SHARED_BRANCH, commit)
    return commit


def test_shared_base_follows_the_owner_and_keeps_accepted_code(repo, tmp_path):
    first = ws.refresh_shared_base(repo)
    assert first["action"] == "created"
    assert ws.refresh_shared_base(repo)["action"] == "unchanged"
    (repo / "a.py").write_text("A = 2\n")                                  # uncommitted owner work
    moved = ws.refresh_shared_base(repo)
    assert moved["action"] == "moved" and _git(repo, "show", f"{ws.SHARED_BRANCH}:a.py") == "A = 2"

    accepted = _accept_cycle(repo, tmp_path, "feature.py", "F = 1\n")
    assert ws.pending_cycle_commits(repo, ws.SHARED_BRANCH) == [accepted]
    (repo / "b.py").write_text("B = 2\n")                                  # more owner work elsewhere
    merged = ws.refresh_shared_base(repo)
    assert merged["action"] == "merged"
    assert _git(repo, "show", f"{ws.SHARED_BRANCH}:feature.py") == "F = 1"   # accepted code kept
    assert _git(repo, "show", f"{ws.SHARED_BRANCH}:b.py") == "B = 2"         # owner's work arrived
    assert _git(repo, "rev-parse", "HEAD") == _git(repo, "rev-parse", "work")  # owner's branch untouched


def test_conflicting_owner_work_stops_the_refresh(repo, tmp_path):
    ws.refresh_shared_base(repo)
    _accept_cycle(repo, tmp_path, "a.py", "A = 'autopilot'\n")
    before = ws.branch_head(repo, ws.SHARED_BRANCH)
    (repo / "a.py").write_text("A = 'owner'\n")
    with pytest.raises(ws.GitError, match="conflicts"):
        ws.refresh_shared_base(repo)
    assert ws.branch_head(repo, ws.SHARED_BRANCH) == before                # nothing changed


def test_accepted_code_merges_when_another_loop_moved_the_branch(repo, tmp_path):
    ws.refresh_shared_base(repo)
    start = ws.branch_head(repo, ws.SHARED_BRANCH)
    wt = ws.add_worktree(repo, tmp_path / "wt_late", "autopilot/other/C001", start)
    (wt / "late.py").write_text("L = 1\n")
    late = ws.commit_cycle(wt, start, "C001: late idea")["commit"]
    ws.remove_worktree(repo, wt)
    _accept_cycle(repo, tmp_path, "early.py", "E = 1\n")                    # another problem got there first
    with pytest.raises(ws.GitError):
        ws.move_branch(repo, ws.SHARED_BRANCH, late)
    ws.merge_into(repo, ws.SHARED_BRANCH, late, "C001 (other): accepted code")
    assert _git(repo, "show", f"{ws.SHARED_BRANCH}:late.py") == "L = 1"
    assert _git(repo, "show", f"{ws.SHARED_BRANCH}:early.py") == "E = 1"


# --------------------------------------------------------------------------- #
# Job worker
# --------------------------------------------------------------------------- #

def _running(outputs, pid=None, finished=None):
    job = jobs.enqueue("toy", "run", {}, "", outputs_root=outputs)
    jobs._update(job["id"], outputs, status="running", pid=pid)
    if finished:
        run_dir = outputs / "toy" / job["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"status": finished}))
    return job["id"]


def test_a_new_worker_settles_jobs_a_dead_worker_left_running(tmp_path):
    outputs = tmp_path / "out"
    lost = _running(outputs, pid=999_999_999)
    done = _running(outputs, pid=999_999_998, finished="completed")
    assert sorted(jobs._reap_orphans(outputs, poll_seconds=0.01)) == sorted([lost, done])
    assert jobs.get_job(lost, outputs)["status"] == "failed"
    assert "worker stopped" in jobs.get_job(lost, outputs)["error"]
    assert jobs.get_job(done, outputs)["status"] == "completed"


def test_only_one_worker_runs_the_queue(tmp_path):
    outputs = tmp_path / "out"
    outputs.mkdir()
    job = jobs.enqueue("toy", "run", {}, "", outputs_root=outputs)
    with open(outputs / "worker.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs.worker_loop(outputs, once=True)                                # returns at once
    assert jobs.get_job(job["id"], outputs)["status"] == "queued"


def test_cancel_stops_a_hung_job_and_ignores_reused_pids(tmp_path):
    outputs = tmp_path / "out"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "src.genML.jobs"],
                            start_new_session=True)
    try:
        job_id = _running(outputs, pid=proc.pid)
        assert jobs._is_job_process(proc.pid)
        cancelled = jobs.cancel_job(job_id, outputs)
        assert cancelled["status"] == "cancelled"
        deadline = time.time() + 10
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
    import os

    assert not jobs._is_job_process(os.getpid())                         # pytest is not a job
    (outputs / "worker.pid").write_text(str(os.getpid()))
    assert not jobs.worker_running(outputs)                                 # a stale pid is not a worker
