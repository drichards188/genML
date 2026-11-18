"""Git plumbing for the autopilot: a snapshot base, one worktree per cycle, guarded commits.

* ``snapshot_working_tree`` records the current working tree (tracked and untracked,
  minus ignored files) as a commit without touching the index, HEAD or any branch, so
  the autopilot can start from uncommitted work.
* Each cycle gets a worktree on ``autopilot/<problem>/<cycle>``, branched from
  ``autopilot/<problem>/main`` (the base plus every accepted cycle).
* ``commit_cycle`` reverts edits to protected paths and to tests that existed at the
  start of the cycle, then commits the rest. Agents therefore cannot change the judge or
  weaken an existing test; new tests are kept.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Paths an agent may not change: the measurement code and its inputs' definitions.
PROTECTED = (
    "src/genML/autopilot/",
    "src/genML/pipeline/diagnostics.py",
    "src/genML/pipeline/metrics.py",
    "src/genML/pipeline/oof.py",
    "tests/conftest.py",
    "tests/test_autopilot.py",
    "tests/test_autopilot_policy.py",
    "tests/test_second_opinion.py",
    "tests/test_honest_judging.py",
    "tests/test_autopilot_ops.py",
    "tests/test_probe.py",
    ".claude/",
    "problems/",
)
SCRATCH = ".autopilot"            # agent result files and scratch, never committed
NEVER_COMMIT = (SCRATCH, ".venv", "datasets", "outputs", ".playwright-mcp")
# Claude Code's sandbox guards these by mounting an empty read-only file over them; it creates that
# file when the path is missing and removes it when the session ends. A session killed midway (WSL shut
# down during C002's build, 2026-11-17) leaves them in the worktree, where `git add -A` would commit them.
SANDBOX_PLACEHOLDERS = (".bash_profile", ".bashrc", ".bash_login", ".profile", ".zshrc", ".zprofile",
                        ".gitconfig", ".gitmodules", ".ripgreprc", ".mcp.json", ".idea", ".vscode")
FALLBACK_IDENTITY = ("genML autopilot", "autopilot@localhost")


class GitError(RuntimeError):
    pass


def git(cwd: Path, *args: str, env: Optional[Dict[str, str]] = None, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {cwd}: {(proc.stderr or proc.stdout).strip()[-800:]}")
    return proc.stdout.strip()


def _identity_args(cwd: Path) -> List[str]:
    """Use the repository's configured identity; fall back to a neutral one (tests, fresh machines)."""
    name = git(cwd, "config", "user.name", check=False)
    email = git(cwd, "config", "user.email", check=False)
    if name and email:
        return []
    return ["-c", f"user.name={FALLBACK_IDENTITY[0]}", "-c", f"user.email={FALLBACK_IDENTITY[1]}"]


def _excludes(cwd: Path, paths: Sequence[str]) -> List[str]:
    """Exclude pathspecs for paths git would otherwise add (naming an ignored path is an error)."""
    return [f":(exclude){p}" for p in paths
            if subprocess.run(["git", "check-ignore", "-q", p], cwd=cwd).returncode != 0]


def repo_root(path: Path) -> Path:
    return Path(git(path, "rev-parse", "--show-toplevel"))


def snapshot_working_tree(repo: Path, message: str = "autopilot base: snapshot of the working tree") -> str:
    """Commit object holding the working tree as it is now; no ref, index or file changes."""
    repo = Path(repo)
    with tempfile.TemporaryDirectory(prefix="genml-index-") as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        has_head = bool(git(repo, "rev-parse", "--verify", "-q", "HEAD", check=False))
        if has_head:
            git(repo, "read-tree", "HEAD", env=env)
        git(repo, "add", "-A", "--", ".", *_excludes(repo, NEVER_COMMIT), env=env)
        tree = git(repo, "write-tree", env=env)
    parents = ["-p", "HEAD"] if has_head else []
    return git(repo, *_identity_args(repo), "commit-tree", tree, *parents, "-m", message)


SHARED_BRANCH = "autopilot/main"   # one code base for every problem's loop
BASE_PREFIX = "autopilot base"     # commit messages of snapshots and merges (not cycle work)


def tree_of(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", f"{ref}^{{tree}}")


def pending_cycle_commits(repo: Path, branch: str) -> List[str]:
    """Accepted cycle commits on ``branch`` that the owner's history (HEAD) does not contain yet."""
    has_head = bool(git(repo, "rev-parse", "--verify", "-q", "HEAD", check=False))
    lines = git(repo, "log", "--format=%H %s", branch, *(["--not", "HEAD"] if has_head else [])).splitlines()
    return [line.split(" ", 1)[0] for line in lines
            if line and not line.split(" ", 1)[-1].startswith(BASE_PREFIX)]


def merge_into(repo: Path, branch: str, other: str, message: str) -> str:
    """Merge commit ``other`` into ``branch`` (in a temporary worktree). On a conflict nothing changes
    and GitError says so."""
    with tempfile.TemporaryDirectory(prefix="genml-merge-") as tmp:
        path = Path(tmp) / "wt"
        git(repo, "worktree", "add", "--detach", str(path), branch)
        try:
            proc = subprocess.run(["git", *_identity_args(path), "merge", "--no-ff", "--no-edit", "-m", message, other],
                                  cwd=path, capture_output=True, text=True)
            if proc.returncode != 0:
                subprocess.run(["git", "merge", "--abort"], cwd=path, capture_output=True)
                raise GitError(f"merging {other[:10]} into {branch} conflicts: "
                               f"{(proc.stdout + proc.stderr).strip()[-600:]}")
            merged = head(path)
        finally:
            remove_worktree(repo, path)
    move_branch(repo, branch, merged)
    return merged


def refresh_shared_base(repo: Path, branch: str = SHARED_BRANCH) -> Dict[str, str]:
    """Bring the shared autopilot branch up to date with the owner's working tree (uncommitted work too).

    No branch yet: create it at a snapshot. Nothing accepted that the owner lacks: move it to the
    snapshot. Accepted cycle code the owner has not merged: merge the snapshot into it; a conflict raises
    GitError and the owner must merge ``branch`` first."""
    snapshot = snapshot_working_tree(repo)
    current = branch_head(repo, branch)
    if current is None:
        create_branch(repo, branch, snapshot)
        return {"commit": snapshot, "action": "created"}
    if tree_of(repo, current) == tree_of(repo, snapshot):
        return {"commit": current, "action": "unchanged"}
    if not pending_cycle_commits(repo, branch):
        git(repo, "update-ref", f"refs/heads/{branch}", snapshot, current)
        return {"commit": snapshot, "action": "moved"}
    merged = merge_into(repo, branch, snapshot, f"{BASE_PREFIX}: merge the working tree")
    return {"commit": merged, "action": "merged"}


def branch_head(repo: Path, branch: str) -> Optional[str]:
    return git(repo, "rev-parse", "--verify", "-q", f"refs/heads/{branch}", check=False) or None


def create_branch(repo: Path, branch: str, start: str) -> None:
    git(repo, "branch", branch, start)


def move_branch(repo: Path, branch: str, new: str) -> None:
    """Fast-forward ``branch`` to ``new`` (refuses anything that is not a fast-forward)."""
    old = branch_head(repo, branch)
    if old and old != new:
        if subprocess.run(["git", "merge-base", "--is-ancestor", old, new], cwd=repo).returncode != 0:
            raise GitError(f"{new[:10]} is not a fast-forward of {branch} ({old[:10]})")
    git(repo, "update-ref", f"refs/heads/{branch}", new, *([old] if old else []))


def add_worktree(repo: Path, path: Path, branch: str, start: str) -> Path:
    """Worktree at ``path`` on a new branch from ``start``, with the shared virtualenv linked in."""
    path = Path(path)
    if path.exists() and (path / ".git").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if branch_head(repo, branch):
        git(repo, "worktree", "add", str(path), branch)
    else:
        git(repo, "worktree", "add", "-b", branch, str(path), start)
    for shared in (".venv", "datasets"):
        source, link = Path(repo) / shared, path / shared
        if source.exists() and not link.exists():
            link.symlink_to(source, target_is_directory=True)
    (path / SCRATCH / "scratch").mkdir(parents=True, exist_ok=True)
    return path


def remove_worktree(repo: Path, path: Path) -> None:
    if Path(path).exists():
        git(repo, "worktree", "remove", "--force", str(path), check=False)
    git(repo, "worktree", "prune", check=False)


def head(cwd: Path) -> str:
    return git(cwd, "rev-parse", "HEAD")


def changed_paths(cwd: Path, base: str) -> List[str]:
    """Files that differ from ``base``: committed, staged, unstaged and untracked."""
    committed = git(cwd, "diff", "--name-only", base, "HEAD").splitlines()
    working = git(cwd, "diff", "--name-only", "HEAD").splitlines()
    untracked = git(cwd, "ls-files", "--others", "--exclude-standard").splitlines()
    skip = tuple(f"{p}/" for p in NEVER_COMMIT)
    paths = {p for p in committed + working + untracked if p and not p.startswith(skip) and p not in NEVER_COMMIT}
    return sorted(paths)


def is_protected(path: str, protected: Sequence[str] = PROTECTED) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in protected)


def exists_at(cwd: Path, commit: str, path: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], cwd=cwd,
                          capture_output=True).returncode == 0


def restore_paths(cwd: Path, base: str, paths: Sequence[str]) -> None:
    """Put ``paths`` back as they were at ``base`` (deleting files that did not exist there)."""
    for path in paths:
        if exists_at(cwd, base, path):
            git(cwd, "checkout", base, "--", path)
        else:
            git(cwd, "rm", "-q", "-f", "--cached", "--ignore-unmatch", "--", path, check=False)
            target = Path(cwd) / path
            if target.exists() and target.is_file():
                target.unlink()


def reset_worktree(cwd: Path, commit: str) -> List[str]:
    """Put the worktree back to ``commit`` exactly (scratch, the venv and data links kept).

    Returns the paths that differed and were discarded."""
    discarded = changed_paths(cwd, commit)
    git(cwd, "reset", "-q", "--hard", commit)
    excludes = [arg for path in NEVER_COMMIT for arg in ("-e", path)]
    git(cwd, "clean", "-fdq", *excludes)
    return discarded


def drop_sandbox_placeholders(cwd: Path, base: str) -> List[str]:
    """Delete the empty placeholder files a killed sandbox left at the worktree root (only names in
    ``SANDBOX_PLACEHOLDERS`` that did not exist at ``base``; anything with content stays)."""
    removed = []
    for name in SANDBOX_PLACEHOLDERS:
        path = Path(cwd) / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size == 0 \
                and not exists_at(cwd, base, name):
            path.unlink()
            removed.append(name)
    return removed


def commit_cycle(cwd: Path, base: str, message: str,
                 protected: Sequence[str] = PROTECTED) -> Dict[str, object]:
    """Guarded commit of an agent's work. Returns the commit (or None) and what was reverted."""
    leftovers = drop_sandbox_placeholders(cwd, base)
    changed = changed_paths(cwd, base)
    reverted_protected = [p for p in changed if is_protected(p, protected)]
    reverted_tests = [p for p in changed if p.startswith("tests/") and not is_protected(p, protected)
                      and exists_at(cwd, base, p)]
    restore_paths(cwd, base, reverted_protected + reverted_tests)
    git(cwd, "add", "-A", "--", ".", *_excludes(cwd, NEVER_COMMIT))
    staged = git(cwd, "diff", "--cached", "--name-only").splitlines()
    commit = None
    if staged:
        git(cwd, *_identity_args(cwd), "commit", "-q", "-m", message)
        commit = head(cwd)
    files = [p for p in git(cwd, "diff", "--name-only", base, "HEAD").splitlines() if p]
    return {"commit": commit, "files": files, "reverted_protected": reverted_protected,
            "reverted_tests": reverted_tests, "sandbox_leftovers": leftovers}


def added_test_functions(cwd: Path, base: str) -> int:
    """Number of ``def test_`` lines added under tests/ since ``base``."""
    diff = git(cwd, "diff", "-U0", base, "HEAD", "--", "tests/")
    return sum(1 for line in diff.splitlines()
               if line.startswith("+") and not line.startswith("+++") and line[1:].lstrip().startswith("def test_"))


def diff_stat(cwd: Path, base: str, ref: str = "HEAD") -> str:
    return git(cwd, "diff", "--stat", base, ref)
