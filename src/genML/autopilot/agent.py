"""Claude Code sessions with tools, on the subscription (``claude -p``).

One session per autopilot step. What a step may do is declared in the header of its mode file
(``policy.py``); the session runs in the cycle's worktree with:

* only the header's tools (``--tools``) and shell commands (allow rules); anything else is denied
  without prompting, and the web tools, Kaggle and history-changing git commands never run,
* Claude Code's sandbox around every shell command: no network, credential files unreadable,
  writes only inside the worktree; without a working sandbox the session refuses to start,
* deny rules for the protected paths and for writes to the outputs and data folders,
* ``--restricted`` (no user, project or local settings, file tools confined to the worktree and
  the header's read folders), no MCP servers, and credential variables removed (API keys too,
  so it bills the plan),
* the settings file and the full event stream saved in the cycle folder.

``UsageLimited`` is raised when the plan's usage limit is hit, with the reset time
when the CLI states one; the loop sleeps and resumes the same session.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.genML.autopilot.policy import clean_env, session_settings

# The CLI's own login; every other credential variable is removed from the session.
CLI_CREDENTIALS = ("CLAUDE_CODE_OAUTH_TOKEN",)
LIMIT_PATTERN = re.compile(
    r"usage limit|limit reached|hit your limit|reached your limit|out of extra usage|"
    r"rate[_ ]limit|\b429\b|overloaded", re.IGNORECASE)   # a bare 429, not digits inside an id
EPOCH_PATTERN = re.compile(r"\|(\d{10})\b")
RESET_PATTERN = re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)


class AgentFailed(RuntimeError):
    """The session ended without doing its job (crash, timeout, error result)."""


class UsageLimited(RuntimeError):
    """The subscription's usage limit was hit; ``reset_at`` is a Unix time when known."""

    def __init__(self, message: str, reset_at: Optional[float] = None, session_id: Optional[str] = None):
        super().__init__(message)
        self.reset_at = reset_at
        self.session_id = session_id


@dataclass
class AgentSettings:
    claude_bin: str = "claude"
    model: Optional[str] = None          # None: the CLI's default model for the plan
    effort: Optional[str] = None
    max_budget_usd: float = 60.0         # runaway guard (notional on a subscription)
    extra_args: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    session_id: str
    text: str
    subtype: str
    is_error: bool
    num_turns: Optional[int]
    cost_usd: Optional[float]
    duration_s: float
    log_path: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def build_command(settings: AgentSettings, *, policy: Mapping[str, Any], settings_path: Path,
                  read_dirs: Sequence[Path], system_append: str, session_id: str, resume: bool) -> List[str]:
    """The ``claude -p`` command for a session whose header is ``policy`` and whose permissions and
    sandbox are in ``settings_path`` (see ``policy.session_settings``)."""
    cmd = [settings.claude_bin, "-p", "--output-format", "stream-json", "--verbose",
           "--restricted", "--tools", ",".join(policy["tools"]), "--settings", str(settings_path),
           "--permission-mode", "acceptEdits", "--permission-prompts", "none", "--strict-mcp-config",
           "--max-budget-usd", f"{settings.max_budget_usd:.2f}",
           "--append-system-prompt", system_append]
    for directory in read_dirs:
        cmd += ["--add-dir", str(directory)]
    cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    if settings.model:
        cmd += ["--model", settings.model]
    if settings.effort:
        cmd += ["--effort", settings.effort]
    return cmd + list(settings.extra_args)


def agent_env() -> Dict[str, str]:
    """The caller's environment without credentials (API keys too, so the CLI bills the
    subscription) and with shell timeouts long enough for the fast test suite.

    TMPDIR is left alone: Claude Code gives each session a writable temp folder, and its sandbox
    creates sockets there, which fail on paths longer than about 108 characters (a worktree path is).

    Every shell command starts in the worktree (``CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR``): the
    allow rules name worktree-relative commands such as ``.venv/bin/python``, which a ``cd`` in an
    earlier command would otherwise break for the rest of the session."""
    env = clean_env(os.environ, keep=CLI_CREDENTIALS)
    env.setdefault("BASH_DEFAULT_TIMEOUT_MS", "900000")
    env.setdefault("BASH_MAX_TIMEOUT_MS", "3600000")
    env["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] = "1"
    return env


def parse_reset(text: str, now: Optional[datetime] = None) -> Optional[float]:
    """Unix time at which a usage limit resets, from the CLI's message, if it says."""
    now = now or datetime.now()
    match = EPOCH_PATTERN.search(text)
    if match:
        return float(match.group(1))
    match = RESET_PATTERN.search(text)
    if not match:
        return None
    hour, minute, half = int(match.group(1)), int(match.group(2) or 0), (match.group(3) or "").lower()
    if half == "pm" and hour < 12:
        hour += 12
    if half == "am" and hour == 12:
        hour = 0
    if hour > 23:
        return None
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.timestamp()


def _last_result(log_path: Path) -> Optional[Dict[str, Any]]:
    result = None
    with open(log_path, errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                result = event
    return result


def run_agent(prompt: str, *, cwd: Path, log_path: Path, settings: AgentSettings, system_append: str,
              policy: Mapping[str, Any], read_dirs: Sequence[Path] = (), deny_paths: Sequence[str] = (),
              allow_unsandboxed: bool = False, session_id: Optional[str] = None,
              resume: bool = False, timeout: int = 3600, env: Optional[Dict[str, str]] = None,
              settings_override: Optional[Mapping[str, Any]] = None) -> AgentResult:
    """Run one headless session (or resume one) under ``policy`` and return its final result.

    ``settings_override``: a complete settings file to use instead of ``session_settings(policy)``
    (the research step builds its own, ``policy.research_session_settings``)."""
    session_id = session_id or str(uuid.uuid4())
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path = log_path.with_suffix(".settings.json")
    content = settings_override if settings_override is not None else session_settings(
        policy, no_write=deny_paths, worktree=Path(cwd), allow_unsandboxed=allow_unsandboxed)
    settings_path.write_text(json.dumps(content, indent=2))
    cmd = build_command(settings, policy=policy, settings_path=settings_path, read_dirs=read_dirs,
                        system_append=system_append, session_id=session_id, resume=resume)
    # Probes run inside the session: cap it like a job (machine memory, no swap), so a large probe is
    # killed instead of pushing WSL into swap on a full host disk
    from src.genML.jobs import memory_scope

    cmd, _, memory_note = memory_scope(cmd, f"genml-session-{log_path.stem}-{session_id[:8]}-{uuid.uuid4().hex[:6]}",
                                       collect=True)
    started = time.time()
    with open(log_path, "a") as log:
        log.write(json.dumps({"type": "genml_start", "at": datetime.now().isoformat(timespec="seconds"),
                              "resume": resume, "session_id": session_id, "memory": memory_note}) + "\n")
        log.flush()
        process = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, env={**agent_env(), **clean_env(env or {})},
                                   start_new_session=True)
        try:
            process.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                os.killpg(process.pid, signal.SIGKILL)
            raise AgentFailed(f"session {session_id} timed out after {timeout}s")
    result = _last_result(log_path)
    tail = log_path.read_text(errors="replace")[-3000:]
    text = (result or {}).get("result") or ""
    is_error = bool((result or {}).get("is_error")) or result is None or process.returncode not in (0, None)
    if is_error and LIMIT_PATTERN.search(f"{text}\n{tail if result is None else ''}"):
        raise UsageLimited(text or tail[-300:], reset_at=parse_reset(text or tail), session_id=session_id)
    if result is None:
        raise AgentFailed(f"session {session_id} ended without a result (exit {process.returncode}): {tail[-600:]}")
    return AgentResult(session_id=result.get("session_id") or session_id, text=text,
                       subtype=result.get("subtype", ""), is_error=is_error, num_turns=result.get("num_turns"),
                       cost_usd=result.get("total_cost_usd"), duration_s=round(time.time() - started, 1),
                       log_path=str(log_path))
