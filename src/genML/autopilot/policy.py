"""Declared permissions for the autopilot: one header per session step, one profile for runs.

Every loop mode file (``.claude/skills/genml/modes/<mode>.md``) starts with a YAML header that
says what that step's Claude Code session may do::

    ---
    tools: [Read, Glob, Grep, Write, Bash]      # built-in tools that exist in the session (--tools)
    bash: [".venv/bin/python *"]                # shell commands it may run (permission allow rules)
    read: [outputs, data, notes]                # folders outside the worktree it may read (--add-dir)
    network: none                               # sandboxed shell commands reach no host
    credentials: hidden                         # credential files unreadable, credential variables removed
    sandbox: required                           # no session without Claude Code's sandbox (bubblewrap)
    ---

``session_settings`` turns a header into the ``--settings`` file and ``build_command`` (agent.py)
into ``--tools``; ``check_session_policy`` refuses headers that would open the network, the web
tools or Kaggle. Pipeline runs of autopilot code follow ``run_policy.yaml`` (next to this file),
which the job worker enforces with bubblewrap (``run_sandbox_command``).

The research step (``research.md``, not a loop mode) is the one session with web access, under its
own header and rules (``check_research_policy``, ``research_session_settings``): only WebFetch, Read
and Write, no shell, no web search, an empty working folder, WebFetch allowed only to the listed
hosts and every fetch checked by the gate hook (``research_gate.py``), which fails closed.

Both live in protected paths (``.claude/`` and ``src/genML/autopilot/``), so an agent cannot widen
its own permissions; ``tests/test_autopilot_policy.py`` (also protected) pins the rules.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

RUN_POLICY_FILE = Path(__file__).with_name("run_policy.yaml")

SESSION_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep", "Bash", "TodoWrite")
FORBIDDEN_TOOLS = ("WebFetch", "WebSearch")
READ_PLACES = ("outputs", "data", "notes")
SESSION_KEYS = {"tools", "bash", "read", "network", "credentials", "sandbox"}
# Words (or two-word phrases) no allowed shell command may contain: network clients, package
# managers, Kaggle, history-changing git commands and deletes. This catches mistakes in a header;
# the sandbox, not this list, is what stops an allowed `python` from doing the same things.
FORBIDDEN_COMMAND_WORDS = ("kaggle", "curl", "wget", "ssh", "scp", "rsync", "nc", "pip", "pip3", "uv", "conda",
                           "mamba", "apt", "apt-get", "sudo", "rm", "git push", "git commit", "git reset",
                           "git checkout", "git branch", "git worktree", "git config", "git remote")

# Always denied, whatever a header says.
DENY_FLOOR = [
    "WebFetch", "WebSearch",
    "Bash(.venv/bin/kaggle*)", "Bash(kaggle*)", "Bash(curl*)", "Bash(wget*)",
    "Bash(git push*)", "Bash(git commit*)", "Bash(git reset*)", "Bash(git checkout*)",
    "Bash(git branch*)", "Bash(git worktree*)", "Bash(rm -rf*)",
]

# Credentials an agent or its code must never see. Paths are hidden from sandboxed shell commands
# (sessions) and masked inside run sandboxes; variables are removed from both environments.
CREDENTIAL_PATHS = (
    "~/.kaggle", "~/.config/kaggle", "~/.ssh", "~/.git-credentials", "~/.netrc", "~/.aws",
    "~/.config/gh", "~/.docker/config.json", "~/.claude/.credentials.json",
)
CREDENTIAL_ENV = (
    "KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_CONFIG_DIR",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GH_TOKEN", "GITHUB_TOKEN",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TABPFN_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
)
CREDENTIAL_SUFFIXES = ("_TOKEN", "_API_KEY", "_SECRET", "_SECRET_KEY", "_PASSWORD")

# The research step: the only session that reaches the web, and only through the gate.
RESEARCH_TOOLS = ("WebFetch", "Read", "Write")
RESEARCH_KEYS = {"tools", "network", "credentials", "sandbox", "workdir"}
HOST_PATTERN = re.compile(r"^(?=.{4,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")

RUN_WRITE_PLACES = ("problem_outputs", "progress", "worktree", "tmp")
RUN_READ_ONLY_PLACES = ("autopilot_state",)
RUN_KEYS = {"network", "credentials", "write", "read_only"}


class PolicyError(ValueError):
    """A permission header or the run profile is missing or breaks the rules."""


# --------------------------------------------------------------------------- #
# Mode files: header + prompt
# --------------------------------------------------------------------------- #

def split_mode(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """(header, prompt) of a mode file; header is None when the file has none."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        raise PolicyError("the permission header has no closing '---' line")
    header = yaml.safe_load(text[4:end]) or {}
    if not isinstance(header, dict):
        raise PolicyError("the permission header must be a YAML mapping")
    return header, text[end + len("\n---\n"):]


def load_mode(path: Path) -> Tuple[Dict[str, Any], str]:
    """Validated header and prompt of a loop mode file."""
    header, prompt = split_mode(Path(path).read_text())
    if header is None:
        raise PolicyError(f"{path} has no permission header")
    errors = check_session_policy(header)
    if errors:
        raise PolicyError(f"{path}: " + "; ".join(errors))
    return header, prompt


def forbidden_words(command: str) -> List[str]:
    """Forbidden words in a command pattern, matched on whole words (paths split at '/')."""
    words = [w for w in re.split(r"[\s/*]+", command.lower()) if w]
    phrase = f" {' '.join(words)} "
    found = []
    for word in FORBIDDEN_COMMAND_WORDS:
        if (" " in word and f" {word} " in phrase) or word in words:
            found.append(word)
    return found


def check_session_policy(policy: Mapping[str, Any]) -> List[str]:
    """Problems with a session header (empty list = acceptable)."""
    errors = [f"unknown key '{k}'" for k in policy if k not in SESSION_KEYS]
    tools = policy.get("tools")
    if not isinstance(tools, list) or not tools or not all(isinstance(t, str) for t in tools):
        errors.append("tools must be a non-empty list of tool names")
        tools = []
    for tool in tools:
        if tool in FORBIDDEN_TOOLS:
            errors.append(f"tool {tool} is not allowed (no web access for autopilot sessions)")
        elif tool not in SESSION_TOOLS:
            errors.append(f"unknown tool {tool} (known: {list(SESSION_TOOLS)})")
    bash = policy.get("bash") or []
    if not isinstance(bash, list) or not all(isinstance(c, str) and c.strip() for c in bash):
        errors.append("bash must be a list of command patterns")
        bash = []
    if "Bash" in tools and not bash:
        errors.append("a session with the Bash tool needs a bash list")
    if bash and "Bash" not in tools:
        errors.append("bash commands are listed but the Bash tool is not")
    for command in bash:
        bad = forbidden_words(command)
        if bad:
            errors.append(f"command '{command}' is not allowed ({bad[0]})")
    read = policy.get("read") or []
    if not isinstance(read, list) or any(p not in READ_PLACES for p in read):
        errors.append(f"read must be a list from {list(READ_PLACES)}")
    if policy.get("network") != "none":
        errors.append("network must be 'none'")
    if policy.get("credentials") != "hidden":
        errors.append("credentials must be 'hidden'")
    if policy.get("sandbox") != "required":
        errors.append("sandbox must be 'required'")
    return errors


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

def _rule_path(path: str) -> str:
    """Permission-rule form of a path: '//abs/path' for absolute paths, '**' under folders."""
    pattern = f"{path}**" if path.endswith("/") else path
    return "/" + pattern if pattern.startswith("/") else pattern


def bash_allow_rules(commands: Sequence[str], worktree: Optional[Path] = None) -> List[str]:
    """Allow rules for a header's shell commands. A worktree-relative ``.venv/...`` command is also
    allowed by its absolute path in ``worktree`` (the same program, so nothing new may run)."""
    rules = [f"Bash({c})" for c in commands]
    if worktree is not None:
        rules += [f"Bash({Path(worktree).resolve()}/{c})" for c in commands if c.startswith(".venv/")]
    return rules


def session_settings(policy: Mapping[str, Any], *, no_write: Sequence[str] = (),
                     writable: Sequence[Path] = (), worktree: Optional[Path] = None,
                     allow_unsandboxed: bool = False) -> Dict[str, Any]:
    """The ``--settings`` content for a session with this header.

    ``no_write``: paths the file tools may not change (protected paths, outputs, data, the main
    checkout). ``writable``: extra folders sandboxed shell commands may write (the worktree, as the
    working directory, is writable already). ``worktree``: the session's working directory, so its
    ``.venv`` commands are allowed by absolute path too. ``allow_unsandboxed`` (an owner's override)
    lets the session start without a working sandbox instead of refusing.
    """
    tools = list(policy["tools"])
    allow = [t for t in tools if t != "Bash"] + bash_allow_rules(policy.get("bash") or [], worktree)
    deny = list(DENY_FLOOR)
    for path in no_write:
        deny += [f"Edit({_rule_path(str(path))})", f"Write({_rule_path(str(path))})"]
    return {
        "permissions": {"allow": allow, "deny": deny, "defaultMode": "acceptEdits"},
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": not allow_unsandboxed,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": False,      # the bash list still decides what may run
            "filesystem": {"denyRead": list(CREDENTIAL_PATHS), "allowWrite": [str(p) for p in writable]},
            "network": {"allowedDomains": [], "deniedDomains": ["*"]},
        },
    }


# --------------------------------------------------------------------------- #
# The research session
# --------------------------------------------------------------------------- #

def check_research_policy(policy: Mapping[str, Any]) -> List[str]:
    """Problems with the research step's header (empty list = acceptable)."""
    errors = [f"unknown key '{k}'" for k in policy if k not in RESEARCH_KEYS]
    tools = policy.get("tools")
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        errors.append("tools must be a list of tool names")
        tools = []
    if "WebFetch" not in tools:
        errors.append("the research step needs WebFetch")
    errors += [f"tool {t} is not allowed in the research step (only {list(RESEARCH_TOOLS)})"
               for t in tools if t not in RESEARCH_TOOLS]
    if policy.get("network") != "gated":
        errors.append("network must be 'gated' (WebFetch only through the gate hook, nothing else)")
    if policy.get("credentials") != "hidden":
        errors.append("credentials must be 'hidden'")
    if policy.get("sandbox") != "required":
        errors.append("sandbox must be 'required'")
    if policy.get("workdir") != "empty":
        errors.append("workdir must be 'empty' (the session sees no code, data, notes or outputs)")
    return errors


def load_research_mode(path: Path) -> Tuple[Dict[str, Any], str]:
    header, prompt = split_mode(Path(path).read_text())
    if header is None:
        raise PolicyError(f"{path} has no permission header")
    errors = check_research_policy(header)
    if errors:
        raise PolicyError(f"{path}: " + "; ".join(errors))
    return header, prompt


def check_hosts(hosts: Sequence[str]) -> List[str]:
    """Problems with a list of hosts the research session may fetch from."""
    from src.genML.autopilot.research_gate import is_answer_host

    errors = []
    for host in hosts:
        if not isinstance(host, str) or not HOST_PATTERN.match(host):
            errors.append(f"'{host}' is not a plain host name (no scheme, path, port or wildcard)")
        elif is_answer_host(host):
            errors.append(f"{host} hosts solutions, notebooks or discussions and can never be a research source")
    return errors


def research_session_settings(policy: Mapping[str, Any], *, hosts: Sequence[str], gate_command: str,
                              allow_unsandboxed: bool = False) -> Dict[str, Any]:
    """The ``--settings`` content for the research session.

    WebFetch is allowed only to ``hosts``; anything else would prompt, and with no one to answer
    (``--permission-prompts none``) it is refused. Claude Code also lets some documentation sites
    through without a rule, so ``gate_command`` (``research_gate.py``) runs before every WebFetch and
    blocks anything the gate does not allow, including searches without the date limit. Answer
    hosts are denied outright as well, and the shell and web search do not exist in the session."""
    errors = check_research_policy(policy) + check_hosts(hosts)
    if errors:
        raise PolicyError("; ".join(errors))
    from src.genML.autopilot.research_gate import ANSWER_HOSTS

    allow = [t for t in policy["tools"] if t != "WebFetch"] + [f"WebFetch(domain:{h})" for h in hosts]
    deny = ["WebSearch", "Bash", "Edit", "NotebookEdit", "Glob", "Grep", "Task", "Agent"]
    deny += [f"WebFetch(domain:{h})" for host in ANSWER_HOSTS for h in (host, f"www.{host}")]
    return {
        "permissions": {"allow": allow, "deny": deny, "defaultMode": "acceptEdits"},
        "hooks": {"PreToolUse": [{"matcher": "WebFetch",
                                  "hooks": [{"type": "command", "command": gate_command, "timeout": 60}]}]},
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": not allow_unsandboxed,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": False,
            "filesystem": {"denyRead": list(CREDENTIAL_PATHS), "allowWrite": []},
            "network": {"allowedDomains": [], "deniedDomains": ["*"]},
        },
    }


def is_credential_variable(name: str) -> bool:
    return name in CREDENTIAL_ENV or name.upper().endswith(CREDENTIAL_SUFFIXES)


def clean_env(env: Mapping[str, str], keep: Sequence[str] = ()) -> Dict[str, str]:
    """``env`` without credential variables (``keep``: ones the process itself needs)."""
    return {k: v for k, v in env.items() if k in keep or not is_credential_variable(k)}


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_run_policy(path: Path = RUN_POLICY_FILE) -> Dict[str, Any]:
    policy = yaml.safe_load(Path(path).read_text()) or {}
    errors = check_run_policy(policy)
    if errors:
        raise PolicyError(f"{path}: " + "; ".join(errors))
    return policy


def check_run_policy(policy: Mapping[str, Any]) -> List[str]:
    errors = [f"unknown key '{k}'" for k in policy if k not in RUN_KEYS]
    if policy.get("network") != "none":
        errors.append("network must be 'none'")
    if policy.get("credentials") != "hidden":
        errors.append("credentials must be 'hidden'")
    write = policy.get("write") or []
    if not isinstance(write, list) or any(p not in RUN_WRITE_PLACES for p in write):
        errors.append(f"write must be a list from {list(RUN_WRITE_PLACES)}")
    read_only = policy.get("read_only") or []
    if not isinstance(read_only, list) or any(p not in RUN_READ_ONLY_PLACES for p in read_only):
        errors.append(f"read_only must be a list from {list(RUN_READ_ONLY_PLACES)}")
    if "autopilot_state" not in read_only:
        errors.append("autopilot_state (the judge's anchors) must be read_only")
    return errors


def sandbox_available() -> Tuple[bool, str]:
    """Whether bubblewrap can build a sandbox here (sessions also need socat for their proxy)."""
    missing = [tool for tool in ("bwrap", "socat") if shutil.which(tool) is None]
    if missing:
        return False, (f"{' and '.join(missing)} not installed; install with "
                       "`sudo apt install bubblewrap socat`")
    probe = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-net",
                            "true"], capture_output=True, text=True)
    if probe.returncode != 0:
        return False, f"bubblewrap cannot create a sandbox here: {(probe.stderr or probe.stdout).strip()[:300]}"
    return True, ""


def _home_path(path: str, home: Path) -> Path:
    return home / path[2:] if path.startswith("~/") else Path(path)


def run_sandbox_command(command: Sequence[str], *, worktree: Path, outputs_root: Path, problem: str,
                        tmp_dir: Path, policy: Optional[Mapping[str, Any]] = None,
                        home: Optional[Path] = None) -> List[str]:
    """``command`` wrapped in bubblewrap per the run profile.

    Everything is read-only except the places the profile lists under ``write``; places under
    ``read_only`` are read-only again even inside a writable folder; credential files are masked
    and the network is cut. Folders that must be writable are created first (bind sources must exist).
    """
    policy = policy or load_run_policy()
    home = Path(home or Path.home())
    outputs_root = Path(outputs_root).resolve()
    places = {
        "problem_outputs": outputs_root / problem,
        "progress": outputs_root / "progress",
        "worktree": Path(worktree).resolve(),
    }
    args = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    if policy["network"] == "none":
        args.append("--unshare-net")
    for name in policy.get("write") or []:
        if name == "tmp":   # a private temp folder at its real path (run_env points TMPDIR at it)
            tmp = Path(tmp_dir).resolve()
            tmp.mkdir(parents=True, exist_ok=True)
            args += ["--bind", str(tmp), str(tmp)]
            continue
        places[name].mkdir(parents=True, exist_ok=True)
        args += ["--bind", str(places[name]), str(places[name])]
    for name in policy.get("read_only") or []:
        if name == "autopilot_state":
            state = outputs_root / problem / "autopilot"
            if state.exists():
                args += ["--ro-bind", str(state), str(state)]
    if policy["credentials"] == "hidden":
        for raw in CREDENTIAL_PATHS:
            path = _home_path(raw, home)
            if path.is_dir():
                args += ["--tmpfs", str(path)]
            elif path.exists():
                args += ["--ro-bind", "/dev/null", str(path)]
    return args + ["--chdir", str(places["worktree"]), "--", *command]


def run_env(env: Mapping[str, str], tmp_dir: Path) -> Dict[str, str]:
    """Environment for a sandboxed run: no credentials, temp files and caches in its private folder.

    /tmp itself stays read-only like the rest of the machine, so paths are the same inside and outside
    the sandbox (replacing /tmp would hide anything that lives under it, such as test folders)."""
    tmp = str(Path(tmp_dir).resolve())
    cleaned = clean_env(env)
    cleaned.update(TMPDIR=tmp, MPLCONFIGDIR=f"{tmp}/matplotlib", NUMBA_CACHE_DIR=f"{tmp}/numba")
    return cleaned


__all__ = ["CREDENTIAL_ENV", "CREDENTIAL_PATHS", "DENY_FLOOR", "PolicyError", "RESEARCH_TOOLS", "check_hosts",
           "check_research_policy", "load_research_mode", "research_session_settings", "check_run_policy",
           "check_session_policy", "clean_env", "is_credential_variable", "load_mode", "load_run_policy",
           "run_env", "run_sandbox_command", "sandbox_available", "session_settings", "split_mode"]
