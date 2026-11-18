"""The autopilot's declared permissions: every session step and every run of agent code.

These tests pin the rules (no network, no web tools, no Kaggle, no credentials, a sandbox that is
required) so that loosening them means changing this protected file on purpose. The research step
is the one session with web access; its own rules are pinned at the end.
"""

import json
import os
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from src.genML import jobs
from src.genML.autopilot import policy
from src.genML.autopilot import research
from src.genML.autopilot import workspace as ws
from src.genML.autopilot.agent import AgentSettings, agent_env, build_command
from src.genML.autopilot.loop import LOOP_MODES, MODES_DIR

GOOD = {"tools": ["Read", "Bash"], "bash": [".venv/bin/python *"], "read": ["outputs"], "network": "none",
        "credentials": "hidden", "sandbox": "required"}


# --------------------------------------------------------------------------- #
# Session headers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", LOOP_MODES)
def test_every_loop_mode_declares_a_safe_policy(mode):
    header, prompt = policy.load_mode(MODES_DIR / f"{mode}.md")         # raises if invalid
    assert header["network"] == "none" and header["credentials"] == "hidden" and header["sandbox"] == "required"
    assert not set(header["tools"]) & {"WebFetch", "WebSearch"}
    assert all(not policy.forbidden_words(c) for c in header.get("bash") or [])
    assert not any("kaggle" in c.lower() for c in header.get("bash") or [])
    assert "{{RESULT_FILE}}" in prompt and not prompt.startswith("---")


def test_only_code_steps_may_edit_or_run_tests():
    for mode in LOOP_MODES:
        header, _ = policy.load_mode(MODES_DIR / f"{mode}.md")
        edits = "Edit" in header["tools"]
        tests = any("pytest" in c for c in header.get("bash") or [])
        assert edits == tests == (mode in ("build", "fix")), mode
    conclude, _ = policy.load_mode(MODES_DIR / "conclude.md")
    assert "Bash" not in conclude["tools"]


@pytest.mark.parametrize("change, problem", [
    ({"tools": ["Read", "WebFetch"]}, "WebFetch"),
    ({"tools": ["Read", "Bash"], "bash": [".venv/bin/kaggle kernels pull *"]}, "kaggle"),
    ({"bash": [".venv/bin/pip install *"]}, "pip"),
    ({"bash": ["git push*"]}, "git push"),
    ({"bash": ["curl *"]}, "curl"),
    ({"network": "open"}, "network"),
    ({"credentials": "visible"}, "credentials"),
    ({"sandbox": "optional"}, "sandbox"),
    ({"read": ["home"]}, "read"),
    ({"tools": ["Read"], "bash": [".venv/bin/python *"]}, "Bash tool is not"),
    ({"surprise": 1}, "unknown key"),
])
def test_unsafe_headers_are_refused(change, problem):
    assert policy.check_session_policy(GOOD) == []
    errors = policy.check_session_policy({**GOOD, **change})
    assert any(problem in e for e in errors), errors


def test_mode_without_header_is_refused(tmp_path):
    path = tmp_path / "plan.md"
    path.write_text("# Plan\n{{RESULT_FILE}}\n")
    with pytest.raises(policy.PolicyError):
        policy.load_mode(path)


def test_session_settings_cut_network_hide_credentials_and_require_the_sandbox():
    header, _ = policy.load_mode(MODES_DIR / "build.md")
    settings = policy.session_settings(header, no_write=["src/genML/autopilot/", "/abs/outputs"],
                                       writable=[Path("/wt/.autopilot/tmp")])
    sandbox = settings["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"] and not sandbox["allowUnsandboxedCommands"]
    assert not sandbox["autoAllowBashIfSandboxed"]                     # the bash list still decides
    assert sandbox["network"]["deniedDomains"] == ["*"] and sandbox["network"]["allowedDomains"] == []
    assert "~/.kaggle" in sandbox["filesystem"]["denyRead"]
    assert sandbox["filesystem"]["allowWrite"] == ["/wt/.autopilot/tmp"]
    deny = settings["permissions"]["deny"]
    assert {"WebFetch", "WebSearch", "Bash(.venv/bin/kaggle*)", "Bash(git push*)"} <= set(deny)
    assert "Edit(src/genML/autopilot/**)" in deny and "Write(//abs/outputs)" in deny
    assert "Bash(.venv/bin/pytest *)" in settings["permissions"]["allow"]
    relaxed = policy.session_settings(header, allow_unsandboxed=True)
    assert relaxed["sandbox"]["failIfUnavailable"] is False


def test_agent_command_uses_the_header(tmp_path):
    header, _ = policy.load_mode(MODES_DIR / "plan.md")
    cmd = build_command(AgentSettings(model="opus"), policy=header, settings_path=tmp_path / "s.json",
                        read_dirs=[Path("/data")], system_append="rules", session_id="abc", resume=False)
    assert cmd[:2] == ["claude", "-p"] and "--restricted" in cmd and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--tools") + 1] == ",".join(header["tools"])
    assert cmd[cmd.index("--settings") + 1] == str(tmp_path / "s.json")
    assert "--setting-sources" not in cmd and "--allowedTools" not in cmd
    assert "--add-dir /data" in " ".join(cmd) and cmd[cmd.index("--model") + 1] == "opus"


def test_session_environment_has_no_credentials(monkeypatch):
    for name in ("KAGGLE_API_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "SOME_SERVICE_API_KEY"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "cli-login")
    monkeypatch.setenv("TMPDIR", "/tmp/short")
    env = agent_env()
    assert not {"KAGGLE_API_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN",
                "SOME_SERVICE_API_KEY"} & set(env)
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "cli-login"                 # the CLI's own login stays
    assert env["TMPDIR"] == "/tmp/short"          # left alone: the sandbox's sockets need a short path


def test_shell_commands_always_start_in_the_worktree(monkeypatch):
    # A `cd` in one command must not break the worktree-relative allow rules for the rest of the
    # session (C001 and C002 on fraud-s6e11 could not run a single probe because of it).
    monkeypatch.setenv("CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR", "0")
    assert agent_env()["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] == "1"


def test_venv_commands_are_also_allowed_by_absolute_path_and_nothing_else_is(tmp_path):
    header, _ = policy.load_mode(MODES_DIR / "plan.md")
    plain = policy.session_settings(header)["permissions"]["allow"]
    settings = policy.session_settings(header, worktree=tmp_path)
    allow = settings["permissions"]["allow"]
    added = set(allow) - set(plain)
    assert added == {f"Bash({tmp_path.resolve()}/.venv/bin/python *)"}
    assert not any(policy.forbidden_words(rule[5:-1]) for rule in added)
    assert settings["permissions"]["deny"] == policy.session_settings(header)["permissions"]["deny"]
    assert settings["sandbox"] == policy.session_settings(header)["sandbox"]


# --------------------------------------------------------------------------- #
# Runs of agent code
# --------------------------------------------------------------------------- #

def test_run_profile_is_safe():
    profile = policy.load_run_policy()
    assert profile["network"] == "none" and profile["credentials"] == "hidden"
    assert "autopilot_state" in profile["read_only"]
    assert policy.check_run_policy({**profile, "network": "open"})
    assert policy.check_run_policy({**profile, "read_only": []})


def test_run_sandbox_command(tmp_path):
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".git-credentials").write_text("x")
    outputs = tmp_path / "outputs"
    (outputs / "toy" / "autopilot").mkdir(parents=True)
    wt = tmp_path / "wt"
    wt.mkdir()
    cmd = policy.run_sandbox_command(["python", "-m", "x"], worktree=wt, outputs_root=outputs, problem="toy",
                                     tmp_dir=tmp_path / "jobtmp", home=home)
    joined = " ".join(cmd)
    assert cmd[0] == "bwrap" and "--unshare-net" in cmd and cmd[-3:] == ["python", "-m", "x"]
    assert "--ro-bind / /" in joined
    problem_rw = joined.index(f"--bind {outputs / 'toy'} {outputs / 'toy'}")
    state_ro = joined.index(f"--ro-bind {outputs / 'toy' / 'autopilot'} {outputs / 'toy' / 'autopilot'}")
    assert problem_rw < state_ro                                         # the anchors stay read-only
    jobtmp = tmp_path / "jobtmp"
    assert f"--bind {wt} {wt}" in joined and f"--bind {jobtmp} {jobtmp}" in joined
    assert " /tmp " not in f" {joined} "                                # /tmp is not replaced
    assert f"--tmpfs {home / '.kaggle'}" in joined
    assert f"--ro-bind /dev/null {home / '.git-credentials'}" in joined
    assert str(home / ".ssh") not in joined                              # absent paths are skipped
    env = policy.run_env({"KAGGLE_API_TOKEN": "t", "PATH": "/bin"}, jobtmp)
    assert "KAGGLE_API_TOKEN" not in env and env["PATH"] == "/bin" and env["TMPDIR"] == str(jobtmp)


def _code_dir(tmp_path):
    code = tmp_path / "wt"
    (code / "src" / "genML").mkdir(parents=True)
    (code / "src" / "genML" / "jobs.py").write_text("# commands: execute, execute-json\n")
    return code


def test_old_code_bases_are_refused_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    code = _code_dir(tmp_path)
    (code / "src" / "genML" / "jobs.py").write_text("# commands: execute\n")
    outputs = tmp_path / "outputs"
    jobs.enqueue("toy", "run", {"code_dir": str(code)}, "old snapshot", outputs_root=outputs)
    finished = jobs.run_one(outputs)
    assert finished["status"] == "failed" and "predates sandboxed runs" in finished["error"]


def test_agent_code_does_not_run_without_a_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sandbox_available", lambda: (False, "bwrap not installed"))
    outputs = tmp_path / "outputs"
    job = jobs.enqueue("toy", "run", {"code_dir": str(_code_dir(tmp_path))}, "agent code", outputs_root=outputs)
    finished = jobs.run_one(outputs)
    assert finished["id"] == job["id"] and finished["status"] == "failed"
    assert "without a sandbox" in finished["error"]
    assert jobs.run_one(outputs) is None                                 # the worker carries on


def test_agent_code_runs_in_bubblewrap_with_the_job_as_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sandbox_available", lambda: (True, ""))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "secret")
    outputs = tmp_path / "outputs"
    code = _code_dir(tmp_path)
    job = jobs.enqueue("toy", "run", {"code_dir": str(code)}, "agent code", outputs_root=outputs)
    claimed = jobs.get_job(job["id"], outputs)
    command, cwd, env, tmp = jobs._launch(claimed, outputs)
    assert command[0] == "bwrap" and command[-2:] == ["execute-json", str(tmp / "job.json")]
    assert json.loads((tmp / "job.json").read_text())["id"] == job["id"]
    assert cwd == code and "KAGGLE_API_TOKEN" not in env
    plain = jobs.enqueue("toy", "run", {}, "trusted code", outputs_root=outputs)
    command, _, _, tmp = jobs._launch(jobs.get_job(plain["id"], outputs), outputs)
    assert command[0] == sys.executable and command[-2:] == ["execute", str(plain["id"])] and tmp is None


def test_unsandboxed_override_still_hides_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sandbox_available", lambda: (False, "bwrap not installed"))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "secret")
    outputs = tmp_path / "outputs"
    job = jobs.enqueue("toy", "run", {"code_dir": str(_code_dir(tmp_path)), "unsandboxed": True}, "",
                       outputs_root=outputs)
    command, _, env, _ = jobs._launch(jobs.get_job(job["id"], outputs), outputs)
    assert command[0] == sys.executable and command[-2] == "execute-json" and "KAGGLE_API_TOKEN" not in env


# --------------------------------------------------------------------------- #
# Worktree reset for settings ideas
# --------------------------------------------------------------------------- #

def test_reset_worktree_discards_unverified_edits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    run("init", "-q", "-b", "main")
    run("config", "user.name", "T")
    run("config", "user.email", "t@example.com")
    (repo / "pipeline.py").write_text("SAFE = True\n")
    run("add", "-A")
    run("commit", "-q", "-m", "base")
    start = ws.head(repo)
    (repo / "pipeline.py").write_text("SAFE = False\n")                  # a plan session's edit
    (repo / "extra.py").write_text("x = 1\n")
    (repo / ws.SCRATCH).mkdir()
    (repo / ws.SCRATCH / "plan.json").write_text("{}")
    discarded = ws.reset_worktree(repo, start)
    assert set(discarded) == {"pipeline.py", "extra.py"}
    assert (repo / "pipeline.py").read_text() == "SAFE = True\n" and not (repo / "extra.py").exists()
    assert (repo / ws.SCRATCH / "plan.json").exists()                     # the plan file is kept


# --------------------------------------------------------------------------- #
# The research step: web access only through the gate, blind to the problem
# --------------------------------------------------------------------------- #

RESEARCH = {"tools": ["WebFetch", "Read", "Write"], "network": "gated", "credentials": "hidden",
            "sandbox": "required", "workdir": "empty"}


def test_research_is_not_a_loop_mode_and_never_learns_the_problem():
    assert "research" not in LOOP_MODES
    header, prompt = policy.load_research_mode(MODES_DIR / "research.md")
    assert set(header["tools"]) == {"WebFetch", "Read", "Write"}
    assert "{{RESULT_FILE}}" in prompt and "{{DOMAIN}}" in prompt and "{{SOURCES_BEFORE}}" in prompt
    for placeholder in ("{{PROBLEM}}", "{{WORKTREE}}", "{{OUTPUTS_DIR}}", "{{DATA_DIR}}", "{{NOTES_FILE}}",
                        "{{CONTEXT_DIR}}", "{{TRAIN_FILE}}", "{{BEST}}"):
        assert placeholder not in prompt
    assert "—" not in prompt


@pytest.mark.parametrize("change, problem", [
    ({"tools": ["WebFetch", "Read", "Bash"]}, "Bash"),
    ({"tools": ["WebFetch", "WebSearch"]}, "WebSearch"),
    ({"tools": ["Read", "Write"]}, "needs WebFetch"),
    ({"network": "open"}, "network"),
    ({"network": "none"}, "network"),
    ({"workdir": "worktree"}, "workdir"),
    ({"credentials": "visible"}, "credentials"),
    ({"sandbox": "optional"}, "sandbox"),
    ({"read": ["outputs"]}, "unknown key"),
    ({"bash": ["ls*"]}, "unknown key"),
])
def test_unsafe_research_headers_are_refused(change, problem):
    assert policy.check_research_policy(RESEARCH) == []
    errors = policy.check_research_policy({**RESEARCH, **change})
    assert any(problem in e for e in errors), errors


def test_loop_headers_still_refuse_the_web():
    # The research header must not become a way to give a loop step web access.
    assert policy.check_session_policy({**GOOD, "network": "gated"})
    assert policy.check_session_policy({**GOOD, "tools": ["Read", "WebFetch"]})


def test_research_settings_gate_every_fetch_and_deny_answer_hosts():
    header, _ = policy.load_research_mode(MODES_DIR / "research.md")
    settings = policy.research_session_settings(header, hosts=["api.openalex.org", "arxiv.org"], gate_command="gate")
    allow, deny = settings["permissions"]["allow"], settings["permissions"]["deny"]
    assert "WebFetch(domain:api.openalex.org)" in allow and "WebFetch" not in allow     # never blanket
    assert all(not rule.startswith("Bash") for rule in allow)
    assert {"WebSearch", "Bash", "Edit"} <= set(deny)
    for host in ("kaggle.com", "www.kaggle.com", "github.com", "www.github.com", "medium.com"):
        assert f"WebFetch(domain:{host})" in deny
    hooks = settings["hooks"]["PreToolUse"]
    assert hooks == [{"matcher": "WebFetch", "hooks": [{"type": "command", "command": "gate", "timeout": 60}]}]
    sandbox = settings["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"] and not sandbox["allowUnsandboxedCommands"]
    assert sandbox["network"]["deniedDomains"] == ["*"] and "~/.kaggle" in sandbox["filesystem"]["denyRead"]
    for bad in (["www.kaggle.com"], ["user.github.io"], ["*.example.com"], ["https://example.com/x"]):
        with pytest.raises(policy.PolicyError):
            policy.research_session_settings(header, hosts=bad, gate_command="gate")
    with pytest.raises(policy.PolicyError):
        policy.research_session_settings({**header, "tools": ["WebFetch", "Bash"]}, hosts=[], gate_command="gate")


def _hook(command, event):
    return subprocess.run(shlex.split(command), input=event, capture_output=True, text=True, timeout=30)


def test_the_gate_command_does_not_depend_on_the_checkout_path(tmp_path, monkeypatch):
    # Autopilot worktrees are worktrees/<problem>/<cycle> with the virtualenv linked in: neither the
    # checkout's path nor its .venv link may reach the command the research session sees.
    checkout = tmp_path / "worktrees" / "card-fraud" / "C001"
    (checkout / ".venv" / "bin").mkdir(parents=True)
    script = checkout / "research_gate.py"
    script.write_bytes(research.GATE_SCRIPT.read_bytes())
    python = checkout / ".venv" / "bin" / "python"
    python.symlink_to(os.path.realpath(sys.executable))
    monkeypatch.setattr(research, "GATE_SCRIPT", script)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(research, "GATE_COPY_ROOT", tmp_path / "gate")
    config = research.ResearchConfig(domain="card payment fraud detection " * 3, sources_before=date(2019, 7, 15))
    command = research.gate_command(config)
    assert "fraud" not in command and str(checkout) not in command
    assert research.gate_self_test(command, config) == []
    copy = Path(shlex.split(command)[2])
    assert copy.read_bytes() == script.read_bytes() and not os.access(copy, os.W_OK)
    copy.chmod(0o644)
    copy.write_text("import sys; sys.exit(0)")                         # a changed copy is replaced
    assert research.gate_command(config) == command and copy.read_bytes() == script.read_bytes()


def test_the_gate_hook_blocks_and_fails_closed():
    config = research.ResearchConfig(domain="card payment fraud detection " * 3, sources_before=date(2019, 7, 15))
    command = research.gate_command(config)
    assert research.gate_self_test(command, config) == []
    assert "fraud" not in command                                     # the session sees this command
    fetch = lambda url: json.dumps({"tool_name": "WebFetch", "tool_input": {"url": url}})   # noqa: E731
    assert _hook(command, fetch("https://www.kaggle.com/competitions/x/discussion")).returncode == 2
    assert _hook(command, fetch("https://docs.python.org/3/")).returncode == 2   # Claude Code pre-approves it
    assert _hook(command, fetch("https://api.openalex.org/works?search=fraud")).returncode == 2   # no date limit
    dated = "https://api.openalex.org/works?search=f&filter=to_publication_date:2019-07-14"
    assert _hook(command, fetch(dated)).returncode == 0
    assert _hook(command, "not json").returncode == 2
    assert _hook(command, json.dumps({"tool_name": "WebFetch", "tool_input": {}})).returncode == 2
    assert _hook(command.replace("2019-07-15", "someday"), fetch("https://arxiv.org/abs/1905.00001v1")).returncode == 2
