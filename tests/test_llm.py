"""Tests for the LLM connection (src/genML/llm), with every backend faked; no real CLI is ever run."""

import json
import subprocess

import pytest

from src.genML.llm import backends as backends_mod
from src.genML.llm import client as client_mod
from src.genML.llm.backends import (
    Backend,
    BackendFailed,
    BackendResult,
    ClaudeCLI,
    CodexCLI,
    LLMRequest,
)
from src.genML.llm.client import LLMError, ask

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def _request():
    return LLMRequest(system="sys", prompt="What?", schema=SCHEMA, purpose="test")


def _settings(**overrides):
    base = dict(client_mod.settings())
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Subscription CLIs
# --------------------------------------------------------------------------- #

def test_claude_cli_runs_without_tools_or_api_keys(monkeypatch):
    seen = {}

    def fake_run(cmd, input, capture_output, text, cwd, timeout, env):
        seen.update(cmd=cmd, input=input, cwd=cwd, env=env)
        payload = {"is_error": False, "structured_output": {"answer": "5"}, "total_cost_usd": 0.01,
                   "usage": {"input_tokens": 3, "cache_creation_input_tokens": 1000, "output_tokens": 9},
                   "modelUsage": {"claude-opus-5-5": {}}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(backends_mod.subprocess, "run", fake_run)
    result = ClaudeCLI(_settings()).run(_request())
    assert result.data == {"answer": "5"} and result.billing == "subscription"
    assert result.model == "claude-opus-5-5" and result.input_tokens == 1003
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert cmd[cmd.index("--system-prompt") + 1] == "sys"
    assert seen["input"] == "What?"
    assert "ANTHROPIC_API_KEY" not in seen["env"]  # bills the subscription, not the API


def test_claude_cli_error_payload_fails(monkeypatch):
    payload = {"is_error": True, "subtype": "error_max_budget_usd", "structured_output": None}
    monkeypatch.setattr(backends_mod.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=json.dumps(payload), stderr=""))
    with pytest.raises(BackendFailed, match="error_max_budget_usd"):
        ClaudeCLI(_settings()).run(_request())


def test_codex_cli_reads_last_message_in_read_only_sandbox(monkeypatch):
    seen = {}

    def fake_run(cmd, input, capture_output, text, timeout, env):
        seen.update(cmd=cmd, input=input, env=env)
        out_file = cmd[cmd.index("-o") + 1]
        with open(out_file, "w") as handle:
            handle.write('{"answer": "5"}')
        events = "\n".join(json.dumps(e) for e in [
            {"type": "thread.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 120, "output_tokens": 7}},
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=events, stderr="")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(backends_mod.subprocess, "run", fake_run)
    result = CodexCLI(_settings(codex_model="gpt-test")).run(_request())
    assert result.data == {"answer": "5"} and result.input_tokens == 120
    cmd = seen["cmd"]
    assert cmd[cmd.index("-s") + 1] == "read-only" and "--ephemeral" in cmd
    assert cmd[cmd.index("-m") + 1] == "gpt-test"
    assert seen["input"].startswith("sys") and seen["input"].endswith("What?")
    assert "OPENAI_API_KEY" not in seen["env"]


# --------------------------------------------------------------------------- #
# Router: order, fallback, validation, logging
# --------------------------------------------------------------------------- #

class _Fake(Backend):
    billing = "subscription"
    answer = {"answer": "ok"}
    fail = None

    def run(self, request):
        if self.fail:
            raise BackendFailed(self.fail)
        return BackendResult(data=self.answer, model=f"{self.name}-model", billing=self.billing, cost_usd=0.5)


def _register(monkeypatch, **classes):
    registry = dict(client_mod.BACKENDS)
    registry.update(classes)
    monkeypatch.setattr(client_mod, "BACKENDS", registry)


def test_ask_falls_back_in_order_and_logs(monkeypatch, tmp_path):
    broken = type("Broken", (_Fake,), {"name": "broken", "fail": "usage limit reached"})
    good = type("Good", (_Fake,), {"name": "good"})
    _register(monkeypatch, broken=broken, good=good)
    result = ask("sys", "prompt", SCHEMA, purpose="unit", problem="toy",
                 backends=["broken", "good"], outputs_root=tmp_path)
    assert result.backend == "good" and result.data == {"answer": "ok"}
    assert [a["status"] for a in result.attempts] == ["failed", "ok"]
    log = json.loads(open(result.log_file).read())
    assert log["prompt"] == "prompt" and log["answer"] == {"answer": "ok"}
    ledger = (tmp_path / "toy" / "llm" / "calls.jsonl").read_text().splitlines()
    assert json.loads(ledger[-1])["backend"] == "good"


def test_ask_rejects_off_schema_answers(monkeypatch, tmp_path):
    sloppy = type("Sloppy", (_Fake,), {"name": "sloppy", "answer": {"wrong": 1}})
    _register(monkeypatch, sloppy=sloppy)
    with pytest.raises(LLMError, match="schema"):
        ask("sys", "prompt", SCHEMA, problem="toy", backends=["sloppy"], outputs_root=tmp_path)


def test_env_overrides_order(monkeypatch):
    monkeypatch.setenv("GENML_LLM_ORDER", "codex-cli, claude-cli")
    monkeypatch.setenv("GENML_CODEX_MODEL", "gpt-x")
    conf = client_mod.settings()
    assert conf["order"] == ["codex-cli", "claude-cli"] and conf["codex_model"] == "gpt-x"


def test_codex_is_not_in_the_default_order(monkeypatch):
    monkeypatch.delenv("GENML_LLM_ORDER", raising=False)
    assert client_mod.settings()["order"] == ["claude-cli"]


def test_only_subscription_backends_exist():
    assert sorted(client_mod.BACKENDS) == ["claude-cli", "codex-cli"]
