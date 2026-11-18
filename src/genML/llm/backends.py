"""Ways to reach a frontier model, all returning JSON that matches a schema.

Both are subscription command-line tools: they use the plan you are already logged in
to, never per-token billing (API keys are removed from their environment).

* ``claude-cli``     Claude Code in print mode (``claude -p``), your Claude plan
* ``codex-cli``      Codex (``codex exec``), your ChatGPT plan

Every backend runs with no tools that can change anything: the Claude CLI with
all tools disabled, Codex in a read-only sandbox inside an empty folder.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BackendUnavailable(Exception):
    """The backend cannot be used here (not installed, not logged in, no key)."""


class BackendFailed(Exception):
    """The backend was tried and did not return a usable answer."""


@dataclass
class LLMRequest:
    system: str
    prompt: str
    schema: Dict[str, Any]
    purpose: str = "general"
    schema_name: str = "response"
    max_usd: float = 2.0
    timeout: int = 600


@dataclass
class BackendResult:
    data: Dict[str, Any]
    model: str
    billing: str  # "subscription" or "api"
    cost_usd: Optional[float] = None  # API spend (or the CLI's notional figure)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _clean_env(*drop: str) -> Dict[str, str]:
    """Environment for a subscription CLI without API keys (so it bills the plan, not the API)."""
    return {k: v for k, v in os.environ.items() if k not in drop}


class Backend:
    name = "base"
    billing = "api"

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings

    def check(self) -> None:
        """Raise BackendUnavailable when this backend cannot run here."""

    def run(self, request: LLMRequest) -> BackendResult:
        raise NotImplementedError


class ClaudeCLI(Backend):
    """Claude Code headless: ``claude -p --json-schema ... --tools ""``."""

    name = "claude-cli"
    billing = "subscription"

    def check(self) -> None:
        if not shutil.which(self.settings.get("claude_bin", "claude")):
            raise BackendUnavailable("claude CLI not found on PATH")

    def command(self, request: LLMRequest) -> list:
        cmd = [
            self.settings.get("claude_bin", "claude"), "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(request.schema),
            "--system-prompt", request.system,   # replaces Claude Code's large default prompt
            "--tools", "",                       # no tools: it can only answer
            "--setting-sources", "",             # no user/project settings, hooks or memory files
            "--strict-mcp-config",               # no MCP servers
            "--no-session-persistence",
            "--model", self.settings.get("claude_model", "opus"),
            "--max-budget-usd", f"{request.max_usd:.2f}",
        ]
        if self.settings.get("claude_effort"):
            cmd += ["--effort", self.settings["claude_effort"]]
        return cmd

    RETRY_SUBTYPES = ("error_max_structured_output_retries",)  # intermittent: the next call usually works

    def run(self, request: LLMRequest) -> BackendResult:
        try:
            return self._run_once(request)
        except BackendFailed as exc:
            if not any(subtype in str(exc) for subtype in self.RETRY_SUBTYPES):
                raise
            logger.warning("claude CLI: %s; retrying once", exc)
            return self._run_once(request)

    def _run_once(self, request: LLMRequest) -> BackendResult:
        with tempfile.TemporaryDirectory(prefix="genml-claude-") as workdir:
            try:
                proc = subprocess.run(
                    self.command(request), input=request.prompt, capture_output=True, text=True,
                    cwd=workdir, timeout=request.timeout,
                    env=_clean_env("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                )
            except subprocess.TimeoutExpired as exc:
                raise BackendFailed(f"claude CLI timed out after {request.timeout}s") from exc
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise BackendFailed(f"claude CLI exit {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}") from exc
        if payload.get("is_error") or payload.get("structured_output") is None:
            reason = payload.get("subtype") or payload.get("api_error_status") or payload.get("result")
            detail = str(payload.get("errors") or payload.get("result") or "")[:300]
            raise BackendFailed(f"claude CLI returned no structured output ({str(reason)[:300]})"
                                + (f": {detail}" if detail and detail != str(reason) else ""))
        usage = payload.get("usage") or {}
        model_usage = payload.get("modelUsage") or {}
        return BackendResult(
            data=payload["structured_output"],
            model=next(iter(model_usage), self.settings.get("claude_model", "opus")),
            billing=self.billing,
            cost_usd=payload.get("total_cost_usd"),
            input_tokens=(usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0),
            output_tokens=usage.get("output_tokens"),
            raw={k: payload.get(k) for k in ("session_id", "duration_ms", "stop_reason", "num_turns")},
        )


def _codex_config_model() -> Optional[str]:
    """The model Codex uses by default (``model`` in ~/.codex/config.toml), for the call log."""
    try:
        import tomllib
    except ImportError:  # Python 3.10
        return None
    path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    try:
        return tomllib.loads(path.read_text()).get("model")
    except (OSError, ValueError):
        return None


class CodexCLI(Backend):
    """Codex headless: ``codex exec --output-schema`` in a read-only sandbox."""

    name = "codex-cli"
    billing = "subscription"

    def check(self) -> None:
        binary = self.settings.get("codex_bin", "codex")
        if not shutil.which(binary):
            raise BackendUnavailable("codex CLI not found on PATH")

    def command(self, workdir: Path, schema_file: Path, out_file: Path) -> list:
        cmd = [
            self.settings.get("codex_bin", "codex"), "exec", "-",
            "--output-schema", str(schema_file),
            "--json",
            "-s", "read-only",
            "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
            "-C", str(workdir),
            "-o", str(out_file),
        ]
        if self.settings.get("codex_model"):
            cmd += ["-m", self.settings["codex_model"]]
        if self.settings.get("codex_effort"):
            cmd += ["-c", f'model_reasoning_effort="{self.settings["codex_effort"]}"']
        return cmd

    def run(self, request: LLMRequest) -> BackendResult:
        with tempfile.TemporaryDirectory(prefix="genml-codex-") as tmp:
            tmp_path = Path(tmp)
            workdir = tmp_path / "work"
            workdir.mkdir()
            schema_file, out_file = tmp_path / "schema.json", tmp_path / "last.txt"
            schema_file.write_text(json.dumps(request.schema))
            # Codex has no system-prompt flag; the instructions lead the prompt
            prompt = f"{request.system}\n\n---\n\n{request.prompt}"
            try:
                proc = subprocess.run(
                    self.command(workdir, schema_file, out_file), input=prompt, capture_output=True,
                    text=True, timeout=request.timeout, env=_clean_env("OPENAI_API_KEY"),
                )
            except subprocess.TimeoutExpired as exc:
                raise BackendFailed(f"codex CLI timed out after {request.timeout}s") from exc
            text = out_file.read_text() if out_file.exists() else ""
        usage, model = {}, self.settings.get("codex_model") or _codex_config_model() or "codex-default"
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed":
                usage = event.get("usage") or {}
            if event.get("type") in ("turn.failed", "error"):
                raise BackendFailed(f"codex CLI error: {str(event)[:300]}")
        if proc.returncode != 0 or not text.strip():
            raise BackendFailed(f"codex CLI exit {proc.returncode}: {(proc.stderr or '')[-400:]}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendFailed(f"codex CLI answer is not JSON: {text[:200]}") from exc
        return BackendResult(
            data=data, model=model, billing=self.billing,
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
        )


BACKENDS = {cls.name: cls for cls in (ClaudeCLI, CodexCLI)}


def elapsed_ms(start: float) -> int:
    return int((time.time() - start) * 1000)
