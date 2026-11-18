"""One entry point for asking a frontier model: ``ask()``.

``ask`` tries the configured backends in order (subscriptions first, see
``config.LLM_CONFIG['order']``), checks the answer against the JSON schema, and
moves on to the next backend when one is unavailable, fails or answers off-schema.

Every call is recorded for audit under ``outputs/<problem>/llm/``: the full
request and answer as ``<timestamp>_<purpose>.json`` and one line in
``calls.jsonl`` (backend, model, tokens, time). Both backends bill a subscription, never per
token; ``max_usd_per_call`` is only the Claude CLI's runaway guard.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.genML.llm.backends import (
    BACKENDS,
    BackendFailed,
    BackendResult,
    BackendUnavailable,
    LLMRequest,
    elapsed_ms,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """No backend produced a valid answer."""


@dataclass
class LLMResult:
    data: Dict[str, Any]
    backend: str
    model: str
    billing: str
    cost_usd: Optional[float]
    duration_ms: int
    log_file: Optional[str]
    attempts: List[Dict[str, Any]] = field(default_factory=list)


def settings() -> Dict[str, Any]:
    """LLM settings from config, with environment overrides (GENML_LLM_ORDER, GENML_*_MODEL)."""
    from src.genML.pipeline import config

    merged = dict(config.LLM_CONFIG)
    if os.environ.get("GENML_LLM_ORDER"):
        merged["order"] = [b.strip() for b in os.environ["GENML_LLM_ORDER"].split(",") if b.strip()]
    for key in ("claude_model", "codex_model"):
        env = os.environ.get(f"GENML_{key.upper()}")
        if env:
            merged[key] = env
    return merged


def backend_status(order: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Which backends can run here, in the order ``ask`` would try them."""
    conf = settings()
    rows = []
    for name in order or conf["order"]:
        cls = BACKENDS.get(name)
        if cls is None:
            rows.append({"backend": name, "available": False, "reason": "unknown backend"})
            continue
        try:
            cls(conf).check()
            rows.append({"backend": name, "available": True, "billing": cls.billing, "reason": ""})
        except BackendUnavailable as exc:
            rows.append({"backend": name, "available": False, "billing": cls.billing, "reason": str(exc)})
    return rows


def _validate(data: Any, schema: Dict[str, Any]) -> None:
    """Raise BackendFailed when the answer does not match the schema."""
    try:
        import jsonschema
    except ImportError:  # minimal check without jsonschema
        if not isinstance(data, dict) or any(k not in data for k in schema.get("required", [])):
            raise BackendFailed("answer is missing required fields")
        return
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise BackendFailed(f"answer does not match the schema: {exc.message}") from exc


def _log_dir(problem: Optional[str], outputs_root: Optional[Path]) -> Path:
    from src.genML.run_context import OUTPUTS_ROOT

    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    return root / (problem or "_general") / "llm"


def ask(system: str, prompt: str, schema: Dict[str, Any], *, purpose: str = "general",
        problem: Optional[str] = None, backends: Optional[Sequence[str]] = None,
        outputs_root: Optional[Path] = None, max_usd: Optional[float] = None,
        timeout: Optional[int] = None) -> LLMResult:
    """Ask for a JSON answer matching ``schema``; try backends in order until one succeeds.

    ``backends`` overrides the configured order (e.g. ``["codex-cli"]``).
    """
    conf = settings()
    order = list(backends or conf["order"])
    request = LLMRequest(
        system=system, prompt=prompt, schema=schema, purpose=purpose,
        schema_name=purpose.replace("-", "_") or "response",
        max_usd=float(max_usd if max_usd is not None else conf["max_usd_per_call"]),
        timeout=int(timeout or conf["timeout_s"]),
    )
    log_dir = _log_dir(problem, outputs_root)
    attempts: List[Dict[str, Any]] = []
    for name in order:
        cls = BACKENDS.get(name)
        if cls is None:
            attempts.append({"backend": name, "status": "unknown"})
            continue
        backend = cls(conf)
        start = time.time()
        try:
            backend.check()
            result: BackendResult = backend.run(request)
            _validate(result.data, schema)
        except BackendUnavailable as exc:
            attempts.append({"backend": name, "status": "unavailable", "reason": str(exc)})
            continue
        except BackendFailed as exc:
            attempts.append({"backend": name, "status": "failed", "reason": str(exc), "ms": elapsed_ms(start)})
            logger.warning("LLM backend %s failed: %s", name, exc)
            continue
        duration = elapsed_ms(start)
        attempts.append({"backend": name, "status": "ok", "ms": duration})
        log_file = _record(log_dir, request, result, name, duration, attempts, problem)
        return LLMResult(data=result.data, backend=name, model=result.model, billing=result.billing,
                         cost_usd=result.cost_usd, duration_ms=duration, log_file=str(log_file),
                         attempts=attempts)
    _record_failure(log_dir, request, attempts, problem)
    summary = "; ".join(f"{a['backend']}: {a['status']} {a.get('reason', '')}".strip() for a in attempts)
    raise LLMError(f"No LLM backend answered ({summary})")


def _record(log_dir: Path, request: LLMRequest, result: BackendResult, backend: str, duration: int,
            attempts: List[Dict[str, Any]], problem: Optional[str]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = log_dir / f"{stamp}_{request.purpose}.json"
    path.write_text(json.dumps({
        "problem": problem, "purpose": request.purpose, "backend": backend, "model": result.model,
        "billing": result.billing, "cost_usd": result.cost_usd, "duration_ms": duration,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "attempts": attempts, "system": request.system, "prompt": request.prompt,
        "schema": request.schema, "answer": result.data, "raw": result.raw,
    }, indent=2, default=str))
    with open(log_dir / "calls.jsonl", "a") as ledger:
        ledger.write(json.dumps({
            "time": stamp, "purpose": request.purpose, "backend": backend, "model": result.model,
            "billing": result.billing, "cost_usd": result.cost_usd if result.billing == "api" else None,
            "notional_usd": result.cost_usd if result.billing == "subscription" else None,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "duration_ms": duration, "log": path.name,
        }) + "\n")
    return path


def _record_failure(log_dir: Path, request: LLMRequest, attempts: List[Dict[str, Any]],
                    problem: Optional[str]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    with open(log_dir / "calls.jsonl", "a") as ledger:
        ledger.write(json.dumps({"time": stamp, "purpose": request.purpose, "backend": None,
                                 "status": "failed", "attempts": attempts}) + "\n")
