"""Frontier-model access for genML through the subscription CLIs (Claude Code, Codex).

    from src.genML.llm import ask
    result = ask(system, prompt, schema, purpose="second-opinion", problem="fraud-s6e11")
    result.data  # dict matching the schema

See ``client.py`` for the order and logging, ``backends.py`` for the two backends.
The autopilot's second opinion (``autopilot/second_opinion.py``) is the main user.
"""

from src.genML.llm.backends import BACKENDS, BackendFailed, BackendUnavailable, LLMRequest
from src.genML.llm.client import LLMError, LLMResult, ask, backend_status, settings

__all__ = [
    "BACKENDS", "BackendFailed", "BackendUnavailable", "LLMError", "LLMRequest", "LLMResult",
    "ask", "backend_status", "settings",
]
