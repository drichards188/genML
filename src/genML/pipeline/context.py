"""Access to the problem being solved by the current run.

The data loading stage resolves the problem spec, saves it as
``reports/resolved_spec.json`` and registers it here. Later stages read it with
``current_problem()``, which falls back to the saved file so stages can still
run on their own.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.genML.pipeline import config
from src.genML.pipeline.problem import ResolvedProblem

logger = logging.getLogger(__name__)

RESOLVED_SPEC_FILENAME = "resolved_spec.json"

_active_problem: Optional[ResolvedProblem] = None
_requested_spec = None


def set_requested_spec(spec) -> None:
    """Spec to resolve in the next data loading stage (None = legacy discovery)."""
    global _requested_spec
    _requested_spec = spec


def get_requested_spec():
    return _requested_spec


def set_active_problem(problem: Optional[ResolvedProblem]) -> None:
    global _active_problem
    _active_problem = problem
    if problem is not None:
        problem.save(config.REPORTS_DIR / RESOLVED_SPEC_FILENAME)


def current_problem() -> Optional[ResolvedProblem]:
    """The resolved problem for this run, or None if no stage has resolved one."""
    if _active_problem is not None:
        return _active_problem
    path = config.REPORTS_DIR / RESOLVED_SPEC_FILENAME
    if path.exists():
        try:
            return ResolvedProblem.load(path)
        except Exception as exc:  # corrupt or old-format file
            logger.warning("Could not read %s: %s", path, exc)
    return None


def ensure_problem(train_df, test_df) -> ResolvedProblem:
    """Current problem, resolving a legacy (all-inferred) spec if none exists yet.

    Lets stages run standalone (tests, scripts) without the data loading stage.
    """
    problem = current_problem()
    if problem is not None:
        return problem
    from src.genML.pipeline.problem import legacy_spec, resolve_problem

    problem = resolve_problem(get_requested_spec() or legacy_spec(), train_df, test_df)
    set_active_problem(problem)
    return problem


def reset() -> None:
    """Clear run state (used between runs and in tests)."""
    global _active_problem, _requested_spec
    _active_problem = None
    _requested_spec = None
