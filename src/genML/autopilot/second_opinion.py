"""A second opinion on the autopilot's work, from a model that did not do it.

The autopilot plans and builds with Claude Code sessions. At two points the loop asks a fresh model
with no tools and none of the session's context (Claude through ``claude -p``; Codex, a different
model family, when ``ORDER`` names it and a ChatGPT plan is signed in) to check that work against the
project's rules and the measurements:

* ``PLAN_QUESTION``: after a plan, before a build or a run is spent on it,
* ``GAIN_QUESTION``: when the judge holds a gain as suspicious (a likely leak), to help the person
  who must check it,

and ``CYCLE_QUESTION`` is for a person asking about any cycle (``main.py --second-opinion``).

The answer is advice. The loop records it, gives a plan one chance to be revised, and fails a cycle
only when the revised plan is rejected again; the trusted judge still decides what counts. The
reviewer runs without tools (Codex in a read-only sandbox in an empty folder, Claude with no tools)
and sees only the material it is given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SHARED_PROMPT = Path(__file__).resolve().parents[3] / ".claude" / "skills" / "genml" / "modes" / "_shared.md"
# Claude only since 2026-11-17 (Codex no longer signed in). Put "codex-cli" first again to have a
# different model family review the Claude sessions' work.
ORDER = ("claude-cli",)
VERDICTS = ("ok", "concerns", "reject")
CONCERN_KINDS = ("banned_source", "leak", "generator_artifact", "no_domain_reason", "weak_evidence",
                 "repeat", "cost", "other")
MAX_DIFF_CHARS = 40_000
MAX_NOTES_CHARS = 8_000

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "summary", "concerns", "suggestion"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "summary": {"type": "string", "description": "Two to four sentences: what you checked and found."},
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "detail"],
                "properties": {
                    "kind": {"type": "string", "enum": list(CONCERN_KINDS)},
                    "detail": {"type": "string", "description": "Specific, citing the material."},
                },
            },
        },
        "suggestion": {"type": "string", "description": "What to change; empty if nothing."},
    },
}

SYSTEM_TEMPLATE = """You give a second opinion on work by genML's autopilot. genML is an automated \
pipeline for tabular machine learning; the autopilot is a loop of AI coding sessions (a different \
model from you) that plans experiments, writes code for them, and has trusted code measure the \
result on fixed cross-validation folds.

The project's rules for every idea:

{rules}

How to answer:
- Base every point on the material you are given and cite it. You cannot see anything else (no \
data, no files, no web). Do not invent measurements.
- verdict "reject": the work breaks a rule above (an idea from this problem's published solutions, \
write-ups, discussions or public notebooks; pooling other people's predictions; modelling or \
exploiting how the dataset was generated; labels leaking into a fold's own model), or a measured \
gain is most likely a leak.
- verdict "concerns": no rule is broken, but something should change first: weak or missing \
evidence, no domain reason a practitioner would accept, a repeat of a measured idea, a cost out of \
proportion to the expected gain, or a result you cannot judge from the material.
- verdict "ok": sound as it is. Say so plainly; do not invent concerns.
- A gain is real only if it beats the noise threshold and wins most folds; treat smaller \
differences as ties."""

PLAN_QUESTION = (
    "The autopilot plans to try the idea below next. Check it before a build or run is spent on it: "
    "does it break a rule, does every feature have a domain reason, does the evidence support the "
    "predicted gain, and was it (or something close) already measured? Where the idea has probe_results "
    "(cheap screens of one untuned model, on rows the judge does not decide on), check that a probe tested "
    "what will be built and that the predicted gain is not above the probe's gain. Use 'reject' only for a "
    "rule break.")

GAIN_QUESTION = (
    "The judge measured the gain below. It is above the limit where gains are held for a person, "
    "because a jump that large from one change is often a leak. Read the change and the numbers: is "
    "the gain most likely a leak (labels reaching a fold's own model, test information, a generator "
    "artifact) or plausible? Say exactly what the person should check. Use 'reject' for a likely leak, "
    "'concerns' when you cannot tell, 'ok' when it looks plausible.")

CYCLE_QUESTION = (
    "Review this autopilot cycle: the idea, and where present the code change and the measured result. "
    "Is the work sound, does it break a rule, and is the measured result trustworthy?")


def rules_text(shared_prompt: Path = SHARED_PROMPT) -> str:
    """The 'Purpose' section of the autopilot's shared prompt, so both models read the same rules."""
    text = shared_prompt.read_text() if shared_prompt.exists() else ""
    start = text.find("## Purpose")
    if start == -1:
        return "(rules not found)"
    end = text.find("\n## ", start + 1)
    return text[start + len("## Purpose"):end if end != -1 else None].strip()


def system_prompt() -> str:
    return SYSTEM_TEMPLATE.format(rules=rules_text())


def ask_second_opinion(problem: str, question: str, material: Dict[str, Any], *,
                       outputs_root: Optional[Path] = None, backends: Optional[Sequence[str]] = None,
                       timeout: Optional[int] = None) -> Dict[str, Any]:
    """Ask the reviewer; returns its answer plus which backend and model gave it.

    Raises ``src.genML.llm.LLMError`` when no backend answers; callers decide what that means."""
    from src.genML.llm import ask

    prompt = f"{question}\n\n## Material\n\n```json\n{json.dumps(material, indent=1, default=str)}\n```"
    result = ask(system_prompt(), prompt, SCHEMA, purpose="second-opinion", problem=problem,
                 backends=list(backends or ORDER), outputs_root=outputs_root, timeout=timeout)
    return {**result.data, "backend": result.backend, "model": result.model, "log_file": result.log_file}


# --------------------------------------------------------------------------- #
# Material
# --------------------------------------------------------------------------- #

def measurements(dossier: Dict[str, Any]) -> Dict[str, Any]:
    """The few dossier facts a reviewer needs: the problem, the best run, the noise bar."""
    run = dossier.get("reviewed_run") or {}
    measured = dossier.get("measurements") or {}
    return {
        "problem": dossier.get("problem"),
        "best_run": {"run_id": run.get("run_id"), "final": run.get("final"),
                     "models": {name: {"oof_score": m.get("oof_score")} for name, m in (run.get("models") or {}).items()}},
        "seed_noise": measured.get("seed_noise"),
        "judge_bar": measured.get("judge_bar"),
        "acceptance_rule": measured.get("acceptance_rule"),
    }


def plan_material(idea: Dict[str, Any], *, notes: Optional[str], history: List[Dict[str, Any]],
                  ledger: List[Dict[str, Any]], dossier: Dict[str, Any],
                  brief_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    material = {
        "idea": idea,
        "problem_notes": (notes or "")[:MAX_NOTES_CHARS],
        "earlier_cycles": history[-25:],
        "measured_ideas": [{k: e.get(k) for k in ("title", "status", "predicted_gain", "measured_gain", "changes")}
                           for e in ledger[-25:]],
        "measurements": measurements(dossier),
    }
    if brief_items:
        # The domain brief items the plan builds on: cited research older than the problem, which the
        # rules allow as general domain knowledge (research.py checked their sources and dates).
        material["cited_brief_items"] = brief_items
    return material


def gain_material(idea: Dict[str, Any], judge: Dict[str, Any], *, build: Optional[Dict[str, Any]],
                  diff: str, changes: Optional[List[str]]) -> Dict[str, Any]:
    return {
        "idea": idea,
        "settings_changes": changes,
        "build_report": {k: (build or {}).get(k) for k in ("summary", "leak_test", "smoke", "notes")},
        "judge": {k: judge.get(k) for k in ("verdict", "gain", "threshold", "folds_won", "fold_diffs",
                                             "candidate_score", "best_score", "suspicious_gain", "models", "metric")},
        "code_diff": diff[:MAX_DIFF_CHARS] + ("\n... (diff cut)" if len(diff) > MAX_DIFF_CHARS else ""),
    }


def format_opinion(opinion: Dict[str, Any]) -> str:
    lines = [f"Second opinion ({opinion.get('backend', '-')} / {opinion.get('model', '-')}): "
             f"{opinion['verdict'].upper()}", "", opinion.get("summary", "")]
    for concern in opinion.get("concerns") or []:
        lines.append(f"  - [{concern['kind']}] {concern['detail']}")
    if opinion.get("suggestion"):
        lines += ["", f"Suggestion: {opinion['suggestion']}"]
    if opinion.get("log_file"):
        lines += ["", f"Full call log: {opinion['log_file']}"]
    return "\n".join(lines)


__all__ = ["CONCERN_KINDS", "CYCLE_QUESTION", "GAIN_QUESTION", "ORDER", "PLAN_QUESTION", "SCHEMA", "VERDICTS",
           "ask_second_opinion", "format_opinion", "gain_material", "measurements", "plan_material", "rules_text",
           "system_prompt"]
