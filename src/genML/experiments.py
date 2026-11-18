"""Settings checks and the experiment ledger for the autopilot.

* ``check_patch`` decides what a settings change may touch: only ``models``, ``budget``,
  ``feature_engineering`` and ``ensemble``, with known keys and sane values, never the folds
  (they stay shared so runs can be compared row by row).
* ``baseline_settings`` / ``deep_merge`` / ``fingerprint`` / ``_changes``: a change is merged into
  the settings of the best run so far (its ``resolved_spec.json``), so accepted changes carry forward
  without editing problem.yaml, and the same settings are never measured twice.
* ``record_entry`` / ``load_ledger`` / ``experiment_log``: ``outputs/<problem>/experiments.json``,
  one entry per measured idea (predicted and measured gain), which the next plan reads.
* ``current_best``: the completed run with the best final out-of-fold score.

Agents propose; the loop and the judge (``autopilot/``) check, run and measure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

LEDGER = "experiments.json"
EXCLUDED_FILE = "excluded_runs.yaml"      # problems/<name>/: runs that do not count, with the reason

KNOWN_MODELS = ["XGBoost", "LightGBM", "GLM-LightGBM", "CatBoost", "Logistic Regression",
                "Linear Regression", "Random Forest", "TabNet", "RealMLP", "TabM", "TabPFN"]
ENCODING_KEYS = {"target_encode", "count_encode", "pairs", "numeric_as_categorical", "max_unique",
                 "max_pairs", "smoothing", "original", "nested", "digits", "digit_keys"}
# key -> (type check, allowed values or (min, max))
BUDGET_RULES = {
    "n_trials": (int, (1, 200)),
    "timeout_per_model": (int, (30, 3600)),
    "tuning_folds": (int, (1, 10)),
    "seeds": (int, (1, 10)),
    "oof_models": (int, (1, 20)),
    "tuning": (bool, None),
    "refit_full": (bool, None),
    "max_trees": (int, (100, 30_000)),
    "search_space": (dict, None),  # checked by tuning.check_search_space
    "model_params": (dict, None),  # checked by neural.check_model_params
}
ENSEMBLE_RULES = {
    "method": (str, ("stack", "hill_climb", "none")),
    "top": (int, (1, 50)),
    "strength": ((int, float), (1e-4, 1e4)),
}
SECTIONS = ("models", "budget", "feature_engineering", "ensemble")
DERIVED_KEYS = ("derived", "keys", "group_stats", "drift_filter", "time_column")  # features/derived.py


_UNAVAILABLE = {
    "RealMLP": "needs torch and pytabkit (pip install genml[nn])",
    "TabM": "needs torch and pytabkit (pip install genml[nn])",
    "TabPFN": "needs the tabpfn package and its weights, which download only after the owner accepts "
              "Prior Labs' license and sets TABPFN_TOKEN",
}


# --------------------------------------------------------------------------- #
# Checking a patch
# --------------------------------------------------------------------------- #

def _check_value(path: str, value: Any, rule) -> List[str]:
    kind, allowed = rule
    if kind is int and isinstance(value, bool) or not isinstance(value, kind):
        return [f"{path} must be {getattr(kind, '__name__', 'a number')}, got {value!r}"]
    if allowed is None:
        return []
    if isinstance(allowed, tuple) and len(allowed) == 2 and all(isinstance(a, (int, float)) for a in allowed):
        if not allowed[0] <= value <= allowed[1]:
            return [f"{path}={value} is outside {allowed[0]}..{allowed[1]}"]
        return []
    return [] if value in allowed else [f"{path}={value!r} must be one of {list(allowed)}"]


def check_patch(patch: Any, target: Optional[str] = None, known_only: bool = True) -> List[str]:
    """Problems with a patch (empty list = acceptable).

    Always refused, because they let labels reach a fold's own model: ``encodings.nested: false``,
    ``base: auto`` (feature selection on every row's label) and, with ``target``, derived features,
    keys or group statistics that read the target. ``known_only=False`` (a code idea, whose build may
    add settings) allows setting and model names this checkout does not know yet; the rest still holds.
    """
    if not isinstance(patch, dict) or not patch:
        return ["patch must be a non-empty mapping"]
    errors = [f"'{key}' cannot be changed by an experiment (allowed: {', '.join(SECTIONS)})"
              for key in patch if key not in SECTIONS]
    unknown = (lambda message: [message]) if known_only else (lambda message: [])
    models = patch.get("models")
    if models is not None:
        if not isinstance(models, list) or not models:
            errors.append("models must be a non-empty list")
        else:
            known = {m.lower() for m in KNOWN_MODELS}
            for m in models:
                if str(m).lower() not in known:
                    errors += unknown(f"unknown model '{m}'")
            from src.genML.pipeline.neural import available_models

            errors += [f"model '{m}' cannot run here: {_UNAVAILABLE.get(m, 'not installed')}"
                       for m, ok in available_models().items() if not ok and m in models]
    for section, rules in (("budget", BUDGET_RULES), ("ensemble", ENSEMBLE_RULES)):
        block = patch.get(section)
        if block is None:
            continue
        if not isinstance(block, dict):
            errors.append(f"{section} must be a mapping")
            continue
        for key, value in block.items():
            if key not in rules:
                errors += unknown(f"{section}.{key} is not a known setting")
            elif section == "budget" and key == "search_space":
                from src.genML.pipeline.tuning import check_search_space

                errors += check_search_space(value)
            elif section == "budget" and key == "model_params":
                from src.genML.pipeline.neural import check_model_params

                errors += check_model_params(value)
            else:
                errors += _check_value(f"{section}.{key}", value, rules[key])
    fe = patch.get("feature_engineering")
    if fe is not None:
        if not isinstance(fe, dict):
            errors.append("feature_engineering must be a mapping")
        else:
            for key, value in fe.items():
                if key == "base":
                    errors += _check_value("feature_engineering.base", value, (str, ("raw", "auto")))
                    if value == "auto":
                        errors.append("feature_engineering.base: auto selects features with every row's label "
                                      "(outside the folds); experiments must use raw")
                elif key == "encodings":
                    if not isinstance(value, dict):
                        errors.append("feature_engineering.encodings must be a mapping")
                        continue
                    for k in value:
                        if k not in ENCODING_KEYS:
                            errors += unknown(f"feature_engineering.encodings.{k} is not a known setting")
                    if value.get("nested") is False:
                        errors.append("feature_engineering.encodings.nested: false would let a fold's labels "
                                      "reach its own model; experiments keep the nested encodings")
                    errors += _check_encoding_values(value)
                    original = value.get("original")
                    if original and not (isinstance(original, dict) and original.get("path")
                                         and original.get("target")):
                        errors.append("encodings.original needs path and target")
                    elif original and not _resolve(original["path"]).exists():
                        errors.append(f"encodings.original.path {original['path']} does not exist")
                elif key in DERIVED_KEYS:
                    continue  # checked together below
                else:
                    errors += unknown(f"feature_engineering.{key} is not a known setting")
            from src.genML.features.derived import check_settings

            errors += [f"feature_engineering.{e}" for e in check_settings(
                {k: v for k, v in fe.items() if k in DERIVED_KEYS}, target)]
    return errors


def leak_problems(settings: Dict[str, Any], target: Optional[str] = None) -> List[str]:
    """The leak rules on complete settings (a patch can finish a chain its baseline started)."""
    fe = settings.get("feature_engineering") or {}
    errors = []
    if fe.get("base", "auto") != "raw":
        errors.append("feature_engineering.base must be raw: auto selects features with every row's label")
    if (fe.get("encodings") or {}).get("nested") is False:
        errors.append("feature_engineering.encodings.nested: false lets a fold's labels reach its own model")
    if target:
        from src.genML.features.derived import label_references

        errors += [f"feature_engineering.{where} reads the target '{target}'"
                   for where in label_references(fe, target)]
    return errors


def _column_option(path: str, value: Any, words: Sequence[Any]) -> List[str]:
    """A setting that takes one of ``words`` or a list of column names."""
    if value in words or (isinstance(value, list) and all(isinstance(v, str) for v in value)):
        return []
    return [f"{path} must be one of {list(words)} or a list of columns, got {value!r}"]


def _check_encoding_values(encodings: Dict[str, Any]) -> List[str]:
    from src.genML.features.encoding import DIGIT_KINDS

    errors = []
    for key in ("target_encode", "count_encode"):
        if key in encodings:
            errors += _column_option(f"encodings.{key}", encodings[key], (True, False, "auto", None))
    if "numeric_as_categorical" in encodings:
        errors += _column_option("encodings.numeric_as_categorical", encodings["numeric_as_categorical"],
                                 (True, False, "auto", "all", None))
    if "digits" in encodings:
        errors += _column_option("encodings.digits", encodings["digits"], (True, False, "auto", None))
    if "digit_keys" in encodings:
        kinds = encodings["digit_keys"]
        if not isinstance(kinds, list) or not kinds or any(k not in DIGIT_KINDS for k in kinds):
            errors.append(f"encodings.digit_keys must be a non-empty list from {list(DIGIT_KINDS)}, got {kinds!r}")
    for key, rule in (("max_unique", (int, (2, 10_000_000))), ("max_pairs", (int, (0, 500))),
                      ("smoothing", ((int, float), (0, 1e6))), ("nested", (bool, None))):
        if key in encodings:
            errors += _check_value(f"encodings.{key}", encodings[key], rule)
    return errors


def _resolve(path: str) -> Path:
    from src.genML.pipeline.problem import PROJECT_ROOT

    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Mappings merge key by key; lists and values replace."""
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _sections(spec) -> Dict[str, Any]:
    return {"models": spec.models, "budget": dict(spec.budget or {}),
            "feature_engineering": dict(spec.feature_engineering or {}), "ensemble": dict(spec.ensemble or {})}


def baseline_settings(problem: str, baseline: Optional[Dict[str, Any]], spec,
                      outputs_root: Optional[Path] = None) -> tuple:
    """Settings an experiment builds on: the baseline run's (so accepted changes carry
    forward without editing problem.yaml), else problem.yaml's. Returns (sections, source)."""
    if baseline:
        path = _root(outputs_root) / problem / baseline["run_id"] / "reports" / "resolved_spec.json"
        try:
            resolved = json.loads(path.read_text())
            return ({"models": resolved.get("models"), "budget": dict(resolved.get("budget") or {}),
                     "feature_engineering": dict(resolved.get("feature_engineering") or {}),
                     "ensemble": dict(resolved.get("ensemble") or {})},
                    f"run {baseline['run_id']}")
        except (OSError, json.JSONDecodeError):
            pass
    return _sections(spec), "problem.yaml"


def fingerprint(sections: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(sections, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _changes(before: Any, after: Any, prefix: str = "") -> List[str]:
    """Human-readable list of settings that differ."""
    if isinstance(before, dict) and isinstance(after, dict):
        lines = []
        for key in sorted(set(before) | set(after)):
            lines += _changes(before.get(key), after.get(key), f"{prefix}{key}.")
        return lines
    if before != after:
        return [f"{prefix.rstrip('.')}: {before!r} -> {after!r}"]
    return []


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

def _root(outputs_root: Optional[Path]) -> Path:
    from src.genML.run_context import OUTPUTS_ROOT

    return Path(outputs_root) if outputs_root else OUTPUTS_ROOT


def load_ledger(problem: str, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = _root(outputs_root) / problem / LEDGER
    return json.loads(path.read_text()) if path.exists() else []


def _save_ledger(problem: str, entries: List[Dict[str, Any]], outputs_root: Optional[Path]) -> None:
    path = _root(outputs_root) / problem / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2, default=str))
    tmp.replace(path)


def record_entry(problem: str, entry: Dict[str, Any], outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Append an already-measured experiment (e.g. an autopilot cycle) to the ledger with the next id."""
    entries = load_ledger(problem, outputs_root)
    entry = {"id": f"E{len(entries) + 1:03d}", **entry}
    entries.append(entry)
    _save_ledger(problem, entries, outputs_root)
    return entry


def excluded_runs(problem: str) -> Dict[str, str]:
    """Runs that do not count (run_id -> reason): ``problems/<name>/excluded_runs.yaml``.

    For runs built on ideas the rules forbid (a problem's published solutions or discussions, other
    people's predictions) or on a known leak. Nothing picks them as a best run, a baseline or a blend
    member. The file sits in a protected path, so agents cannot change it."""
    import yaml

    from src.genML.pipeline import problem as problem_module

    path = Path(problem_module.PROBLEMS_DIR) / problem / EXCLUDED_FILE
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {str(run_id): str(reason) for run_id, reason in data.items()}


def exclude_run(problem: str, run_id: str, reason: str) -> Path:
    """Add a run to ``excluded_runs.yaml`` (the owner's decision; agents cannot)."""
    import yaml

    from src.genML.pipeline import problem as problem_module

    if not reason.strip():
        raise ValueError("say why the run does not count (--reason)")
    path = Path(problem_module.PROBLEMS_DIR) / problem / EXCLUDED_FILE
    entries = excluded_runs(problem)
    entries[run_id] = reason.strip()
    header = ("# Runs that do not count for this problem (run_id: reason). Nothing picks them as a best\n"
              "# run, a baseline or a blend member. See DECISIONS.md.\n")
    path.write_text(header + yaml.safe_dump(entries, sort_keys=True, allow_unicode=True, width=1000))
    return path


def current_best(problem: str, outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The completed, not excluded run with the best final OOF score, and its final experiment."""
    from src.genML.run_context import list_runs

    root = _root(outputs_root)
    best = None
    excluded = excluded_runs(problem)
    for run in list_runs(problem, outputs_root=root):
        if run.get("status") != "completed" or run["run_id"] in excluded:
            continue
        path = root / problem / run["run_id"] / "reports" / "final_predictions.json"
        if not path.exists():
            continue
        final = json.loads(path.read_text())
        if not final.get("final_experiment_id") or final.get("final_score") is None:
            continue
        signed = final["final_score"] if final.get("greater_is_better", True) else -final["final_score"]
        if best is None or signed > best["signed"]:
            best = {"run_id": run["run_id"], "experiment_id": final["final_experiment_id"],
                    "score": final["final_score"], "signed": signed, "fold_signature": final.get("fold_signature")}
    return best


# --------------------------------------------------------------------------- #
# Plan, launch, evaluate
# --------------------------------------------------------------------------- #

def experiment_log(problem: str, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Compact ledger for the dossier: what was tried, predicted and measured."""
    keys = ("id", "status", "title", "source", "changes", "predicted_gain", "probe_gain", "measured_gain",
            "threshold", "folds_won", "score", "same_model", "error")
    excluded = excluded_runs(problem)
    log = []
    for entry in load_ledger(problem, outputs_root):
        row = {k: entry.get(k) for k in keys if entry.get(k) is not None}
        if entry.get("run_id") in excluded:
            row["excluded"] = excluded[entry["run_id"]]
        log.append(row)
    return log
