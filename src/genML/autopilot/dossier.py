"""The dossier: what the pipeline measured about a problem, as plain data for the planners.

``build_dossier`` collects a run's scores, paired comparisons, seed noise, tuned values at the
edges of their search ranges, how similar the models' predictions are, the run history, the
experiment ledger and the leaderboard. ``SPEC_CAPABILITIES`` lists the settings a plan can change
without new code. The autopilot writes the dossier into each cycle's context folder; the second
opinion reads parts of it. Building it changes nothing on disk.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# What a settings idea can change today (merged into the best run's settings). Anything else is a
# code idea: the build step implements it.
SPEC_CAPABILITIES: Dict[str, Any] = {
    "models": "list from: XGBoost, LightGBM, GLM-LightGBM (binary only: LightGBM trees boosted from a "
              "logistic regression's logit on the target-encoded features; tuned like LightGBM), CatBoost, "
              "Logistic Regression / Linear Regression, Random Forest, TabNet, "
              "RealMLP (MLP with robust scaling and numeric embeddings; strong on medium and large data), "
              "TabM (an efficient ensemble of MLPs, early-stopped), "
              "TabPFN (pretrained foundation model, no training; best under about 10k-50k rows, uses a "
              "max_context sample of larger folds). The three are not Optuna-tuned; fix their parameters "
              "with budget.model_params and check models_available for what can run here",
    "budget": {
        "n_trials": "Optuna trials per model",
        "timeout_per_model": "seconds of tuning per model",
        "tuning_folds": "how many of the shared folds tuning uses (default 2 above 200k rows)",
        "seeds": "N: average each model over N seeds in the OOF stage",
        "tuning": "false: use default parameters",
        "refit_full": "true: refit the best single model on all rows for the submission",
        "oof_models": "N: only the N best tuned models get OOF fits",
        "max_trees": "tree-count ceiling for early stopping during tuning (default 6000; 100..30000)",
        "search_space": "{model: {param: [low, high]}} moves a GBDT model's Optuna range "
                        "(see search_space_defaults; type and log scale stay)",
        "model_params": "{RealMLP|TabM|TabPFN: {param: value}} fixed parameters. RealMLP: n_epochs "
                        "(default by size: 16 at 500k+ rows, up to 256 on small data), batch_size, lr, "
                        "hidden_sizes, p_drop, wd, val_fraction. TabM: n_epochs, patience, batch_size, lr, "
                        "k, d_block, n_blocks, dropout, val_fraction. TabPFN: max_context (default 10000), "
                        "context_bags (default 1), n_estimators (default 4), softmax_temperature. "
                        "A neural model costs minutes per fold on GPU; seeds multiply that",
    },
    "feature_engineering": {
        "base": "raw (original columns) or auto (automated scaled/log/binned copies)",
        "encodings": {
            "target_encode": "auto | list of columns (nested per fold)",
            "count_encode": "auto | list | false",
            "pairs": "auto | [[a, b], ...] | false (crosses, target and count encoded)",
            "numeric_as_categorical": "auto (numerics with <= max_unique values) | all (every numeric "
                                      "column: per-value target encoding) | list",
            "digits": "auto (numeric columns with decimals) | list | false: digit keys, target-encoded "
                      "per fold like categoricals",
            "digit_keys": "subset of [int, int10, tenths, cents, lead1, lead2] (default all): integer part, "
                          "integer part of value/10, value to 1 decimal, cents, leading 1 or 2 digits with "
                          "the digit count",
            "max_unique": "int", "max_pairs": "int", "smoothing": "float",
            "original": "{path, target}: per-value target means from an original dataset file",
            "nested": "true (default) | false",
        },
        "derived": "{name: expression}: numeric columns from arithmetic on columns, e.g. "
                   "ratio: 'amount / balance'; functions floor, ceil, round, abs, sqrt, log, log1p, exp, "
                   "sign, isnull, fillna, minimum, maximum; col('odd name') for awkward names",
        "keys": "{name: [col, col, ...]}: composite key (values joined) as a new categorical column, "
                "e.g. store and product",
        "group_stats": "[{by: key or [cols], cols: [numeric cols], stats: [count, mean, std, min, max, nunique, "
                       "deviation]}]: per-group statistics without labels; past-only (each row sees only earlier "
                       "rows of its group) when time_column is set; past_only: false must be asked for and is not "
                       "available in live use",
        "time_column": "column giving each row's time (makes group_stats past-only)",
        "drift_filter": "{threshold: 0.8, sample: 50000, keep: [cols]}: drop columns that alone tell train "
                        "from test (adversarial AUC above threshold)",
    },
    "ensemble": {"method": "stack | hill_climb | none", "top": "int", "strength": "stacker C"},
    "not_configurable_yet": [
        "Optuna tuning of RealMLP / TabM / TabPFN", "changing cv (folds are fixed per problem in problem.yaml)",
        "pseudo-labeling",
        "seed bagging of target encodings (different inner folds)", "hand-picked pair lists for the GLM only",
    ],
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _prediction_agreement(store, experiment_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Rank correlation between models' OOF predictions (near 1.0 = they learned the same thing)."""
    import pandas as pd

    columns = {}
    for experiment_id in experiment_ids:
        oof = store.get(experiment_id).oof
        columns[experiment_id.split("__", 1)[-1]] = oof[:, -1] if oof.ndim == 2 else oof
    if len(columns) < 2:
        return []
    ranks = pd.DataFrame(columns).rank()
    corr = ranks.corr()
    names = list(columns)
    return [{"a": a, "b": b, "rank_corr": round(float(corr.loc[a, b]), 5)}
            for i, a in enumerate(names) for b in names[i + 1:]]


def _submissions(problem_dir: Path) -> List[Dict[str, Any]]:
    path = problem_dir / "submissions.csv"
    if not path.exists():
        return []
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    keep = ("submitted_at", "run_id", "cv_score", "public_score", "private_score", "best_model")
    return [{k: row.get(k) for k in keep} for row in rows]


def build_dossier(problem_name: str, run_id: Optional[str] = None,
                  outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Everything measured about a problem that a planner needs, as plain data (reads only)."""
    from src.genML.pipeline.diagnostics import compare_experiments, latest_sigma_seed, range_edges
    from src.genML.pipeline.oof import ExperimentStore
    from src.genML.pipeline.problem import PROBLEMS_DIR
    from src.genML.run_context import OUTPUTS_ROOT, list_runs

    from src.genML.experiments import experiment_log

    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    problem_dir = root / problem_name
    runs = list_runs(problem_name, outputs_root=root)
    completed = [r for r in runs if r.get("status") == "completed"]
    if run_id is None:
        if not completed:
            raise FileNotFoundError(f"No completed run of '{problem_name}' to review")
        run_id = completed[0]["run_id"]
    run_dir = problem_dir / run_id
    reports = run_dir / "reports"
    training = _read_json(reports / "model_training_report.json") or {}
    final = _read_json(reports / "final_predictions.json") or {}
    features = _read_json(reports / "feature_engineering_report.json") or {}
    resolved = _read_json(reports / "resolved_spec.json") or {}
    run_meta = _read_json(run_dir / "run.json") or {}
    store = ExperimentStore(problem_dir)

    models, experiment_ids, edges = {}, [], []
    for name, result in (training.get("all_results") or {}).items():
        entry = {"oof_score": result.get("oof_score"), "tuned": result.get("tuned"),
                 "tuning_score": result.get("mean_score")}
        if result.get("best_params"):
            entry["params"] = result["best_params"]
            edges.extend(range_edges(name, result["best_params"], budget=resolved.get("budget")))
        if result.get("members"):
            entry["members"] = result["members"]
        models[name] = entry
        if result.get("experiment_id") and not name.startswith("Ensemble"):
            experiment_ids.append(result["experiment_id"])

    comparisons = []
    final_id = final.get("final_experiment_id")
    best_single = max(
        (eid for eid in experiment_ids), default=None,
        key=lambda eid: store.get(eid).meta.get("oof_score") or float("-inf"))
    pairs = []
    if final_id and best_single and final_id != best_single:
        pairs.append((best_single, final_id, "final blend vs best single model (same run)"))
    from src.genML.experiments import excluded_runs

    excluded = excluded_runs(problem_name)
    previous = next((r for r in completed if r["run_id"] != run_id and r["run_id"] not in excluded), None)
    if previous and final_id:
        prev_final = (_read_json(problem_dir / previous["run_id"] / "reports" / "final_predictions.json")
                      or {}).get("final_experiment_id")
        if prev_final:
            pairs.append((prev_final, final_id, f"this run vs previous run {previous['run_id']}"))
    for a, b, label in pairs:
        try:
            result = compare_experiments(problem_name, a, b, outputs_root=root, n_boot=100)
            comparisons.append({"label": label, "gain": result["diff"], "threshold": result["threshold"],
                                "folds_won": f"{result['folds_won']}/{result['n_folds']}",
                                "real_improvement": result["accept"]})
        except (FileNotFoundError, ValueError, KeyError) as exc:
            comparisons.append({"label": label, "error": str(exc)})

    target_rate = None
    signature = final.get("fold_signature")
    if signature and resolved.get("task") == "binary":
        y, _ = store.load_targets(signature)
        target_rate = round(float(np.mean(y)), 5)

    notes_path = PROBLEMS_DIR / problem_name / "notes.md"
    spec_path = PROBLEMS_DIR / problem_name / "problem.yaml"
    noise = _read_json(reports / "noise_test.json")
    sigma = latest_sigma_seed(problem_name, root)
    encodings = features.get("encodings") or {}
    return {
        "problem": {
            "name": problem_name,
            "description": resolved.get("description"),
            "task": resolved.get("task"),
            "metric": resolved.get("metric"),
            "target": resolved.get("target"),
            "train_rows": run_meta.get("train_rows"),
            "positive_rate": target_rate,
        },
        "problem_yaml": spec_path.read_text() if spec_path.exists() else None,
        "experiment_base": _experiment_base(problem_name, root),
        "spec_capabilities": SPEC_CAPABILITIES,
        "models_available": _models_available(),
        "search_space_defaults": _search_space_defaults(),
        "problem_notes": notes_path.read_text() if notes_path.exists() else None,
        "run_history": [{**{k: r.get(k) for k in ("run_id", "status", "cv_score", "best_model", "notes")},
                         **({"excluded": excluded[r["run_id"]]} if r["run_id"] in excluded else {})}
                        for r in runs],
        "reviewed_run": {
            "run_id": run_id,
            "duration_minutes": _duration_minutes(run_meta),
            "n_features": run_meta.get("n_features"),
            "feature_method": features.get("feature_engineering_method"),
            "nested_target_encoding": features.get("nested_target_encoding"),
            "encodings": {k: encodings.get(k) for k in (
                "target_encoded", "count_encoded", "numeric_as_categorical", "original_mean_encoded")},
            "n_digit_keys": len(encodings.get("digit_keys") or []),
            "n_pairs": len(encodings.get("pairs") or []),
            "final": {k: final.get(k) for k in ("final_name", "final_score", "ensemble")},
            "models": models,
        },
        "measurements": {
            "comparisons": comparisons,
            "seed_noise": {k: noise.get(k) for k in ("model", "sigma_seed", "se_rows", "threshold")}
            if noise else ({"sigma_seed": sigma} if sigma is not None else None),
            "acceptance_rule": ("A change is a real improvement only if its paired gain exceeds "
                                "3 x sqrt(row_se^2 + 2 x sigma_seed^2) and it wins at least 80% of the folds."),
            "params_at_search_edges": edges,
            "prediction_agreement": _prediction_agreement(store, experiment_ids),
        },
        "experiment_log": experiment_log(problem_name, root),
        "leaderboard": _submissions(problem_dir),
    }


def _experiment_base(problem_name: str, root: Path) -> Dict[str, Any]:
    """The settings a spec_change patch is merged into (the best run's), and where they come from."""
    from src.genML.experiments import baseline_settings, current_best
    from src.genML.pipeline.problem import load_problem

    best = current_best(problem_name, root)
    try:
        spec = load_problem(problem_name)
    except Exception:  # the dossier must not fail on a spec problem
        spec = None
    if spec is None and best is None:
        return {}
    try:
        settings, source = baseline_settings(problem_name, best, spec, root)
    except Exception:
        return {}
    return {"source": source, "best_score": best["score"] if best else None, "settings": settings}


def _models_available() -> Dict[str, bool]:
    """Which optional models can run on this machine (installed, and weights usable)."""
    from src.genML.pipeline.neural import available_models

    return available_models()


def _search_space_defaults() -> Dict[str, Any]:
    from src.genML.pipeline.tuning import GBDT_KINDS, GBDT_SPACE, MAX_TREES

    out: Dict[str, Any] = {"max_trees": MAX_TREES}
    for model, kind in GBDT_KINDS.items():
        out[model] = {name: [low, high] + (["log"] if log else []) for name, (_, low, high, log) in GBDT_SPACE[kind].items()}
    return out


def _duration_minutes(run_meta: Dict[str, Any]) -> Optional[float]:
    from datetime import datetime

    try:
        start = datetime.fromisoformat(run_meta["started_at"])
        end = datetime.fromisoformat(run_meta["finished_at"])
        return round((end - start).total_seconds() / 60, 1)
    except (KeyError, TypeError, ValueError):
        return None
