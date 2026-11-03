"""Service layer shared by the MCP server (and usable from the API or scripts).

Every function returns a small JSON-serializable summary rather than raw files,
so LLM clients get compact answers; large artifacts are exposed separately as
MCP resources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.genML import jobs
from src.genML.pipeline import problem as problem_module
from src.genML.pipeline.problem import (
    ProblemSpec,
    find_sample_submission,
    list_problems as _list_problem_names,
    load_problem,
    locate_data_dir,
    resolve_problem,
    save_problem,
)
from src.genML import run_context
from src.genML.run_context import list_runs as _list_runs

REPORT_SECTIONS = ("summary", "models", "features", "spec", "errors")
SAFE_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


class ServiceError(ValueError):
    """A request that cannot be served; the message says how to fix it."""


def _check_name(value: str, label: str) -> str:
    if not value or not set(value) <= SAFE_NAME_CHARS or value in (".", ".."):
        raise ServiceError(f"Invalid {label} '{value}': use letters, digits, '-', '_' or '.'")
    return value


def _root(outputs_root: Optional[Path]) -> Path:
    return Path(outputs_root or run_context.OUTPUTS_ROOT)


def _run_dir(problem: str, run_id: str, outputs_root: Optional[Path] = None) -> Path:
    path = _root(outputs_root) / _check_name(problem, "problem") / _check_name(run_id, "run_id")
    if not path.is_dir():
        raise ServiceError(f"No run {run_id} for problem {problem}; use list_runs to see run ids")
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _score(value) -> Optional[float]:
    return round(float(value), 6) if isinstance(value, (int, float)) else None


# --------------------------------------------------------------------------- #
# Problems
# --------------------------------------------------------------------------- #

def list_problems(outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    rows = []
    for name in _list_problem_names():
        spec = load_problem(name)
        runs = [r for r in _list_runs(name, outputs_root=_root(outputs_root)) if r.get("status") == "completed"]
        best = None
        if runs and spec.metric:
            from src.genML.pipeline.metrics import get_metric

            metric = get_metric(spec.metric)
            scored = [r for r in runs if isinstance(r.get("cv_score"), (int, float))]
            if scored:
                top = max(scored, key=lambda r: metric.signed(r["cv_score"]))
                best = {"run_id": top["run_id"], "cv_score": _score(top["cv_score"])}
        rows.append({
            "name": name,
            "description": spec.description,
            "task": spec.task,
            "metric": spec.metric,
            "competition": spec.source.get("competition"),
            "completed_runs": len(runs),
            "best_run": best,
        })
    return rows


def inspect_dataset(problem: Optional[str] = None, data_path: Optional[str] = None,
                    nrows: int = 50_000) -> Dict[str, Any]:
    """Shape, column types and what inference would decide for a dataset."""
    if problem:
        spec = load_problem(problem)
    elif data_path:
        spec = ProblemSpec(name="inspect", source={"type": "csv", "path": data_path})
    else:
        raise ServiceError("Pass a declared problem name or a data_path folder with train.csv/test.csv")
    data_dir = locate_data_dir(spec)
    train = pd.read_csv(data_dir / "train.csv", nrows=nrows)
    test = pd.read_csv(data_dir / "test.csv", nrows=min(nrows, 5000))
    sample_path = find_sample_submission(data_dir)
    sample = pd.read_csv(sample_path, nrows=5000) if sample_path else None
    resolved = resolve_problem(spec, train, test, sample, data_dir=data_dir)

    from src.genML.features.data_analyzer import DataTypeAnalyzer

    analysis = DataTypeAnalyzer().analyze_dataset(train.drop(columns=[resolved.target]))
    columns = []
    for column in train.columns:
        info = analysis["column_types"].get(column, {})
        columns.append({
            "name": column,
            "dtype": str(train[column].dtype),
            "unique": int(train[column].nunique()),
            "missing_pct": round(float(train[column].isna().mean()) * 100, 2),
            "detected_type": "target" if column == resolved.target else info.get("detected_type"),
        })
    with open(data_dir / "train.csv") as handle:
        total_rows = sum(1 for _ in handle) - 1
    return {
        "data_dir": str(data_dir),
        "train_rows": total_rows,
        "test_rows_sampled": len(test),
        "columns": columns,
        "resolved": {
            "target": resolved.target, "id_column": resolved.id_column, "task": resolved.task,
            "metric": resolved.metric, "submission_output": resolved.output,
            "submission_columns": resolved.submission.get("target_columns"),
        },
        "inferred_fields": resolved.inferred,
        "target_distribution": (
            {str(k): round(float(v), 4) for k, v in train[resolved.target].value_counts(normalize=True).head(10).items()}
            if resolved.is_classification else
            {k: round(float(v), 4) for k, v in train[resolved.target].describe().items()}
        ),
    }


def create_problem(name: str, data_path: Optional[str] = None, kaggle_competition: Optional[str] = None,
                   target: Optional[str] = None, id_column: Optional[str] = None, task: Optional[str] = None,
                   metric: Optional[str] = None, description: str = "", overwrite: bool = False) -> Dict[str, Any]:
    """Write problems/<name>/problem.yaml after validating it against the data."""
    _check_name(name, "problem name")
    path = problem_module.PROBLEMS_DIR / name / "problem.yaml"
    if path.exists() and not overwrite:
        raise ServiceError(f"Problem '{name}' already exists; pass overwrite=true to replace it")

    if kaggle_competition:
        from src.genML.pipeline.kaggle import init_problem_from_kaggle

        written, resolved = init_problem_from_kaggle(kaggle_competition, name=name)
        spec = load_problem(name)
        for key, value in (("target", target), ("id_column", id_column), ("task", task), ("metric", metric)):
            if value is not None:
                setattr(spec, key, value)
        if description:
            spec.description = description
        spec.validate()
        save_problem(spec)
    else:
        if not data_path:
            raise ServiceError("Pass data_path (a folder with train.csv and test.csv) or kaggle_competition")
        spec = ProblemSpec(name=name, source={"type": "csv", "path": data_path}, target=target,
                           id_column=id_column, task=task, metric=metric, description=description)
        spec.validate()
        save_problem(spec)

    inspection = inspect_dataset(problem=name)
    spec = load_problem(name)
    # Persist the resolved essentials so later runs do not depend on inference
    resolved = inspection["resolved"]
    spec.target, spec.id_column, spec.task, spec.metric = (
        resolved["target"], resolved["id_column"], resolved["task"], resolved["metric"])
    save_problem(spec)
    return {"problem": name, "spec_file": str(path), "resolved": resolved,
            "inferred_fields": inspection["inferred_fields"]}


# --------------------------------------------------------------------------- #
# Runs and jobs
# --------------------------------------------------------------------------- #

def start_run(problem: str, models: Optional[List[str]] = None, n_trials: Optional[int] = None,
              timeout_per_model: Optional[int] = None, seeds: Optional[int] = None,
              encodings: Optional[Dict[str, Any]] = None, ensemble_method: Optional[str] = None,
              notes: str = "", outputs_root: Optional[Path] = None, start_worker: bool = True) -> Dict[str, Any]:
    """Queue a pipeline run with optional overrides of the problem spec."""
    load_problem(_check_name(problem, "problem"))  # fail fast on unknown problems
    budget = {key: value for key, value in (("n_trials", n_trials), ("timeout_per_model", timeout_per_model),
                                            ("seeds", seeds)) if value is not None}
    options: Dict[str, Any] = {}
    if models:
        options["models"] = models
    if budget:
        options["budget"] = budget
    if encodings is not None:
        options["feature_engineering"] = {"encodings": encodings}
    if ensemble_method:
        if ensemble_method not in ("stack", "hill_climb", "none"):
            raise ServiceError("ensemble_method must be stack, hill_climb or none")
        options["ensemble"] = {"method": ensemble_method}
    job = jobs.enqueue(problem, "run", options, notes, outputs_root=outputs_root)
    worker_started = jobs.ensure_worker(outputs_root) if start_worker else False
    return {"job_id": job["id"], "run_id": job["run_id"], "status": job["status"],
            "queue_position": job.get("queue_position", 0), "worker_started": worker_started,
            "options": options}


def start_ensemble(problem: str, method: str = "stack", top: int = 10, notes: str = "",
                   outputs_root: Optional[Path] = None, start_worker: bool = True) -> Dict[str, Any]:
    """Queue a blend of the best stored experiments across runs."""
    load_problem(_check_name(problem, "problem"))
    job = jobs.enqueue(problem, "ensemble", {"method": method, "top": top}, notes, outputs_root=outputs_root)
    worker_started = jobs.ensure_worker(outputs_root) if start_worker else False
    return {"job_id": job["id"], "run_id": job["run_id"], "status": job["status"],
            "worker_started": worker_started}


def _live_progress(run_id: str, outputs_root: Optional[Path]) -> Optional[Dict[str, Any]]:
    progress = _read_json(_root(outputs_root) / "progress" / "current_run.json")
    if progress.get("run_id") != run_id:
        return None
    models = progress.get("models") or []
    training = next((m for m in reversed(models) if m.get("status") == "training"), None)
    return {
        "stage": f"{progress.get('current_stage')}/5 {progress.get('current_stage_name')}",
        "stage_progress_pct": progress.get("stage_progress_pct"),
        "current_task": progress.get("current_task"),
        "current_model": (training or {}).get("name"),
        "current_trial": (training or {}).get("current_trial"),
        "models_done": [{"name": m.get("name"), "score": _score(m.get("mean_score"))}
                        for m in models if m.get("status") == "completed"],
        "elapsed_seconds": (progress.get("resources") or {}).get("elapsed_seconds"),
    }


def get_run_status(job_id: int, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    try:
        job = jobs.get_job(int(job_id), outputs_root)
    except KeyError as exc:
        raise ServiceError(f"{exc}; use list_jobs to see job ids") from exc
    status = {key: job.get(key) for key in ("id", "kind", "problem", "status", "run_id", "notes",
                                            "created_at", "started_at", "finished_at", "error",
                                            "queue_position")}
    if job["status"] == "running":
        status["progress"] = _live_progress(job["run_id"], outputs_root)
    summary = _read_json(_root(outputs_root) / job["problem"] / job["run_id"] / "run.json")
    if summary:
        status["result"] = {"cv_score": _score(summary.get("cv_score")), "metric": summary.get("metric"),
                            "best_model": summary.get("best_model"), "submission": summary.get("submission")}
    return status


def list_jobs(limit: int = 10, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    return [{key: job.get(key) for key in ("id", "kind", "problem", "status", "run_id", "notes", "created_at")}
            for job in jobs.list_jobs(limit=limit, outputs_root=outputs_root)]


def cancel_run(job_id: int, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    job = jobs.cancel_job(int(job_id), outputs_root)
    return {"id": job["id"], "status": job["status"]}


def list_runs(problem: str, limit: int = 10, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    runs = _list_runs(_check_name(problem, "problem"), outputs_root=_root(outputs_root))[:limit]
    return [{"run_id": r["run_id"], "status": r.get("status"), "metric": r.get("metric"),
             "cv_score": _score(r.get("cv_score")), "best_model": r.get("best_model"),
             "notes": r.get("notes"), "finished_at": r.get("finished_at")} for r in runs]


def compare_runs(problem: str, run_ids: Optional[List[str]] = None, limit: int = 10,
                 outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Runs side by side with each model's OOF score and the gap to the best run."""
    from src.genML.pipeline.metrics import get_metric

    runs = _list_runs(_check_name(problem, "problem"), outputs_root=_root(outputs_root))
    if run_ids:
        runs = [r for r in runs if r["run_id"] in set(run_ids)]
    runs = [r for r in runs if r.get("status") == "completed"][:limit]
    if not runs:
        return {"problem": problem, "runs": []}
    metric = get_metric(runs[0]["metric"])
    best = max(metric.signed(r["cv_score"]) for r in runs if isinstance(r.get("cv_score"), (int, float)))
    rows = []
    for run in runs:
        final = _read_json(_root(outputs_root) / problem / run["run_id"] / "reports" / "final_predictions.json")
        score = run.get("cv_score")
        rows.append({
            "run_id": run["run_id"], "notes": run.get("notes"), "cv_score": _score(score),
            "gap_to_best": _score(best - metric.signed(score)) if isinstance(score, (int, float)) else None,
            "final": run.get("best_model"),
            "model_oof_scores": {k: _score(v) for k, v in (final.get("single_scores") or {}).items()},
            "n_features": run.get("n_features"),
        })
    return {"problem": problem, "metric": metric.name, "greater_is_better": metric.greater_is_better,
            "runs": rows}


def get_run_report(problem: str, run_id: str, section: str = "summary",
                   outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    if section not in REPORT_SECTIONS:
        raise ServiceError(f"section must be one of {REPORT_SECTIONS}")
    run_dir = _run_dir(problem, run_id, outputs_root)
    reports = run_dir / "reports"
    if section == "summary":
        return _read_json(run_dir / "run.json")
    if section == "models":
        training = _read_json(reports / "model_training_report.json")
        return {
            "metric": training.get("scoring_metric"),
            "final": training.get("final_predictions"),
            "models": {name: {k: v for k, v in result.items() if k in ("mean_score", "std_score", "oof_score",
                                                                       "tuned", "best_params", "members")}
                       for name, result in (training.get("all_results") or {}).items()},
        }
    if section == "features":
        features = _read_json(reports / "feature_engineering_report.json")
        names = features.get("features_used") or []
        return {"n_features": len(names), "features": names[:200], "truncated": len(names) > 200,
                "encodings": features.get("encodings"), "detected_domains": features.get("detected_domains"),
                "target_transformation": features.get("target_transformation")}
    if section == "spec":
        return _read_json(reports / "resolved_spec.json")
    summary = _read_json(run_dir / "run.json")
    log = run_dir / "pipeline.log"
    tail = log.read_text(errors="replace").splitlines()[-60:] if log.exists() else []
    return {"status": summary.get("status"), "error": summary.get("error"), "log_tail": tail}


def get_feature_diagnostics(problem: str, run_id: str, max_rows: int = 100_000,
                            outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Per-column signal on the raw training data, to guide feature ideas.

    Binary: ROC AUC of each numeric column (0.5 = no signal, symmetric) and the
    spread of target rates across a categorical's values. Regression and
    multiclass: Spearman correlation with the target (multiclass: |corr| with
    each class indicator, max over classes).
    """
    run_dir = _run_dir(problem, run_id, outputs_root)
    resolved = _read_json(run_dir / "reports" / "resolved_spec.json")
    train = pd.read_pickle(run_dir / "data" / "train_data.pkl")
    if len(train) > max_rows:
        train = train.sample(max_rows, random_state=0)
    target, task = resolved["target"], resolved["task"]
    y = train[target]
    rows = []
    for column in train.columns:
        if column in (target, resolved.get("id_column")):
            continue
        series = train[column]
        entry = {"column": column, "unique": int(series.nunique()),
                 "missing_pct": round(float(series.isna().mean()) * 100, 2)}
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if task == "binary":
            positive = (y == sorted(y.unique())[-1]).astype(int)
            if numeric:
                from sklearn.metrics import roc_auc_score

                filled = series.fillna(series.median())
                auc = roc_auc_score(positive, filled) if filled.nunique() > 1 else 0.5
                entry["signal"] = round(abs(auc - 0.5) * 2, 4)
                entry["auc"] = round(float(auc), 4)
            else:
                rates = positive.groupby(series.astype(str)).mean()
                entry["signal"] = round(float(rates.max() - rates.min()), 4)
                entry["top_values"] = {k: round(float(v), 4) for k, v in rates.sort_values().tail(3).items()}
        else:
            codes = series if numeric else series.astype("category").cat.codes
            if task == "multiclass":
                corr = max(abs(pd.Series(codes).corr((y == c).astype(int), method="spearman") or 0)
                           for c in y.unique())
            else:
                corr = abs(pd.Series(codes).corr(y, method="spearman") or 0)
            entry["signal"] = round(float(np.nan_to_num(corr)), 4)
        rows.append(entry)
    rows.sort(key=lambda r: r.get("signal", 0), reverse=True)
    features = get_run_report(problem, run_id, "features", outputs_root)
    return {"task": task, "rows_used": len(train), "columns": rows,
            "engineered_features": features["n_features"], "encodings": features["encodings"],
            "how_to_use": "Columns with high signal are candidates for interactions, target encoding or "
                          "binning; pass encodings or models to start_run to test an idea."}


# --------------------------------------------------------------------------- #
# Kaggle
# --------------------------------------------------------------------------- #

def submit_to_kaggle(problem: str, run_id: Optional[str] = None, message: str = "", confirm: bool = False,
                     outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """Submit a run to Kaggle. Without confirm=true this only previews the submission."""
    from src.genML.submissions import read_log, submit_run

    runs = [r for r in _list_runs(_check_name(problem, "problem"), outputs_root=_root(outputs_root))
            if r.get("status") == "completed"]
    if run_id:
        runs = [r for r in runs if r["run_id"] == run_id]
    if not runs:
        raise ServiceError("No completed run to submit; check list_runs")
    run = runs[0]
    previous = read_log(problem, outputs_root)
    preview = {"problem": problem, "run_id": run["run_id"], "cv_score": _score(run.get("cv_score")),
               "metric": run.get("metric"), "submission": run.get("submission"),
               "submitted_before": len(previous)}
    if not confirm:
        return {**preview, "submitted": False,
                "note": "Preview only. Submissions count against the daily limit; call again with "
                        "confirm=true to submit."}
    row = submit_run(problem, run_id=run["run_id"], message=message, outputs_root=outputs_root)
    return {**preview, "submitted": True, "kaggle_response": row.get("kaggle_response")}


def get_submissions(problem: str, refresh: bool = True, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    from src.genML.submissions import read_log, refresh_scores

    rows = refresh_scores(problem, outputs_root) if refresh else read_log(problem, outputs_root)
    return [{k: row.get(k) for k in ("submitted_at", "run_id", "cv_score", "public_score", "private_score",
                                     "status", "best_model")} for row in rows]
