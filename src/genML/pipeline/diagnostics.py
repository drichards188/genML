"""Is a difference between two experiments real? Measurement tools for the OOF store.

Two sources of noise decide whether a gain in out-of-fold (OOF) score is real:

* Row sampling: a different draw of rows would score differently. Because every
  experiment of a problem is scored on the same rows (shared folds), the paired
  bootstrap of the score *difference* measures this tightly (``paired_compare``).
* Training randomness: the same configuration with another seed scores a little
  differently. The paired bootstrap does not see this; ``seed_noise_test`` refits
  a run's model with several seeds and measures it.

A change counts as an improvement when its paired gain exceeds
``threshold = z * sqrt(se_rows**2 + 2 * sigma_seed**2)`` and it wins on most
folds. ``z = 3`` keeps the chance of a false accept low across a few dozen
adaptively chosen experiments (roughly Bonferroni for 50 comparisons).
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.genML.pipeline.oof import ExperimentStore, FoldFeatures, fit_oof
from src.genML.pipeline.problem import ResolvedProblem
from src.genML.pipeline.targets import inverse_regression

logger = logging.getLogger(__name__)

DEFAULT_Z = 3.0
NOISE_REPORT = "noise_test.json"
NOISE_OOFS = "noise_test_oofs.npy"      # every seed's OOF predictions, so the judge can rescore any rows


def _outputs_root(outputs_root: Optional[Path]) -> Path:
    from src.genML.run_context import OUTPUTS_ROOT

    return Path(outputs_root) if outputs_root else OUTPUTS_ROOT


def _signed_scorer(problem: ResolvedProblem, y: np.ndarray, transform_info: Dict[str, Any]) -> Callable:
    """score(rows, pred) on the metric's own scale, signed so higher is better."""
    metric, task = problem.metric_obj, problem.task
    y_eval = inverse_regression(y.astype(float), transform_info) if task == "regression" else y

    def score(rows: np.ndarray, pred: np.ndarray) -> float:
        values = pred[rows]
        if task == "regression":
            values = inverse_regression(values, transform_info)
        return metric.signed(metric.score(y_eval[rows], values, task))

    return score


def paired_compare(score: Callable, pred_a: np.ndarray, pred_b: np.ndarray, fold_ids: np.ndarray,
                   n_boot: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Paired comparison of two OOF prediction sets on the same rows (b minus a).

    ``score(rows, pred)`` must be signed (higher is better). Returns the full-data
    difference, its bootstrap standard error over rows, and per-fold differences.
    """
    n = len(fold_ids)
    every = np.arange(n)
    diff = score(every, pred_b) - score(every, pred_a)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        boots.append(score(rows, pred_b) - score(rows, pred_a))
    fold_diffs = []
    for fold in range(int(fold_ids.max()) + 1):
        rows = np.where(fold_ids == fold)[0]
        fold_diffs.append(score(rows, pred_b) - score(rows, pred_a))
    return {
        "diff": float(diff),
        "se_rows": float(np.std(boots, ddof=1)),
        "fold_diffs": [float(d) for d in fold_diffs],
        "folds_won": int(sum(d > 0 for d in fold_diffs)),
        "n_folds": len(fold_diffs),
        "n_boot": n_boot,
    }


def verdict(comparison: Dict[str, Any], sigma_seed: Optional[float] = None, z: float = DEFAULT_Z,
            min_fold_share: float = 0.8) -> Dict[str, Any]:
    """Accept b over a only if the gain clears the noise threshold and most folds agree."""
    sigma = sigma_seed or 0.0
    se_total = float(np.sqrt(comparison["se_rows"] ** 2 + 2 * sigma ** 2))
    threshold = z * se_total
    folds_needed = int(np.ceil(min_fold_share * comparison["n_folds"]))
    accept = comparison["diff"] > threshold and comparison["folds_won"] >= folds_needed
    return {
        "accept": bool(accept),
        "threshold": threshold,
        "se_total": se_total,
        "sigma_seed": sigma_seed,
        "seed_noise_measured": sigma_seed is not None,
        "folds_needed": folds_needed,
        "z": z,
    }


def latest_sigma_seed(problem_name: str, outputs_root: Optional[Path] = None) -> Optional[float]:
    """Seed noise from the most recent noise test of this problem, if any."""
    problem_dir = _outputs_root(outputs_root) / problem_name
    reports = sorted(problem_dir.glob(f"*/reports/{NOISE_REPORT}"), key=lambda p: p.stat().st_mtime)
    if not reports:
        return None
    return float(json.loads(reports[-1].read_text())["sigma_seed"])


NOISE_RUNS = "noise_runs.json"


def best_single_model(problem_name: str, run_id: str, outputs_root: Optional[Path] = None) -> str:
    """The run's best single model by out-of-fold score (from its final_predictions.json)."""
    final = json.loads((_outputs_root(outputs_root) / problem_name / run_id / "reports"
                        / "final_predictions.json").read_text())
    scores = final.get("single_scores") or {}
    if not scores:
        raise ValueError(f"run {run_id} has no single-model scores")
    pick = max if final.get("greater_is_better", True) else min
    return pick(scores, key=scores.get)


def queue_baseline_runs(problem_name: str, repeats: int = 1, outputs_root: Optional[Path] = None,
                        notes: str = "", start_worker: bool = True) -> Dict[str, Any]:
    """Queue the baseline the autopilot's judge needs, and its noise measurement.

    ``repeats`` runs of the problem's current settings, run k with ``budget.seed = k`` (k = 0 is the usual
    run). With one run (the default), a seed test follows it (``seed_noise_test`` on its best single model,
    minutes rather than a run): cheaper, but it misses the variation of re-tuning. With two or more, their
    spread is the noise, tuning path included. Recorded in ``outputs/<problem>/noise_runs.json``."""
    from datetime import datetime

    from src.genML import jobs

    if repeats < 1:
        raise ValueError("a baseline needs at least one run")
    root = _outputs_root(outputs_root)
    runs = []
    for k in range(repeats):
        label = f"[baseline {k + 1}/{repeats}, seed offset {k}]" + (f" {notes}" if notes else "")
        job = jobs.enqueue(problem_name, "run", {"budget": {"seed": k}}, label, outputs_root=root)
        runs.append({"job_id": job["id"], "run_id": job["run_id"], "seed": k})
    record = {"created_at": datetime.now().isoformat(timespec="seconds"), "repeats": repeats, "runs": runs}
    if repeats == 1:
        job = jobs.enqueue(problem_name, "noise", {"run_id": runs[0]["run_id"], "seeds": 3},
                           f"[baseline noise] seed test on {runs[0]['run_id']}", outputs_root=root)
        record["seed_test"] = {"job_id": job["id"], "run_id": runs[0]["run_id"]}
    path = root / problem_name / NOISE_RUNS
    path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(history + [record], indent=2))
    if start_worker:
        jobs.ensure_worker(root)
    return record


def run_noise(problem_name: str, outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The newest baseline group: its runs' final OOF scores and whether all of them finished."""
    root = _outputs_root(outputs_root)
    path = root / problem_name / NOISE_RUNS
    if not path.exists():
        return None
    group = json.loads(path.read_text())[-1]
    scores, missing = {}, []
    for run in group["runs"]:
        final = root / problem_name / run["run_id"] / "reports" / "final_predictions.json"
        if final.exists():
            scores[run["run_id"]] = json.loads(final.read_text()).get("final_score")
        else:
            missing.append(run["run_id"])
    values = [v for v in scores.values() if v is not None]
    sigma_seed = None
    if group.get("seed_test"):
        report = root / problem_name / group["seed_test"]["run_id"] / "reports" / NOISE_REPORT
        if report.exists():
            sigma_seed = json.loads(report.read_text()).get("sigma_seed")
        else:
            missing.append(f"seed test on {group['seed_test']['run_id']}")
    measured = len(values) >= 2 or sigma_seed is not None
    return {**group, "run_ids": [r["run_id"] for r in group["runs"]], "scores": scores, "missing": missing,
            "complete": not missing and measured, "sigma_seed": sigma_seed,
            "sigma_final": float(np.std(values, ddof=1)) if len(values) >= 2 else None}


def compare_experiments(problem_name: str, experiment_a: str, experiment_b: str,
                        outputs_root: Optional[Path] = None, n_boot: int = 200,
                        z: float = DEFAULT_Z) -> Dict[str, Any]:
    """Is stored experiment b really better than experiment a? (Both from the OOF store.)"""
    root = _outputs_root(outputs_root)
    problem_dir = root / problem_name
    store = ExperimentStore(problem_dir)
    exp_a, exp_b = store.get(experiment_a), store.get(experiment_b)
    signature = exp_a.meta.get("fold_signature")
    if signature != exp_b.meta.get("fold_signature"):
        raise ValueError("Experiments use different folds; paired comparison needs shared folds")
    run_id = exp_b.meta.get("run_id") or experiment_b.split("__")[0]
    problem = ResolvedProblem.load(problem_dir / run_id / "reports" / "resolved_spec.json")
    y, transform_info = store.load_targets(signature)
    fold_ids = np.load(problem_dir / f"folds_{signature}.npy")
    score = _signed_scorer(problem, y, transform_info)
    comparison = paired_compare(score, exp_a.oof, exp_b.oof, fold_ids, n_boot=n_boot)
    sigma = latest_sigma_seed(problem_name, root)
    result = {
        "problem": problem_name,
        "a": experiment_a,
        "b": experiment_b,
        "metric": problem.metric,
        "score_a": exp_a.meta.get("oof_score"),
        "score_b": exp_b.meta.get("oof_score"),
        **comparison,
        **verdict(comparison, sigma, z=z),
    }
    return result


def seed_noise_test(problem_name: str, run_id: str, model: str, n_seeds: int = 3,
                    outputs_root: Optional[Path] = None, n_boot: int = 200) -> Dict[str, Any]:
    """Refit one model of a finished run with extra seeds; measure the seed-to-seed spread.

    Reuses the run's saved features (including nested encoding blocks) and the
    model's tuned parameters, and the run's stored OOF predictions as the first
    seed. Writes ``reports/noise_test.json`` in the run folder.
    """
    from src.genML.pipeline.training import build_model_factories, load_feature_names

    root = _outputs_root(outputs_root)
    problem_dir = root / problem_name
    run_dir = problem_dir / run_id
    reports, features = run_dir / "reports", run_dir / "features"
    problem = ResolvedProblem.load(reports / "resolved_spec.json")
    training = json.loads((reports / "model_training_report.json").read_text())
    final = json.loads((reports / "final_predictions.json").read_text())
    result = training["all_results"].get(model)
    if result is None or not result.get("experiment_id"):
        raise ValueError(f"Run {run_id} has no stored OOF experiment for model '{model}'")
    params = dict(result.get("best_params") or {}) if result.get("tuned") else None

    store = ExperimentStore(problem_dir)
    base = store.get(result["experiment_id"])
    signature = final["fold_signature"]
    y, transform_info = store.load_targets(signature)
    fold_ids = np.load(problem_dir / f"folds_{signature}.npy")
    X = np.load(features / "X_train.npy")
    X_test = np.load(features / "X_test.npy")
    fold_features = FoldFeatures.load(features)
    task = problem.task
    n_classes = len(problem.classes) if problem.is_classification else 0
    problem_type = "regression" if task == "regression" else "classification"
    factory = build_model_factories(problem_type, task, X.shape[0], X.shape[1],
                                    feature_names=load_feature_names(features),
                                    model_params=(problem.budget or {}).get("model_params"))[model]
    score = _signed_scorer(problem, y, transform_info)
    every = np.arange(len(y))

    base_seeds = base.meta.get("seeds") or [42]
    seeds: List[int] = [int(base_seeds[0])]
    oofs = [base.oof]
    candidate = 1000
    while len(seeds) < n_seeds:
        print(f"Noise test: refitting {model} with seed {candidate} ({len(seeds) + 1}/{n_seeds})")
        oof, _ = fit_oof(factory, params, X, y, X_test, fold_ids, task, n_classes,
                         seeds=[candidate], fold_features=fold_features)
        seeds.append(candidate)
        oofs.append(oof)
        candidate += 1

    scores = [score(every, oof) for oof in oofs]
    pair_diffs, pair_ses = [], []
    for i, j in combinations(range(len(oofs)), 2):
        comparison = paired_compare(score, oofs[i], oofs[j], fold_ids, n_boot=n_boot)
        pair_diffs.append(comparison["diff"])
        pair_ses.append(comparison["se_rows"])
    sigma_seed = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    se_rows = float(np.mean(pair_ses)) if pair_ses else 0.0
    se_total = float(np.sqrt(se_rows ** 2 + 2 * sigma_seed ** 2))
    report = {
        "problem": problem_name,
        "run_id": run_id,
        "model": model,
        "metric": problem.metric,
        "seeds": seeds,
        "scores": [problem.metric_obj.signed(s) for s in scores],
        "sigma_seed": sigma_seed,
        "seed_pair_diffs": pair_diffs,
        "se_rows": se_rows,
        "se_total": se_total,
        "threshold": DEFAULT_Z * se_total,
        "z": DEFAULT_Z,
        "note": ("A new experiment must beat the current best by more than 'threshold' "
                 "(and win most folds) to count as a real improvement."),
    }
    (reports / NOISE_REPORT).write_text(json.dumps(report, indent=2))
    np.save(reports / NOISE_OOFS, np.stack([np.asarray(o, dtype=np.float32) for o in oofs]))
    return report


def range_edges(model: str, params: Dict[str, Any], tolerance: float = 0.03,
                budget: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Tuned values that sit at (or within ``tolerance`` of) an edge of their search range.

    A value on an edge means the best setting may lie outside the range searched,
    so the range should be widened there. Log-scale ranges are compared in log space.
    ``budget`` is the run's budget, whose ``search_space`` may have moved the ranges.
    """
    from src.genML.pipeline.tuning import model_search_space

    space = model_search_space(model, budget)
    if space is None or not params:
        return []
    edges = []
    for name, (_, low, high, log) in space.items():
        if name not in params:
            continue
        value = float(params[name])
        lo, hi, val = (np.log(low), np.log(high), np.log(value)) if log else (low, high, value)
        position = (val - lo) / (hi - lo)
        if position <= tolerance:
            edges.append({"model": model, "param": name, "value": value, "edge": "low", "bound": low})
        elif position >= 1 - tolerance:
            edges.append({"model": model, "param": name, "value": value, "edge": "high", "bound": high})
    return edges


def format_comparison(result: Dict[str, Any]) -> str:
    lines = [
        f"{result['b']}  vs  {result['a']}  ({result['metric']})",
        f"  gain (b - a):   {result['diff']:+.6f}",
        f"  row noise SE:   {result['se_rows']:.6f}",
        f"  seed noise:     {result['sigma_seed']:.6f}" if result["seed_noise_measured"]
        else "  seed noise:     not measured (run --noise-test first; threshold ignores it)",
        f"  threshold:      {result['threshold']:.6f}  ({result['z']:.0f} x total SE)",
        f"  folds won:      {result['folds_won']}/{result['n_folds']} (need {result['folds_needed']})",
        f"  verdict:        {'REAL IMPROVEMENT' if result['accept'] else 'not shown to be better'}",
    ]
    return "\n".join(lines)


def format_noise(report: Dict[str, Any]) -> str:
    scores = ", ".join(f"{s:.6f}" for s in report["scores"])
    return "\n".join([
        f"Seed noise for {report['model']} in run {report['run_id']} ({report['metric']})",
        f"  seeds:          {report['seeds']}",
        f"  scores:         {scores}",
        f"  seed spread:    {report['sigma_seed']:.6f} (std across seeds)",
        f"  row noise SE:   {report['se_rows']:.6f} (paired, between seeds)",
        f"  total SE:       {report['se_total']:.6f}",
        f"  accept a gain only above {report['threshold']:.6f} (3 x total SE) and on most folds",
    ])


__all__: Sequence[str] = (
    "compare_experiments", "format_comparison", "format_noise", "latest_sigma_seed",
    "paired_compare", "range_edges", "seed_noise_test", "verdict",
)
