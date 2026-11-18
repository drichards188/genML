"""The trusted judge: measurement inputs pinned before any agent runs, then a paired verdict.

Anchors (``make_anchors``) are taken once, from clean runs made by trusted code:

* the fold assignment, the stored training target, the metric and task;
* which rows decide (``rows.npy``): acceptance rows, and an untouched holdout the judge never uses to
  accept anything, so the gain the loop accumulates by choosing among many ideas can be checked on rows
  no choice looked at. Random and grouped folds: a stratified 20% of rows is held out. Time blocks:
  only the latest block counts (its models were trained on earlier blocks only, as in real use); half
  of it decides, the other half is the holdout;
* the run-to-run noise: the spread of the final scores of repeated baseline runs that differ only in
  their seeds (``--baseline``; it includes the tuning path, which a seed-only refit misses), plus the
  seed noise of ``--noise-test`` if there is one.

Every verdict first checks those files against their hashes and the current best experiment's
predictions against theirs, then rescores both experiments with the anchored metric on the acceptance
rows. A gain far above the noise bar is held as ``suspicious`` (likely a leak) for a person. Nothing the
agents' code writes (run reports, noise tests, resolved specs) decides a verdict.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.genML.pipeline.diagnostics import (DEFAULT_Z, NOISE_OOFS, _signed_scorer, latest_sigma_seed, paired_compare,
                                             run_noise, verdict)
from src.genML.pipeline.metrics import get_metric
from src.genML.pipeline.oof import ExperimentStore

ROWS_FILE = "rows.npy"            # per row: 1 decides, 2 holdout (never decides), 0 not scored
HOLDOUT_FRACTION = 0.2
ROWS_SEED = 20261110
ACCEPT, HOLDOUT = 1, 2
TIME_CHUNKS = 5                   # the time block's acceptance rows are split this way for the fold-win rule
SUSPICIOUS_RATIO = 10.0           # a gain this many times the noise bar is held for a person


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(problem_dir: Path, signature: str) -> Dict[str, Path]:
    targets = problem_dir / "oof" / "_targets"
    return {"folds": problem_dir / f"folds_{signature}.npy", "targets": targets / f"y_{signature}.npy",
            "transform": targets / f"transform_{signature}.json"}


def best_record(problem_dir: Path, experiment_id: str, score: float, run_id: str) -> Dict[str, Any]:
    oof_path = ExperimentStore(problem_dir).get(experiment_id).path / "oof.npy"
    return {"run_id": run_id, "experiment_id": experiment_id, "score": float(score), "oof_sha": sha256(oof_path)}


# --------------------------------------------------------------------------- #
# Which rows decide
# --------------------------------------------------------------------------- #

def choose_rows(fold_ids: np.ndarray, y: np.ndarray, task: str, cv_method: Optional[str]) -> Tuple[np.ndarray, Dict]:
    """Per-row codes (ACCEPT, HOLDOUT, 0) and a description of the scheme."""
    rng = np.random.default_rng(ROWS_SEED)
    codes = np.zeros(len(fold_ids), dtype=np.int8)
    if cv_method == "time_blocks":
        block = int(fold_ids.max())
        rows = rng.permutation(np.where(fold_ids == block)[0])
        codes[rows[: len(rows) // 2]] = ACCEPT
        codes[rows[len(rows) // 2:]] = HOLDOUT
        return codes, {"kind": "time_forward", "block": block,
                       "why": "only the latest block's models were trained on the past alone"}
    codes[:] = ACCEPT
    groups = [np.where(y == value)[0] for value in np.unique(y)] if task != "regression" else [np.arange(len(y))]
    for rows in groups:
        chosen = rng.choice(rows, size=int(round(len(rows) * HOLDOUT_FRACTION)), replace=False)
        codes[chosen] = HOLDOUT
    return codes, {"kind": "random_holdout", "fraction": HOLDOUT_FRACTION}


def _acceptance(codes: np.ndarray, fold_ids: np.ndarray, scheme: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Acceptance rows and the fold id of each, for the fold-win rule."""
    keep = np.where(codes == ACCEPT)[0]
    if scheme.get("kind") == "time_forward":
        chunks = np.random.default_rng(ROWS_SEED + 1).permutation(len(keep)) % TIME_CHUNKS
        return keep, chunks
    return keep, fold_ids[keep]


def _scorer(anchors: Dict[str, Any], problem_dir: Path):
    store = ExperimentStore(problem_dir)
    y, transform = store.load_targets(anchors["fold_signature"])
    metric = get_metric(anchors["metric"])
    problem = SimpleNamespace(metric_obj=metric, task=anchors["task"])
    return metric, problem, y, transform


def _subset_scorer(problem, y: np.ndarray, transform: Dict[str, Any], rows: np.ndarray):
    """score(sub_rows, pred_on_rows) over ``rows`` only (what paired_compare expects)."""
    return _signed_scorer(problem, y[rows], transform)


def _final_oof(problem_dir: Path, run_id: str) -> Optional[np.ndarray]:
    path = problem_dir / run_id / "reports" / "final_predictions.json"
    if not path.exists():
        return None
    experiment_id = json.loads(path.read_text()).get("final_experiment_id")
    try:
        return ExperimentStore(problem_dir).get(experiment_id).oof
    except KeyError:
        return None


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #

def make_anchors(problem: str, outputs_root: Path) -> Dict[str, Any]:
    """Pin the judge's inputs from the best clean run that exists now (made by trusted code).

    Refuses a baseline whose settings let labels leak, and refuses to start without a finished noise
    measurement (``--baseline``)."""
    from src.genML.experiments import current_best, leak_problems

    problem_dir = Path(outputs_root) / problem
    best = current_best(problem, outputs_root)
    if best is None:
        raise ValueError(f"No completed (and not excluded) run of '{problem}' to start from; "
                         f"run --baseline {problem} first")
    resolved = json.loads((problem_dir / best["run_id"] / "reports" / "resolved_spec.json").read_text())
    leaks = leak_problems({"feature_engineering": resolved.get("feature_engineering") or {}}, resolved.get("target"))
    if leaks:
        raise ValueError(f"The best run {best['run_id']} has settings that let labels leak ({'; '.join(leaks)}). "
                         f"Fix problems/{problem}/problem.yaml, then run --baseline {problem}.")
    noise = run_noise(problem, outputs_root)
    if not noise or not noise.get("complete"):
        state = "is still running" if noise else "is missing"
        raise ValueError(f"The noise measurement for '{problem}' {state}: run --baseline {problem} (repeated runs "
                         "of the same settings with different seeds) and wait for every run to finish.")
    signature = best["fold_signature"]
    paths = _paths(problem_dir, signature)
    metric, scored_problem, y, transform = _scorer({"fold_signature": signature, "metric": resolved["metric"],
                                                    "task": resolved["task"]}, problem_dir)
    fold_ids = np.load(paths["folds"])
    cv_method = (resolved.get("cv") or {}).get("method")
    codes, scheme = choose_rows(fold_ids, y, resolved["task"], cv_method)
    home = problem_dir / "autopilot"
    home.mkdir(parents=True, exist_ok=True)
    rows_path = home / ROWS_FILE
    np.save(rows_path, codes)

    keep = np.where(codes == ACCEPT)[0]
    hold = np.where(codes == HOLDOUT)[0]
    accept_score = _subset_scorer(scored_problem, y, transform, keep)
    every_kept = np.arange(len(keep))
    noise_scores = []
    for run_id in noise["run_ids"]:
        oof = _final_oof(problem_dir, run_id)
        if oof is not None and len(oof) == len(y):
            noise_scores.append(accept_score(every_kept, oof[keep]))
    sigma_run = float(np.std(noise_scores, ddof=1)) if len(noise_scores) >= 2 else None
    # One baseline run: the seed test's spread, rescored on the same acceptance rows (fewer rows, more noise)
    sigma_seed = noise.get("sigma_seed")
    seed_oofs_path = problem_dir / noise["seed_test"]["run_id"] / "reports" / NOISE_OOFS if noise.get(
        "seed_test") else None
    if seed_oofs_path is not None and seed_oofs_path.exists():
        seed_scores = [accept_score(every_kept, oof[keep]) for oof in np.load(seed_oofs_path)]
        sigma_seed = float(np.std(seed_scores, ddof=1))
    if sigma_run is None and sigma_seed is None:
        raise ValueError(f"No noise figure for '{problem}': the baseline needs two runs or a seed test.")
    best_oof = ExperimentStore(problem_dir).get(best["experiment_id"]).oof
    holdout_score = _subset_scorer(scored_problem, y, transform, hold)
    anchors = {
        "fold_signature": signature,
        "hashes": {name: sha256(path) for name, path in paths.items()},
        "metric": resolved["metric"],
        "task": resolved["task"],
        "target": resolved.get("target"),
        "cv_method": cv_method,
        "n_rows": int(len(y)),
        "rows": {"file": str(rows_path), "sha": sha256(rows_path), "scheme": scheme,
                 "n_accept": int(len(keep)), "n_holdout": int(len(hold))},
        "sigma_run": sigma_run,
        "noise_runs": noise["run_ids"],
        "sigma_seed": sigma_seed if sigma_seed is not None else latest_sigma_seed(problem, outputs_root),
        "z": DEFAULT_Z,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best": best_record(problem_dir, best["experiment_id"],
                            metric.signed(accept_score(every_kept, best_oof[keep])), best["run_id"]),
        "holdout_start": metric.signed(holdout_score(np.arange(len(hold)), best_oof[hold])),
    }
    return anchors


def check_integrity(anchors: Dict[str, Any], best: Dict[str, Any], problem_dir: Path) -> List[str]:
    """Anchored files, the row split and the best experiment's predictions must be unchanged."""
    errors = []
    if "rows" not in anchors:
        return ["these anchors predate the holdout and run-noise rules; reset this problem's autopilot"]
    for name, path in _paths(problem_dir, anchors["fold_signature"]).items():
        if not path.exists():
            errors.append(f"{name} file {path} is missing")
        elif sha256(path) != anchors["hashes"][name]:
            errors.append(f"{name} file {path} changed since the anchors were taken")
    rows_path = Path(anchors["rows"]["file"])
    if not rows_path.exists() or sha256(rows_path) != anchors["rows"]["sha"]:
        errors.append(f"the row split {rows_path} is missing or changed")
    try:
        oof_path = ExperimentStore(problem_dir).get(best["experiment_id"]).path / "oof.npy"
        if sha256(oof_path) != best["oof_sha"]:
            errors.append(f"predictions of the best experiment {best['experiment_id']} changed")
    except KeyError:
        errors.append(f"best experiment {best['experiment_id']} is missing from the store")
    return errors


def _check_predictions(oof: np.ndarray, anchors: Dict[str, Any]) -> List[str]:
    errors = []
    if oof.shape[0] != anchors["n_rows"]:
        errors.append(f"OOF predictions have {oof.shape[0]} rows, expected {anchors['n_rows']}")
    if not np.all(np.isfinite(oof)):
        errors.append("OOF predictions contain NaN or infinite values")
    elif anchors["task"] != "regression" and (oof.min() < -1e-6 or oof.max() > 1 + 1e-6):
        errors.append("OOF class probabilities fall outside [0, 1]")
    return errors


def noise_sigma(anchors: Dict[str, Any]) -> float:
    """The run-to-run noise the bar uses: the larger of the repeated-run and seed-refit estimates."""
    return float(max(anchors.get("sigma_run") or 0.0, anchors.get("sigma_seed") or 0.0))


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

def judge_run(anchors: Dict[str, Any], best: Dict[str, Any], problem: str, run_id: str, outputs_root: Path,
              suspicious_ratio: float = SUSPICIOUS_RATIO, suspicious_gain: Optional[float] = None,
              n_boot: int = 200) -> Dict[str, Any]:
    """Verdict on a finished run against the current best, from anchored inputs only.

    The result's ``holdout`` entry (scores on rows that never decide) is for the person; the loop keeps it
    out of what the agents read."""
    problem_dir = Path(outputs_root) / problem
    errors = check_integrity(anchors, best, problem_dir)
    if errors:
        return {"verdict": "integrity_failure", "errors": errors}
    final_path = problem_dir / run_id / "reports" / "final_predictions.json"
    if not final_path.exists():
        return {"verdict": "failed", "errors": [f"run {run_id} wrote no final_predictions.json"]}
    candidate_id = json.loads(final_path.read_text()).get("final_experiment_id")
    store = ExperimentStore(problem_dir)
    try:
        candidate = store.get(candidate_id)
    except KeyError:
        return {"verdict": "failed", "errors": [f"final experiment {candidate_id!r} is not in the store"]}
    if candidate.meta.get("run_id") != run_id:
        return {"verdict": "failed", "errors": [f"{candidate_id} does not belong to run {run_id}"]}
    if candidate.meta.get("fold_signature") != anchors["fold_signature"]:
        return {"verdict": "failed", "errors": ["the run used different folds from the anchored ones"]}

    metric, scored_problem, y, transform = _scorer(anchors, problem_dir)
    fold_ids = np.load(_paths(problem_dir, anchors["fold_signature"])["folds"])
    codes = np.load(anchors["rows"]["file"])
    keep, keep_folds = _acceptance(codes, fold_ids, anchors["rows"]["scheme"])
    hold = np.where(codes == HOLDOUT)[0]
    base_oof, cand_oof = store.get(best["experiment_id"]).oof, candidate.oof
    errors = _check_predictions(cand_oof, anchors)
    if errors:
        return {"verdict": "failed", "errors": errors}
    accept_score = _subset_scorer(scored_problem, y, transform, keep)
    comparison = paired_compare(accept_score, base_oof[keep], cand_oof[keep], keep_folds, n_boot=n_boot)
    rule = verdict(comparison, noise_sigma(anchors), z=anchors.get("z", DEFAULT_Z))
    every_kept = np.arange(len(keep))
    candidate_score = metric.signed(accept_score(every_kept, cand_oof[keep]))
    status = "accepted" if rule["accept"] else "rejected"
    limit = suspicious_ratio * rule["threshold"]
    if suspicious_gain is not None:
        limit = min(limit, suspicious_gain)
    if rule["accept"] and comparison["diff"] > limit:
        status = "suspicious"
    per_model = {}
    for experiment in store.list(anchors["fold_signature"]):
        if experiment.meta.get("run_id") == run_id and not _check_predictions(experiment.oof, anchors):
            per_model[experiment.meta.get("name") or experiment.experiment_id] = round(
                metric.signed(accept_score(every_kept, experiment.oof[keep])), 6)
    holdout_score = _subset_scorer(scored_problem, y, transform, hold)
    every_held = np.arange(len(hold))
    return {
        "verdict": status,
        "candidate_experiment_id": candidate_id,
        "candidate_score": candidate_score,
        "best_experiment_id": best["experiment_id"],
        "best_score": metric.signed(accept_score(every_kept, base_oof[keep])),
        "gain": comparison["diff"],
        "threshold": rule["threshold"],
        "suspicious_limit": limit,
        "folds_won": f"{comparison['folds_won']}/{comparison['n_folds']}",
        "fold_diffs": comparison["fold_diffs"],
        "rows": anchors["rows"]["scheme"]["kind"],
        "models": per_model,
        "metric": anchors["metric"],
        "holdout": {"candidate": metric.signed(holdout_score(every_held, cand_oof[hold])),
                    "best": metric.signed(holdout_score(every_held, base_oof[hold]))},
    }


# --------------------------------------------------------------------------- #
# Shuffled-label test
# --------------------------------------------------------------------------- #

def canary_verdict(anchors: Dict[str, Any], canary_root: Path, problem: str, run_id: str,
                   z: float = DEFAULT_Z) -> Dict[str, Any]:
    """Did a run on shuffled labels learn anything? It must not beat a constant prediction.

    The canary run (the cycle's code and settings, labels permuted) stores its own target and folds in
    ``canary_root``; with no relation left between features and labels, any real gain over predicting
    the label rate (or mean) means labels reach the predictions some other way: a leak."""
    problem_dir = Path(canary_root) / problem
    final_path = problem_dir / run_id / "reports" / "final_predictions.json"
    if not final_path.exists():
        return {"verdict": "failed", "errors": [f"the canary run {run_id} wrote no final_predictions.json"]}
    final = json.loads(final_path.read_text())
    store = ExperimentStore(problem_dir)
    try:
        oof = store.get(final.get("final_experiment_id")).oof
    except KeyError:
        return {"verdict": "failed", "errors": ["the canary's final experiment is not in its store"]}
    signature = final.get("fold_signature") or store.get(final["final_experiment_id"]).meta.get("fold_signature")
    y, transform = store.load_targets(signature)
    fold_ids = np.load(problem_dir / f"folds_{signature}.npy")
    metric = get_metric(anchors["metric"])
    scored_problem = SimpleNamespace(metric_obj=metric, task=anchors["task"])
    score = _signed_scorer(scored_problem, y, transform)
    if anchors["task"] == "regression":
        constant = np.full(len(y), float(np.mean(y)), dtype=float)
    elif oof.ndim == 2:
        priors = np.array([np.mean(y == k) for k in range(oof.shape[1])], dtype=float)
        constant = np.tile(priors, (len(y), 1))
    else:
        constant = np.full(len(y), float(np.mean(y)), dtype=float)
    comparison = paired_compare(score, constant, oof, fold_ids, n_boot=200)
    bar = z * comparison["se_rows"]
    leak = comparison["diff"] > bar and comparison["folds_won"] >= int(np.ceil(0.8 * comparison["n_folds"]))
    return {"verdict": "leak" if leak else "clean", "gain_over_constant": comparison["diff"], "bar": bar,
            "folds_won": f"{comparison['folds_won']}/{comparison['n_folds']}", "metric": anchors["metric"]}
