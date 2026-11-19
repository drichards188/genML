"""Graded verdicts: is one entry really better than another on the hidden rows?

Every entry is scored on the deciding rows with the contest's metric. An entry run with several seeds is
scored on its seed-averaged predictions, and the spread of its per-seed scores measures its run-to-run
noise. Comparing a candidate with a baseline uses a paired bootstrap over the deciding rows (row noise)
plus both entries' seed noise:

    se_total = sqrt(se_rows**2 + sd_base**2 / n_base + sd_cand**2 / n_cand)

and gives one grade, so a real but small gain is reported as such instead of being thrown away:

    proven        gain > 3 se_total and the candidate wins at least 80% of the row chunks
    likely        gain > 2 se_total and it wins at least 60% of the chunks (real more often than not,
                  not proven); also the ceiling when no entry's seed noise was measured
    no_evidence   anything in between: the data cannot tell
    worse         loss > 3 se_total and it loses at least 80% of the chunks
    suspicious    gain > 10 x the proven bar: too good to be true until a leak check clears it

``smallest_provable_gain`` (3 se_total) is what this contest can resolve. The holdout rows never decide;
their difference is reported, and a proven or likely gain whose holdout difference is negative is
flagged. ``optimism`` is what a tool claims about itself (its self-reported validation score) minus what
it scores on the hidden rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.genML.pipeline.diagnostics import paired_compare
from src.genML.pipeline.metrics import get_metric
from src.genML.referee.arena import entries
from src.genML.referee.contest import DECIDE, HOLDOUT, labels_and_rows, load_contest

Z_PROVEN, Z_LIKELY = 3.0, 2.0
PROVEN_SHARE, LIKELY_SHARE = 0.8, 0.6
SUSPICIOUS_RATIO = 10.0
CHUNKS = 5
CHUNK_SEED = 20261119


def _truth(record: Dict[str, Any], labels: np.ndarray) -> np.ndarray:
    if record["task"] == "binary":
        return (labels == record["classes"][1]).astype(int)
    if record["task"] == "multiclass":
        index = {value: k for k, value in enumerate(record["classes"])}
        return np.array([index[value] for value in labels])
    return labels.astype(float)


def _scorer(record: Dict[str, Any], y: np.ndarray, rows: np.ndarray):
    """score(sub_rows, pred_on_rows), signed so higher is better, over ``rows`` only."""
    metric, task, y_rows = get_metric(record["metric"]), record["task"], y[rows]

    def score(sub: np.ndarray, pred: np.ndarray) -> float:
        return metric.signed(metric.score(y_rows[sub], pred[sub], task))

    return score


def _group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [np.load(item["predictions_path"]) for item in items]
    return {"seeds": [item["seed"] for item in items], "per_seed": preds, "mean": np.mean(preds, axis=0),
            "self_reports": [item.get("self_report") for item in items]}


def _split_rows(codes: np.ndarray):
    decide, hold = np.where(codes == DECIDE)[0], np.where(codes == HOLDOUT)[0]
    chunks = np.random.default_rng(CHUNK_SEED).permutation(len(decide)) % CHUNKS
    return decide, hold, chunks


def entry_summary(contest: str, outputs_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Per entry: seeds, deciding-row score of the seed average, per-seed scores and spread, holdout
    score, and the tool's own claim about its score (with the optimism it implies)."""
    record = load_contest(contest, outputs_root)
    labels, codes = labels_and_rows(contest, outputs_root)
    y = _truth(record, labels)
    decide, hold, _ = _split_rows(codes)
    on_decide, on_hold = _scorer(record, y, decide), _scorer(record, y, hold)
    every_d, every_h = np.arange(len(decide)), np.arange(len(hold))
    metric = get_metric(record["metric"])
    out = {}
    for name, items in entries(contest, outputs_root).items():
        group = _group(items)
        per_seed = [metric.signed(on_decide(every_d, p[decide])) for p in group["per_seed"]]
        score = metric.signed(on_decide(every_d, group["mean"][decide]))
        claims = [r.get("validation_score") for r in group["self_reports"] if isinstance(r, dict)
                  and isinstance(r.get("validation_score"), (int, float))]
        claim = float(np.mean(claims)) if claims else None
        out[name] = {"seeds": group["seeds"], "score": score, "per_seed": per_seed,
                     "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
                     "holdout": metric.signed(on_hold(every_h, group["mean"][hold])),
                     "self_reported": claim,
                     "optimism": (metric.signed(claim) - metric.signed(score)) if claim is not None else None}
    return out


def compare(contest: str, base: str, candidate: str, outputs_root: Optional[Path] = None,
            prior_sd: Optional[float] = None, n_boot: int = 300) -> Dict[str, Any]:
    """Graded verdict on ``candidate`` against ``base`` (see the module docstring)."""
    record = load_contest(contest, outputs_root)
    groups = entries(contest, outputs_root)
    missing = [name for name in (base, candidate) if name not in groups]
    if missing:
        raise KeyError(f"no finished entries named {missing} in contest {contest!r}")
    labels, codes = labels_and_rows(contest, outputs_root)
    result = grade_groups(record, _truth(record, labels), codes, _group(groups[base]), _group(groups[candidate]),
                          prior_sd=prior_sd, n_boot=n_boot)
    return {"contest": contest, "base": base, "candidate": candidate, **result}


def grade_groups(record: Dict[str, Any], y: np.ndarray, codes: np.ndarray, a: Dict[str, Any], b: Dict[str, Any],
                 prior_sd: Optional[float] = None, n_boot: int = 300) -> Dict[str, Any]:
    """The verdict on group ``b`` against group ``a`` (each: ``seeds``, ``per_seed`` predictions and their
    ``mean``), for callers that form groups of runs themselves (the seed-noise study)."""
    decide, hold, chunks = _split_rows(codes)
    on_decide = _scorer(record, y, decide)
    comparison = paired_compare(on_decide, a["mean"][decide], b["mean"][decide], chunks, n_boot=n_boot)

    def seed_sd(group):
        scores = [on_decide(np.arange(len(decide)), p[decide]) for p in group["per_seed"]]
        return float(np.std(scores, ddof=1)) if len(scores) > 1 else None

    sd_a, sd_b = seed_sd(a), seed_sd(b)
    measured = [sd for sd in (sd_a, sd_b) if sd is not None]
    fallback = float(np.mean(measured)) if measured else prior_sd
    noise_known = fallback is not None
    var_a = (sd_a if sd_a is not None else (fallback or 0.0)) ** 2 / len(a["seeds"])
    var_b = (sd_b if sd_b is not None else (fallback or 0.0)) ** 2 / len(b["seeds"])
    se_total = float(np.sqrt(comparison["se_rows"] ** 2 + var_a + var_b))
    diff, n = comparison["diff"], comparison["n_folds"]
    won = comparison["folds_won"]
    lost = sum(d < 0 for d in comparison["fold_diffs"])
    proven_bar = Z_PROVEN * se_total
    if diff > SUSPICIOUS_RATIO * proven_bar and won >= np.ceil(PROVEN_SHARE * n):
        grade = "suspicious"
    elif diff > proven_bar and won >= np.ceil(PROVEN_SHARE * n) and noise_known:
        grade = "proven"
    elif diff > Z_LIKELY * se_total and won >= np.ceil(LIKELY_SHARE * n):
        grade = "likely"
    elif diff < -proven_bar and lost >= np.ceil(PROVEN_SHARE * n):
        grade = "worse"
    else:
        grade = "no_evidence"
    on_hold = _scorer(record, y, hold)
    every_h = np.arange(len(hold))
    holdout_diff = on_hold(every_h, b["mean"][hold]) - on_hold(every_h, a["mean"][hold])
    flags = []
    if not noise_known:
        flags.append("seed noise not measured: run each entry with 2 or more seeds; 'proven' is unavailable")
    if grade in ("proven", "likely", "suspicious") and holdout_diff < 0:
        flags.append("the holdout rows disagree (negative difference there)")
    if grade == "suspicious":
        flags.append("gain far above the noise: run the leak checks before trusting it")
    return {"grade": grade, "gain": diff,
            "se_rows": comparison["se_rows"], "seed_sd": {"base": sd_a, "candidate": sd_b},
            "seeds": {"base": len(a["seeds"]), "candidate": len(b["seeds"])}, "se_total": se_total,
            "smallest_provable_gain": proven_bar, "chunks_won": f"{won}/{n}",
            "chunk_diffs": comparison["fold_diffs"], "holdout_gain": float(holdout_diff), "flags": flags,
            "metric": record["metric"], "rows": {"decide": len(decide), "holdout": len(hold)}}
