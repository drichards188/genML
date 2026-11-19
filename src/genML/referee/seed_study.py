"""How often would a single-run comparison call a rerun of the same settings a gain?

Takes one entry run with many seeds (identical settings, so every difference between its runs is
noise) and reports, on the contest's deciding rows:

- DeLong's paired AUC test on every pair of single runs: the share with p < 0.05. A test that only
  counts row noise and is not fooled by the tool's run-to-run spread gives about 5%.
- the same test on seed averages of disjoint groups of runs (the referee's compute), and the referee's
  own grades on those groups (``verdict.grade_groups``), which should be likely or proven at most 5%
  of the time;
- per run: its score, seconds, and the models it fitted (AutoGluon logs), since a time limit makes a
  tool's output depend on the machine's load.

The decision rule was fixed before the data (DECISIONS.md, 2026-11-18): single-run DeLong p < 0.05 in
at least 15% of pairs confirms that seed noise fools row-only tests for this tool; under 10% means it
does not; between is inconclusive.

    python -m src.genML.referee.seed_study --contest ieee-100k --entry ag-medium
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from src.genML.referee import contest as contests
from src.genML.referee.arena import entries
from src.genML.referee.verdict import _group, _truth, grade_groups

CONFIRMED, NOT_CONFIRMED = 0.15, 0.10
GROUP_SEED = 20261118


def delong(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """DeLong's test for two correlated AUCs on the same rows (Sun & Xu's fast form): AUCs, and the
    standard error, z and two-sided p-value of AUC(a) - AUC(b)."""
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    v10, v01, aucs = [], [], []
    for s in (a, b):
        tx, ty = rankdata(s[pos]), rankdata(s[neg])
        tz = rankdata(np.concatenate([s[pos], s[neg]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.array(v10)), np.cov(np.array(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    se = float(np.sqrt(var)) if var > 0 else 0.0
    z = (aucs[0] - aucs[1]) / se if se > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "se": se, "z": float(z),
            "p": float(2 * norm.sf(abs(z)))}


def _rate(ps: List[float]) -> float:
    return float(np.mean([p < 0.05 for p in ps])) if ps else float("nan")


def _run_facts(item: Dict[str, Any]) -> Dict[str, Any]:
    folder = Path(item["predictions_path"]).parent
    record = json.loads((folder / "entry.json").read_text())
    log = folder / "log.txt"
    models = sorted(set(re.findall(r"Fitting model: (\w+)", log.read_text(errors="replace")))) if log.exists() else []
    return {"seed": item["seed"], "seconds": record.get("seconds"), "models": len(models), "model_names": models}


def study(contest: str, entry: str, size: int = 3, n_groups: int = 100, n_boot: int = 300,
          outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    record = contests.load_contest(contest, outputs_root)
    if record["task"] != "binary":
        raise ValueError("the seed study uses DeLong's AUC test, so it needs a binary contest")
    items = sorted(entries(contest, outputs_root)[entry], key=lambda item: item["seed"])
    labels, codes = contests.labels_and_rows(contest, outputs_root)
    y = _truth(record, labels)
    decide = codes == contests.DECIDE
    preds = {item["seed"]: np.load(item["predictions_path"]) for item in items}
    seeds = list(preds)

    singles = []
    for s, t in itertools.combinations(seeds, 2):
        result = delong(y[decide], preds[s][decide], preds[t][decide])
        singles.append({"a": s, "b": t, **result})

    rng = np.random.default_rng(GROUP_SEED)
    seen, grouped = set(), []
    possible = comb(len(seeds), size) * comb(len(seeds) - size, size) // 2   # unordered disjoint pairs
    while len(grouped) < min(n_groups, possible) and len(seeds) >= 2 * size:
        drawn = [int(v) for v in rng.permutation(seeds)[: 2 * size]]
        a, b = tuple(sorted(drawn[:size])), tuple(sorted(drawn[size:]))
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        group_a = _group([item for item in items if item["seed"] in a])
        group_b = _group([item for item in items if item["seed"] in b])
        test = delong(y[decide], group_a["mean"][decide], group_b["mean"][decide])
        verdict = grade_groups(record, y, codes, group_a, group_b, n_boot=n_boot)
        grouped.append({"a": list(a), "b": list(b), "delong_z": test["z"], "delong_p": test["p"],
                        "grade": verdict["grade"], "gain": verdict["gain"], "se_total": verdict["se_total"]})

    single_rate = _rate([r["p"] for r in singles])
    if single_rate >= CONFIRMED:
        conclusion = "confirmed: seed noise fools single-run DeLong for this tool"
    elif single_rate < NOT_CONFIRMED:
        conclusion = "not confirmed: single-run DeLong is close enough to calibrated for this tool"
    else:
        conclusion = "inconclusive"
    grades = [g["grade"] for g in grouped]
    per_run = [{**_run_facts(item), "auc": float(roc_auc_score(y[decide], preds[item["seed"]][decide]))}
               for item in items]
    return {
        "contest": contest, "entry": entry, "seeds": seeds, "decide_rows": int(decide.sum()),
        "positives": int(y[decide].sum()),
        "single_runs": {"pairs": len(singles), "delong_p05_rate": single_rate,
                        "abs_z_mean": float(np.mean([abs(r["z"]) for r in singles])),
                        "abs_z_max": float(np.max([abs(r["z"]) for r in singles])), "pairs_detail": singles},
        "groups": {"size": size, "comparisons": len(grouped),
                   "delong_p05_rate": _rate([g["delong_p"] for g in grouped]),
                   "referee_grades": {g: grades.count(g) for g in sorted(set(grades))},
                   "referee_likely_or_proven_rate": float(np.mean([g in ("likely", "proven", "suspicious")
                                                                   for g in grades])) if grades else None,
                   "detail": grouped},
        "per_run": per_run,
        "auc_sd": float(np.std([r["auc"] for r in per_run], ddof=1)),
        "rule": {"confirmed_at_least": CONFIRMED, "not_confirmed_below": NOT_CONFIRMED},
        "conclusion": conclusion,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contest", default="ieee-100k")
    parser.add_argument("--entry", default="ag-medium")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--groups", type=int, default=100)
    args = parser.parse_args(argv)
    result = study(args.contest, args.entry, size=args.size, n_groups=args.groups)
    out = contests.referee_root() / "benchmark" / f"{args.contest}-{args.entry}-seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    s, g = result["single_runs"], result["groups"]
    print(f"{args.entry} on {args.contest}: {len(result['seeds'])} runs, AUC sd {result['auc_sd']:.4f}")
    for run in result["per_run"]:
        print(f"  seed {run['seed']}: AUC {run['auc']:.4f}, {run['seconds']} s, {run['models']} models")
    print(f"single runs: DeLong p<0.05 in {s['delong_p05_rate']:.0%} of {s['pairs']} pairs "
          f"(|z| mean {s['abs_z_mean']:.2f}, max {s['abs_z_max']:.2f})")
    print(f"{g['size']}-run groups: DeLong p<0.05 in {g['delong_p05_rate']:.0%} of {g['comparisons']}; "
          f"referee grades {g['referee_grades']}")
    print(f"conclusion: {result['conclusion']}\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
