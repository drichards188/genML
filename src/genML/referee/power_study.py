"""Does the referee find real gains, and avoid false ones, better than simpler tests?

Every comparison is 3 runs against 3 on the contest's deciding rows, judged five ways:

- ``delong_single``: DeLong's paired AUC test on one run per side (row noise only);
- ``delong_mean``: DeLong on the 3-run seed averages (row noise only, more compute);
- ``t_test``: Welch's t-test on each run's AUC (seed noise only: "better on these exact rows");
- ``formula``: DeLong's row standard error plus each side's seed variance over its runs, |z| > 2 (the
  textbook combination);
- ``referee``: ``verdict.grade_groups`` (likely or proven; the mirror rule for a loss).

Three kinds of comparison:

- same settings: disjoint groups of identical runs (every call is false);
- different but equal: models that differ only in which pure-noise columns they got (none is better
  in general, but each errs on different rows; every call is false);
- planted gains: a model with real columns withheld against the full model (the full one is better).

Each call is also checked against the holdout rows (same sign there or not). The decision rule was
fixed before the data (DECISIONS.md, 2026-11-18).

    python -m src.genML.referee.power_study [--contest ieee-100k] [--skip-runs]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

from src.genML.referee import benchmark
from src.genML.referee import contest as contests
from src.genML.referee.arena import entries
from src.genML.referee.seed_study import delong
from src.genML.referee.verdict import LIKELY_SHARE, Z_LIKELY, _truth, grade_groups

TESTS = ("delong_single", "delong_mean", "t_test", "formula", "referee")
CONFIRMED, NOT_CONFIRMED = 0.15, 0.10           # the t-test's call rate on different-but-equal pairs
REFEREE_AT_MOST = 0.05
DRAW_SEED = 20261122

FULL = list(benchmark.NULL_GROUPS)                                    # the full model, seeds 0 to 14
WITHHELD_SEEDS = list(range(30, 45))
NOISE_DRAWS = {"lgb-noise10": 777, **{f"lgb-noise10-n{k}": k for k in range(1001, 1008)}}
DESIGN = {"same": {"full": FULL, **{entry: [entry] for entry in benchmark.WITHHELD}},
          "equal": list(NOISE_DRAWS),
          "planted": {entry: FULL for entry in benchmark.WITHHELD}}   # worse entry: pool of better runs


def make_entries(name: str, outputs_root: Optional[Path] = None) -> None:
    """The runs the default design needs (resumable: finished runs are kept)."""
    for entry, extra in benchmark.WITHHELD.items():
        benchmark._enter(name, entry, [*benchmark.LGB, *extra], WITHHELD_SEEDS, outputs_root=outputs_root)
    for k, (entry, draw) in enumerate(NOISE_DRAWS.items()):
        if entry == "lgb-noise10":                                    # made by the benchmark, seeds 20 to 22
            continue
        command = [*benchmark.LGB, "--noise-columns", "10", "--noise-seed", str(draw)]
        benchmark._enter(name, entry, command, range(300 + 3 * k, 303 + 3 * k), outputs_root=outputs_root)


def _sign(value: float) -> int:
    return int(np.sign(value))


def _judge(ctx: Dict[str, Any], a: List[Dict[str, Any]], b: List[Dict[str, Any]], n_boot: int) -> Dict[str, Any]:
    """Every test's call on group ``b`` against group ``a`` (+1 b better, -1 worse, 0 none) and the sign of
    the same difference on the holdout rows."""
    y, d, h = ctx["y"], ctx["decide"], ctx["hold"]
    pa, pb = [ctx["load"](item) for item in a], [ctx["load"](item) for item in b]
    ma, mb = np.mean(pa, axis=0), np.mean(pb, axis=0)
    out: Dict[str, Any] = {"a": [item["entry"] + f"__s{item['seed']}" for item in a],
                           "b": [item["entry"] + f"__s{item['seed']}" for item in b]}

    one = delong(y[d], pb[0][d], pa[0][d])
    out["delong_single"] = {"z": one["z"], "call": _sign(one["z"]) if one["p"] < 0.05 else 0,
                            "holdout": _sign(roc_auc_score(y[h], pb[0][h]) - roc_auc_score(y[h], pa[0][h]))}
    hold = _sign(roc_auc_score(y[h], mb[h]) - roc_auc_score(y[h], ma[h]))
    mean = delong(y[d], mb[d], ma[d])
    out["delong_mean"] = {"z": mean["z"], "call": _sign(mean["z"]) if mean["p"] < 0.05 else 0, "holdout": hold}

    ra = np.array([roc_auc_score(y[d], p[d]) for p in pa])
    rb = np.array([roc_auc_score(y[d], p[d]) for p in pb])
    t = ttest_ind(rb, ra, equal_var=False)
    t_call = _sign(rb.mean() - ra.mean()) if np.isfinite(t.pvalue) and t.pvalue < 0.05 else 0
    out["t_test"] = {"z": float(t.statistic), "call": t_call, "holdout": hold}

    se = np.sqrt(mean["se"] ** 2 + ra.var(ddof=1) / len(ra) + rb.var(ddof=1) / len(rb))
    z = (mean["auc_a"] - mean["auc_b"]) / se if se > 0 else 0.0
    out["formula"] = {"z": float(z), "call": _sign(z) if abs(z) > Z_LIKELY else 0, "holdout": hold}

    v = grade_groups(ctx["record"], y, ctx["codes"], {"seeds": [i["seed"] for i in a], "per_seed": pa, "mean": ma},
                     {"seeds": [i["seed"] for i in b], "per_seed": pb, "mean": mb}, n_boot=n_boot)
    lost = sum(x < 0 for x in v["chunk_diffs"])
    if v["grade"] in ("likely", "proven", "suspicious"):
        call = 1
    elif v["gain"] < -Z_LIKELY * v["se_total"] and lost >= ceil(LIKELY_SHARE * len(v["chunk_diffs"])):
        call = -1
    else:
        call = 0
    out["referee"] = {"z": v["gain"] / v["se_total"] if v["se_total"] else 0.0, "call": call,
                      "holdout": _sign(v["holdout_gain"]), "grade": v["grade"]}
    return out


def _disjoint_pairs(rng, pool: List[Dict[str, Any]], size: int, n: int):
    seen, pairs = set(), []
    for _ in range(50 * n):
        if len(pairs) >= n or len(pool) < 2 * size:
            break
        drawn = rng.permutation(len(pool))[: 2 * size]
        a, b = tuple(sorted(drawn[:size])), tuple(sorted(drawn[size:]))
        if (min(a, b), max(a, b)) in seen:
            continue
        seen.add((min(a, b), max(a, b)))
        pairs.append(([pool[i] for i in a], [pool[i] for i in b]))
    return pairs


def _summary(rows: List[Dict[str, Any]], truth: int) -> Dict[str, Any]:
    out = {}
    for test in TESTS:
        calls = np.array([r[test]["call"] for r in rows])
        agree = [r[test]["holdout"] == r[test]["call"] for r in rows if r[test]["call"] != 0]
        item = {"n": len(rows), "called": float(np.mean(calls != 0)) if rows else None,
                "holdout_agrees": float(np.mean(agree)) if agree else None}
        if truth:
            item["right"] = float(np.mean(calls == truth)) if rows else None
            item["wrong_way"] = float(np.mean(calls == -truth)) if rows else None
        out[test] = item
    return out


def study(contest: str, design: Dict[str, Any] = DESIGN, n: int = 20, size: int = 3, n_boot: int = 300,
          outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    record = contests.load_contest(contest, outputs_root)
    if record["task"] != "binary":
        raise ValueError("the power study uses AUC tests, so it needs a binary contest")
    labels, codes = contests.labels_and_rows(contest, outputs_root)
    found = entries(contest, outputs_root)
    cache: Dict[str, np.ndarray] = {}

    def load(item):
        if item["predictions_path"] not in cache:
            cache[item["predictions_path"]] = np.load(item["predictions_path"])
        return cache[item["predictions_path"]]

    def pool(names):
        missing = [name for name in names if name not in found]
        if missing:
            raise KeyError(f"no finished entries named {missing} in contest {contest!r}")
        return sorted((item for name in names for item in found[name]), key=lambda i: (i["entry"], i["seed"]))

    ctx = {"record": record, "y": _truth(record, labels), "codes": codes, "load": load,
           "decide": codes == contests.DECIDE, "hold": codes == contests.HOLDOUT}
    rng = np.random.default_rng(DRAW_SEED)
    same = {key: [_judge(ctx, a, b, n_boot) for a, b in _disjoint_pairs(rng, pool(names), size, n)]
            for key, names in design["same"].items()}
    equal = [_judge(ctx, pool([x])[:size], pool([w])[:size], n_boot)
             for x, w in itertools.combinations(design["equal"], 2)]
    planted = {}
    for worse, better in design["planted"].items():
        low, high, rows, seen = pool([worse]), pool(better), [], set()
        for _ in range(50 * n):
            if len(rows) >= n:
                break
            a = tuple(sorted(rng.choice(len(low), size, replace=False)))
            b = tuple(sorted(rng.choice(len(high), size, replace=False)))
            if (a, b) not in seen:
                seen.add((a, b))
                rows.append(_judge(ctx, [low[i] for i in a], [high[i] for i in b], n_boot))
        planted[worse] = rows

    same_rows = [r for rows in same.values() for r in rows]
    planted_rows = [r for rows in planted.values() for r in rows]
    result = {
        "contest": contest, "size": size, "decide_rows": int(ctx["decide"].sum()),
        "summary": {"same_settings": _summary(same_rows, 0), "different_but_equal": _summary(equal, 0),
                    "planted": _summary(planted_rows, 1),
                    "planted_by_entry": {k: _summary(rows, 1) for k, rows in planted.items()}},
        "detail": {"same_settings": same, "different_but_equal": equal, "planted": planted},
        "rule": {"t_test_confirmed_at_least": CONFIRMED, "t_test_not_confirmed_below": NOT_CONFIRMED,
                 "referee_at_most": REFEREE_AT_MOST},
    }
    eq = result["summary"]["different_but_equal"]
    t_rate, r_rate = eq["t_test"]["called"], eq["referee"]["called"]
    if t_rate is None:
        result["conclusion"] = "no different-but-equal pairs"
    elif t_rate >= CONFIRMED and r_rate <= REFEREE_AT_MOST:
        result["conclusion"] = "confirmed: the referee's row noise avoids false calls the t-test makes"
    elif t_rate < NOT_CONFIRMED:
        result["conclusion"] = "not confirmed: a t-test on per-run scores is enough for this tool"
    else:
        result["conclusion"] = "inconclusive"
    keys = [("same_settings", "called"), ("different_but_equal", "called"), ("planted", "right"),
            ("planted", "wrong_way")]
    gaps = [abs(result["summary"][k]["formula"][m] - result["summary"][k]["referee"][m]) for k, m in keys
            if result["summary"][k]["formula"][m] is not None]
    gaps += [abs(s["formula"][m] - s["referee"][m]) for s in result["summary"]["planted_by_entry"].values()
             for m in ("right", "wrong_way")]
    result["formula_vs_referee_max_gap"] = float(max(gaps)) if gaps else None
    return result


def _table(result: Dict[str, Any]) -> str:
    s = result["summary"]
    lines = [f"{'test':<14}{'same':>7}{'equal':>7}{'right':>7}{'wrong':>7}{'holdout':>9}"]
    for test in TESTS:
        hold = s["planted"][test]["holdout_agrees"]
        same, equal = s["same_settings"][test]["called"], s["different_but_equal"][test]["called"]
        lines.append(f"{test:<14}{same:>7.0%}{equal:>7.0%}"
                     f"{s['planted'][test]['right']:>7.0%}{s['planted'][test]['wrong_way']:>7.0%}"
                     f"{(f'{hold:.0%}' if hold is not None else '-'):>9}")
    lines.append("same/equal: share of null comparisons called; right/wrong: planted gains called the right or "
                 "wrong way; holdout: planted calls the holdout rows agree with")
    for entry, by in s["planted_by_entry"].items():
        lines.append(f"  {entry:<18}" + "  ".join(f"{t} {by[t]['right']:.0%}/{by[t]['wrong_way']:.0%}"
                                                  for t in TESTS))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contest", default="ieee-100k")
    parser.add_argument("--skip-runs", action="store_true", help="analyse the entries that exist")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.skip_runs:
        make_entries(args.contest)
    result = study(args.contest, n=args.n)
    out = contests.referee_root() / "benchmark" / f"{args.contest}-power.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(_table(result))
    print(f"conclusion: {result['conclusion']}; formula vs referee, largest gap "
          f"{result['formula_vs_referee_max_gap']:.0%}\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
