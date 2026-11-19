"""AutoGluon as a referee contestant. Runs in its own environment, not genML's:

    ~/.venvs/genml-autogluon/bin/python contestants/autogluon_contestant.py --presets medium_quality

The referee adds --contest, --out, --seed, --workdir and --time-limit. Reads public/train.parquet and
public/hidden.parquet, fits a TabularPredictor on the training rows (the id column dropped, everything
else as given), writes predictions for the hidden rows and self_report.json with AutoGluon's own
validation score on the metric's own scale.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# genML metric name -> AutoGluon eval_metric (normalized Gini ranks like AUC)
METRICS = {"roc_auc": "roc_auc", "normalized_gini": "roc_auc", "accuracy": "accuracy", "log_loss": "log_loss",
           "rmse": "root_mean_squared_error", "mae": "mean_absolute_error", "r2": "r2", "f1": "f1"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--drop", nargs="*", default=[], help="extra columns to leave out")
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--time-validation", type=float, default=None,
                        help="validate on this latest share of the training rows (by the contest's time column) "
                             "and leave the time column out, as a careful user of time-ordered data would")
    args = parser.parse_args()

    from autogluon.tabular import TabularPredictor

    public, work = Path(args.contest), Path(args.workdir)
    facts = json.loads((public / "contest.json").read_text())
    target, task = facts["target"], facts["task"]
    train = pd.read_parquet(public / "train.parquet")
    hidden = pd.read_parquet(public / "hidden.parquet")
    drop = [c for c in [facts.get("id_column"), *args.drop] if c and c in train.columns]
    train, hidden = train.drop(columns=drop), hidden.drop(columns=[c for c in drop if c in hidden.columns])
    problem_type = {"binary": "binary", "multiclass": "multiclass", "regression": "regression"}[task]
    tuning = None
    time_column = facts.get("time_column")
    if args.time_validation and time_column in train.columns:
        train = train.sort_values(time_column, kind="stable")
        cut = int(len(train) * (1 - args.time_validation))
        train, tuning = train.iloc[:cut].drop(columns=[time_column]), train.iloc[cut:].drop(columns=[time_column])
        hidden = hidden.drop(columns=[time_column])
        drop.append(time_column)
    model_dir = work / "autogluon"
    predictor = TabularPredictor(label=target, problem_type=problem_type, path=str(model_dir),
                                 eval_metric=METRICS.get(facts["metric"]), verbosity=2,
                                 learner_kwargs={"random_state": args.seed})
    predictor.fit(train, tuning_data=tuning, presets=args.presets, time_limit=args.time_limit,
                  ag_args_ensemble={"model_random_seed": args.seed})
    if task == "binary":
        proba = predictor.predict_proba(hidden)
        pred = proba[facts["classes"][1]].to_numpy(dtype=float)
    elif task == "multiclass":
        proba = predictor.predict_proba(hidden)
        pred = proba[facts["classes"]].to_numpy(dtype=float)
    else:
        pred = predictor.predict(hidden).to_numpy(dtype=float)
    np.save(args.out, pred)
    board = predictor.leaderboard(silent=True)
    best = float(board.iloc[0]["score_val"])
    report = {"validation_score": float(predictor.eval_metric.convert_score_to_original(best)),
              "metric": predictor.eval_metric.name, "best_model": str(board.iloc[0]["model"]),
              "how": "AutoGluon's own validation score of its best model"
                     + (" on the latest training rows" if tuning is not None else ""), "presets": args.presets,
              "dropped": drop}
    (work / "self_report.json").write_text(json.dumps(report, indent=2))
    if not args.keep_models:
        shutil.rmtree(model_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
