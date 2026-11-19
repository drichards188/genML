"""A fast, fixed LightGBM contestant for calibrating the referee (seconds on 100k rows).

    .venv/bin/python contestants/lightgbm_contestant.py [--drop COL ...] [--drop-prefix P ...] [--noise-columns K]

The referee adds --contest, --out, --seed, --workdir. No tuning: fixed parameters, so differences
between entries come only from the columns given and the seed. ``--drop``/``--drop-prefix`` withhold
columns (a baseline missing something useful); ``--noise-columns`` adds K columns of pure noise, the same
for every seed (a change that should not help); ``--noise-seed`` picks which noise (default 777), so two
draws are different models, equally good in general, that err on different rows. Text columns become
category codes shared by train and hidden rows. Uses no labels other than the training rows'.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8,
          "subsample_freq": 1, "colsample_bytree": 0.5, "n_estimators": 400, "verbose": -1}
NOISE_SEED = 777


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("--contest", "--out", "--workdir"):
        parser.add_argument(name, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", default=None)
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--drop-prefix", nargs="*", default=[])
    parser.add_argument("--noise-columns", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    args = parser.parse_args()

    import lightgbm as lgb

    public = Path(args.contest)
    facts = json.loads((public / "contest.json").read_text())
    target = facts["target"]
    train = pd.read_parquet(public / "train.parquet")
    hidden = pd.read_parquet(public / "hidden.parquet")
    y = train.pop(target)
    if facts["task"] == "binary":
        y = (y == facts["classes"][1]).astype(int)
    drop = {facts.get("id_column"), *args.drop}
    drop |= {c for c in train.columns for p in args.drop_prefix if c.startswith(p)}
    columns = [c for c in train.columns if c not in drop]
    both = pd.concat([train[columns], hidden[columns]], ignore_index=True)
    for column in columns:
        if not pd.api.types.is_numeric_dtype(both[column]) or pd.api.types.is_bool_dtype(both[column]):
            both[column] = pd.Categorical(both[column].astype(str)).codes.astype("int32")
    if args.noise_columns:
        rng = np.random.default_rng(args.noise_seed)
        for k in range(args.noise_columns):
            both[f"noise_{k}"] = rng.normal(size=len(both)).astype("float32")
    X_train, X_hidden = both.iloc[: len(train)], both.iloc[len(train):]
    params = {**PARAMS, "random_state": args.seed, "n_jobs": os.cpu_count()}
    if facts["task"] == "regression":
        model = lgb.LGBMRegressor(**params).fit(X_train, y)
        pred = model.predict(X_hidden)
    else:
        model = lgb.LGBMClassifier(**params).fit(X_train, y)
        pred = model.predict_proba(X_hidden)
        pred = pred[:, 1] if facts["task"] == "binary" else pred
    np.save(args.out, np.asarray(pred, dtype=float))
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    (Path(args.workdir) / "columns.json").write_text(json.dumps({"used": len(X_train.columns),
                                                                 "dropped": sorted(c for c in drop if c)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
