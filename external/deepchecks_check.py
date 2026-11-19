"""Deepchecks' train-test validation and data integrity suites on a referee contest, for comparison
with the referee's checks.

Runs in its own environment (Deepchecks 0.19 needs NumPy < 2, scikit-learn < 1.6, pandas < 2.3):

    ~/.venvs/genml-deepchecks/bin/python external/deepchecks_check.py --contest-dir DIR --out result.json

The referee runs it (trusted, unsandboxed): it reads the contest's private labels so Deepchecks gets
labelled train and test sets, as a person validating a model would give it. The id column is passed as
the index and the time column as the datetime, so Deepchecks' own leakage checks for them can run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contest-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity, train_test_validation

    folder = Path(args.contest_dir)
    facts = json.loads((folder / "public" / "contest.json").read_text())
    target, id_column, time_column = facts["target"], facts.get("id_column"), facts.get("time_column")
    train = pd.read_parquet(folder / "public" / "train.parquet")
    test = pd.read_parquet(folder / "public" / "hidden.parquet")
    test[target] = np.load(folder / "private" / "labels.npy", allow_pickle=True)
    cat = [c for c in train.columns if c not in (target, id_column, time_column)
           and not pd.api.types.is_numeric_dtype(train[c])]
    kwargs = {"label": target, "cat_features": cat}
    if id_column:
        kwargs["index_name"] = id_column
    if time_column:
        kwargs["datetime_name"] = time_column
        for frame in (train, test):              # seconds from a reference time, as a datetime for Deepchecks
            frame[time_column] = pd.to_datetime(frame[time_column], unit="s")
    started = time.time()
    train_ds, test_ds = Dataset(train, **kwargs), Dataset(test, **kwargs)
    failed, errors = [], []
    for suite, result in (("train_test_validation", train_test_validation().run(train_ds, test_ds)),
                          ("data_integrity", data_integrity().run(train_ds))):
        for check_result in result.get_not_passed_checks():
            for condition in check_result.conditions_results:
                if not condition.is_pass():
                    failed.append({"suite": suite, "check": check_result.get_header(), "condition": condition.name,
                                   "details": str(condition.details)[:300]})
        errors += [{"suite": suite, "check": str(getattr(r, "check", r))[:120],
                    "error": str(getattr(r, "exception", ""))[:200]}
                   for r in result.failures if not hasattr(r, "conditions_results")]
    Path(args.out).write_text(json.dumps({"failed": failed, "errors": errors,
                                          "seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
