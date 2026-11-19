"""Leak checks on a contest's data, run by the referee (which may read the hidden labels).

A contestant never sees the hidden labels, so leaks inside its own pipeline cannot raise its referee
score; they only inflate what it claims about itself (``verdict.entry_summary``'s optimism). What can
raise its referee score without being real is the data itself:

- duplicates: hidden rows that also appear, features and all, among the training rows;
- one column that predicts the label almost alone (often a field recorded after the outcome);
- an id that carries signal on the hidden rows (ids that track time or the label, not the case).

Each check reports what it measured and raises a flag above a fixed threshold. A flag means "explain
this before trusting a gain", not "reject".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.genML.referee.contest import contest_dir, labels_and_rows, load_contest

DUPLICATE_SHARE = 0.01        # flag when more than 1% of hidden rows also appear among training rows
# Flag a column whose AUC alone (or |Spearman| for regression) exceeds this. It was 0.97 at first; the
# 2026-11-18 benchmark's planted leak scored 0.95 and passed, so it is 0.90 until another dataset tests it.
ALONE_STRENGTH = 0.90
ID_STRENGTH = 0.65            # flag an id whose AUC alone on the hidden rows exceeds this
MAX_ROWS = 200_000
SEED = 20261120


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    """AUC of ``x`` for a 0/1 ``y``, folded so 0.5 is no signal and 1.0 is perfect either way."""
    from scipy.stats import rankdata

    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(x)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))


def _strength(y: np.ndarray, x: np.ndarray, task: str) -> float:
    if task == "binary":
        return _auc(y, x)
    from scipy.stats import spearmanr

    value = spearmanr(x, y).statistic
    return float(abs(value)) if np.isfinite(value) else 0.0


def _numeric(column: pd.Series, y: np.ndarray, task: str) -> np.ndarray:
    """A column as numbers: numeric values (missing as a value below all others), or for text and
    categories the out-of-fold target mean of each level (so a level seen once tells nothing)."""
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        values = column.to_numpy(dtype=float)
        low = np.nanmin(values) - 1 if np.isfinite(values).any() else -1.0
        return np.where(np.isfinite(values), values, low)
    codes = pd.factorize(column.astype(str))[0]
    folds = np.random.default_rng(SEED).permutation(len(codes)) % 5
    out = np.empty(len(codes))
    prior = float(np.mean(y))
    for k in range(5):
        fit, apply = folds != k, folds == k
        sums = np.bincount(codes[fit], weights=y[fit], minlength=codes.max() + 1)
        counts = np.bincount(codes[fit], minlength=codes.max() + 1)
        out[apply] = ((sums + 10 * prior) / (counts + 10))[codes[apply]]
    return out


def check_contest(contest: str, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    record = load_contest(contest, outputs_root)
    public = contest_dir(contest, outputs_root) / "public"
    train = pd.read_parquet(public / "train.parquet")
    hidden = pd.read_parquet(public / "hidden.parquet")
    labels, _ = labels_and_rows(contest, outputs_root)
    task, target, id_column = record["task"], record["target"], record.get("id_column")
    if task == "binary":
        positive = record["classes"][1]
        y_train, y_hidden = (train[target] == positive).to_numpy(int), (labels == positive).astype(int)
    elif task == "regression":
        y_train, y_hidden = train[target].to_numpy(float), labels.astype(float)
    else:
        y_train = y_hidden = None
    features = [c for c in train.columns if c not in (target, id_column)]
    flags: List[str] = []
    result: Dict[str, Any] = {"contest": contest}

    # Duplicates: hidden rows whose features all match a training row
    key_train = pd.util.hash_pandas_object(train[features], index=False)
    key_hidden = pd.util.hash_pandas_object(hidden[features], index=False)
    duplicated = key_hidden.isin(set(key_train.to_numpy()))
    share = float(duplicated.mean())
    result["duplicates"] = {"hidden_rows_also_in_train": int(duplicated.sum()), "share": share}
    if share > DUPLICATE_SHARE:
        flags.append(f"{share:.1%} of hidden rows also appear among the training rows")

    # Single columns that predict the label almost alone
    if y_train is not None:
        rows = np.arange(len(train))
        if len(rows) > MAX_ROWS:
            rows = np.sort(np.random.default_rng(SEED).choice(rows, MAX_ROWS, replace=False))
        sub, y_sub = train.iloc[rows], y_train[rows]
        strengths = {c: _strength(y_sub, _numeric(sub[c], y_sub, task), task) for c in features}
        top = sorted(strengths.items(), key=lambda kv: -kv[1])[:5]
        result["strongest_columns"] = [{"column": c, "strength": round(s, 4)} for c, s in top]
        for column, value in top:
            if value > ALONE_STRENGTH:
                flags.append(f"column {column!r} predicts the label almost alone (strength {value:.3f})")

    # An id that carries signal on the hidden rows
    if id_column and id_column in hidden.columns and y_hidden is not None:
        value = _strength(y_hidden, _numeric(hidden[id_column], y_hidden, task), task)
        result["id_signal"] = {"column": id_column, "strength": round(value, 4)}
        if value > ID_STRENGTH:
            flags.append(f"the id column {id_column!r} carries signal on the hidden rows (strength {value:.3f})")
    result["flags"] = flags
    return result
