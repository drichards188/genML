"""Leak-safe categorical encodings used by top Playground solutions.

Enabled per problem through ``feature_engineering.encodings`` in the spec::

    feature_engineering:
      encodings:
        target_encode: auto          # or a list of columns
        count_encode: auto           # or a list of columns
        pairs: auto                  # or [[a, b], ...] or false
        numeric_as_categorical: auto # numeric columns with <= max_unique values
        max_unique: 50
        max_pairs: 28
        smoothing: 20
        original:                    # optional: the dataset a Playground set was generated from
          path: datasets/original/fraud.csv
          target: is_fraud           # its target column (numeric or two classes)

Target encodings for training rows are computed out-of-fold with the problem's
shared folds, so a row's own target never leaks into its feature. Test rows use
statistics from all training rows. Count encodings use train and test together
(no target involved).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULTS = {
    "target_encode": None,
    "count_encode": None,
    "pairs": False,
    "numeric_as_categorical": None,
    "max_unique": 50,
    "max_pairs": 28,
    "smoothing": 20.0,
    "original": None,
}


def _resolve_columns(option, auto_columns: List[str], available: Sequence[str]) -> List[str]:
    if option in (None, False):
        return []
    if option in (True, "auto"):
        return [c for c in auto_columns if c in available]
    return [c for c in option if c in available]


def _target_matrix(y: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    """Columns to average: y itself, or one-hot class indicators for multiclass."""
    if task == "multiclass":
        return np.eye(n_classes)[y.astype(int)]
    return y.astype(float).reshape(-1, 1)


def _smoothed_means(keys: pd.Series, targets: np.ndarray, prior: np.ndarray, smoothing: float) -> pd.DataFrame:
    frame = pd.DataFrame(targets, index=keys.index)
    frame["_key"] = keys.to_numpy()
    grouped = frame.groupby("_key", sort=False)
    sums, counts = grouped.sum(), grouped.size()
    return (sums.add(prior * smoothing, axis=1)).div(counts + smoothing, axis=0)


def _apply_means(keys: pd.Series, means: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
    mapped = means.reindex(keys.to_numpy()).to_numpy()
    missing = np.isnan(mapped).any(axis=1)
    mapped[missing] = prior
    return mapped


def target_encode(train_keys: pd.Series, test_keys: pd.Series, y: np.ndarray, fold_ids: np.ndarray,
                  task: str, n_classes: int, smoothing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold target encoding for train, full-train encoding for test."""
    targets = _target_matrix(y, task, n_classes)
    train_out = np.zeros((len(train_keys), targets.shape[1]))
    for fold in range(int(fold_ids.max()) + 1):
        fit_mask = fold_ids != fold
        prior = targets[fit_mask].mean(axis=0)
        means = _smoothed_means(train_keys[fit_mask], targets[fit_mask], prior, smoothing)
        valid_idx = np.where(~fit_mask)[0]
        train_out[valid_idx] = _apply_means(train_keys.iloc[valid_idx], means, prior)
    prior = targets.mean(axis=0)
    test_out = _apply_means(test_keys, _smoothed_means(train_keys, targets, prior, smoothing), prior)
    return train_out, test_out


def _add_original_means(train_df, test_df, train_feats, test_feats, columns, original_opts, smoothing):
    """Map each value to its smoothed target mean in the original dataset (no leakage:
    the original rows are separate from train/test)."""
    from pathlib import Path

    from src.genML.pipeline.problem import PROJECT_ROOT

    path = Path(original_opts["path"])
    path = path if path.is_absolute() else PROJECT_ROOT / path
    original = pd.read_csv(path)
    target = original[original_opts["target"]]
    if not pd.api.types.is_numeric_dtype(target):
        classes = sorted(target.dropna().unique())
        if len(classes) != 2:
            raise ValueError("original.target must be numeric or have exactly two classes")
        target = (target == classes[1]).astype(float)
    values = target.to_numpy(dtype=float).reshape(-1, 1)
    prior = values.mean(axis=0)
    added = []
    for column in columns:
        if column not in original.columns:
            continue
        means = _smoothed_means(original[column].astype(str), values, prior, smoothing)
        train_feats[f"{column}_orig_mean"] = _apply_means(train_df[column].astype(str), means, prior)[:, 0]
        test_feats[f"{column}_orig_mean"] = _apply_means(test_df[column].astype(str), means, prior)[:, 0]
        added.append(column)
    return added


def build_encodings(train_df: pd.DataFrame, test_df: pd.DataFrame, y: np.ndarray, fold_ids: np.ndarray,
                    task: str, n_classes: int, categorical_columns: List[str], numeric_columns: List[str],
                    options: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Return (train features, test features, report) for the configured encodings."""
    opts = {**DEFAULTS, **(options or {})}
    available = [c for c in train_df.columns if c in test_df.columns]

    numeric_cats = []
    if opts["numeric_as_categorical"] in (True, "auto"):
        numeric_cats = [c for c in numeric_columns if c in available
                        and train_df[c].nunique() <= int(opts["max_unique"])]
    elif opts["numeric_as_categorical"]:
        numeric_cats = [c for c in opts["numeric_as_categorical"] if c in available]
    auto_columns = list(dict.fromkeys(list(categorical_columns) + numeric_cats))

    te_columns = _resolve_columns(opts["target_encode"], auto_columns, available)
    count_columns = _resolve_columns(opts["count_encode"], auto_columns, available)

    pairs: List[Tuple[str, str]] = []
    if opts["pairs"] in (True, "auto"):
        pairs = list(itertools.combinations(auto_columns, 2))[: int(opts["max_pairs"])]
    elif opts["pairs"]:
        pairs = [tuple(pair) for pair in opts["pairs"] if all(c in available for c in pair)]

    def keys(frame: pd.DataFrame, column) -> pd.Series:
        if isinstance(column, tuple):
            return frame[column[0]].astype(str) + "|" + frame[column[1]].astype(str)
        return frame[column].astype(str)

    train_feats: Dict[str, np.ndarray] = {}
    test_feats: Dict[str, np.ndarray] = {}
    columns_to_encode = [(c, c) for c in te_columns] + [(f"{a}__{b}", (a, b)) for a, b in pairs]
    count_targets = [(c, c) for c in count_columns] + [(f"{a}__{b}", (a, b)) for a, b in pairs]

    for name, column in columns_to_encode:
        train_keys, test_keys = keys(train_df, column), keys(test_df, column)
        tr, te = target_encode(train_keys, test_keys, y, fold_ids, task, n_classes, float(opts["smoothing"]))
        for k in range(tr.shape[1]):
            suffix = "_te" if tr.shape[1] == 1 else f"_te{k}"
            train_feats[f"{name}{suffix}"] = tr[:, k].astype(np.float32)
            test_feats[f"{name}{suffix}"] = te[:, k].astype(np.float32)

    for name, column in count_targets:
        train_keys, test_keys = keys(train_df, column), keys(test_df, column)
        counts = pd.concat([train_keys, test_keys]).value_counts()
        train_feats[f"{name}_count"] = train_keys.map(counts).to_numpy(dtype=np.float32)
        test_feats[f"{name}_count"] = test_keys.map(counts).to_numpy(dtype=np.float32)

    original_columns: List[str] = []
    if opts["original"]:
        original_columns = _add_original_means(train_df, test_df, train_feats, test_feats, auto_columns,
                                               opts["original"], float(opts["smoothing"]))

    report = {
        "original_mean_encoded": original_columns,
        "target_encoded": te_columns,
        "count_encoded": count_columns,
        "pairs": [list(p) for p in pairs],
        "numeric_as_categorical": numeric_cats,
        "n_features": len(train_feats),
    }
    logger.info("Encodings: %d target-encoded, %d count-encoded, %d pairs -> %d features",
                len(te_columns), len(count_columns), len(pairs), len(train_feats))
    return (pd.DataFrame(train_feats, index=train_df.index),
            pd.DataFrame(test_feats, index=test_df.index), report)
