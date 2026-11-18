"""Leak-safe categorical encodings used by top Playground solutions.

Enabled per problem through ``feature_engineering.encodings`` in the spec::

    feature_engineering:
      encodings:
        target_encode: auto          # or a list of columns
        count_encode: auto           # or a list of columns
        pairs: auto                  # or [[a, b], ...] or false
        numeric_as_categorical: auto # numeric columns with <= max_unique values; all = every numeric
                                     # column, whatever its cardinality (per-value target encoding)
        max_unique: 50
        max_pairs: 28
        smoothing: 20
        digits: auto                 # or a list of columns, or false (default)
        digit_keys: [int, int10, tenths, cents, lead1, lead2]
        original:                    # optional: the dataset a Playground set was generated from
          path: datasets/original/fraud.csv
          target: is_fraud           # its target column (numeric or two classes)

        nested: true                 # default; per-fold target encodings (see below)

Digit keys describe how a decimal number is written, not its size: its integer
part, the integer part of a tenth of it, the value to one decimal, its cents
(two decimals), and its leading one or two digits together with its number of
digits. Synthetic generators often leave signal there (some decimal digits run
"hot"). Each key is a categorical that is target-encoded like any other, nested
per fold. ``digits: auto`` uses every numeric column with fractional values.

Target encodings for training rows are computed out-of-fold with the problem's
shared folds, so a row's own target never leaks into its feature. Test rows use
statistics from all training rows. Count encodings use train and test together
(no target involved).

That single out-of-fold matrix still leaks across folds during cross-validation:
when fold k is the validation fold, the training rows' encodings were computed
partly from fold k's labels. ``nested_target_encodings`` therefore also builds one
block per outer fold k: training rows are encoded with inner folds that exclude
fold k, and fold k's rows and the test rows with statistics from the other folds
only. Training swaps these blocks in per fold (``pipeline/oof.FoldFeatures``); the
single matrix is kept for refitting on all rows.
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
    "nested": True,
    "digits": False,
    "digit_keys": None,  # None = all of DIGIT_KINDS
}

DIGIT_KINDS = ("int", "int10", "tenths", "cents", "lead1", "lead2")
_DIGIT_MISSING = np.iinfo(np.int64).min  # key for missing or huge values


def digit_key(values: pd.Series, kind: str) -> np.ndarray:
    """One digit key of a numeric column, as integers (see the module docstring).

    The value is first rounded to cents: 1371.37 -> int 1371, int10 137, tenths
    13713, cents 37, lead1 41 (4 digits, leading 1), lead2 413 (4 digits, leading 13).
    Negative values carry a negative key (cents excepted); missing values share one key.
    """
    if kind not in DIGIT_KINDS:
        raise ValueError(f"Unknown digit key '{kind}' (known: {', '.join(DIGIT_KINDS)})")
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v) & (np.abs(v) < 1e13)
    units = np.abs(np.rint(np.where(ok, v, 0.0) * 100)).astype(np.int64)
    sign = np.where(np.where(ok, v, 0.0) < 0, -1, 1).astype(np.int64)
    integer = units // 100
    if kind == "int":
        out = sign * integer
    elif kind == "int10":
        out = sign * (units // 1000)
    elif kind == "tenths":
        out = sign * (units // 10)
    elif kind == "cents":
        out = units % 100
    else:
        n_digits = np.char.str_len(integer.astype(str)).astype(np.int64)
        if kind == "lead1":
            out = sign * (n_digits * 10 + integer // 10 ** (n_digits - 1))
        else:
            out = sign * (n_digits * 100 + integer // 10 ** np.maximum(n_digits - 2, 0))
    out = out.astype(np.int64)
    out[~ok] = _DIGIT_MISSING
    return out


class DigitKey(tuple):
    """(column, kind): a digit key used as an encoded column (see ``_keys``)."""

    def __new__(cls, column: str, kind: str):
        return super().__new__(cls, (column, kind))


def _digit_columns(train_df: pd.DataFrame, numeric_columns: List[str], available: Sequence[str],
                   option) -> List[str]:
    """Columns that get digit keys: numeric columns with fractional values (auto) or a list."""
    if option in (None, False):
        return []
    if option in (True, "auto"):
        columns = []
        for c in numeric_columns:
            if c not in available or not pd.api.types.is_numeric_dtype(train_df[c]) \
                    or pd.api.types.is_bool_dtype(train_df[c]):
                continue
            values = train_df[c].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) and np.any(values != np.round(values)):
                columns.append(c)
        return columns
    return [c for c in option if c in available]


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


def _codes(train_keys: pd.Series, test_keys: pd.Series) -> Tuple[np.ndarray, np.ndarray, int]:
    """Integer level of every key, shared by train and test (missing keys form one level)."""
    codes, uniques = pd.factorize(pd.concat([train_keys, test_keys], ignore_index=True))
    codes = codes.astype(np.int64)
    n_levels = len(uniques) + 1
    codes[codes < 0] = n_levels - 1
    return codes[:len(train_keys)], codes[len(train_keys):], n_levels


def _level_means(codes: np.ndarray, targets: np.ndarray, n_levels: int, prior: np.ndarray,
                 smoothing: float) -> np.ndarray:
    """Smoothed target mean of every level (levels without rows get the prior)."""
    counts = np.bincount(codes, minlength=n_levels).astype(float)
    sums = np.stack([np.bincount(codes, weights=targets[:, j], minlength=n_levels)
                     for j in range(targets.shape[1])], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (sums + prior * smoothing) / (counts[:, None] + smoothing)
    means[counts == 0] = prior
    return means


def target_encode(train_keys: pd.Series, test_keys: pd.Series, y: np.ndarray, fold_ids: np.ndarray,
                  task: str, n_classes: int, smoothing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold target encoding for train, full-train encoding for test."""
    targets = _target_matrix(y, task, n_classes)
    train_codes, test_codes, n_levels = _codes(train_keys, test_keys)
    train_out = np.zeros((len(train_keys), targets.shape[1]))
    for fold in range(int(fold_ids.max()) + 1):
        fit_mask = fold_ids != fold
        prior = targets[fit_mask].mean(axis=0)
        means = _level_means(train_codes[fit_mask], targets[fit_mask], n_levels, prior, smoothing)
        train_out[~fit_mask] = means[train_codes[~fit_mask]]
    prior = targets.mean(axis=0)
    test_out = _level_means(train_codes, targets, n_levels, prior, smoothing)[test_codes]
    return train_out, test_out


def nested_target_encode(train_keys: pd.Series, test_keys: pd.Series, y: np.ndarray, fold_ids: np.ndarray,
                         task: str, n_classes: int, smoothing: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per outer fold k: (train encodings, test encodings) that never use fold k's labels
    for rows outside fold k, and never use a row's own label.

    Rows in another fold j are encoded from folds other than j and k; rows in fold k
    and the test rows from all folds other than k (the rows the fold-k model trains on).
    """
    targets = _target_matrix(y, task, n_classes)
    train_codes, test_codes, n_levels = _codes(train_keys, test_keys)
    n_folds = int(fold_ids.max()) + 1
    blocks = []
    for outer in range(n_folds):
        outer_fit = fold_ids != outer
        prior = targets[outer_fit].mean(axis=0)
        means = _level_means(train_codes[outer_fit], targets[outer_fit], n_levels, prior, smoothing)
        train_out = np.empty((len(train_keys), targets.shape[1]), dtype=np.float32)
        train_out[~outer_fit] = means[train_codes[~outer_fit]]
        test_out = means[test_codes].astype(np.float32)
        for inner in range(n_folds):
            if inner == outer:
                continue
            inner_fit = outer_fit & (fold_ids != inner)
            inner_prior = targets[inner_fit].mean(axis=0)
            inner_means = _level_means(train_codes[inner_fit], targets[inner_fit], n_levels, inner_prior,
                                       smoothing)
            rows = fold_ids == inner
            train_out[rows] = inner_means[train_codes[rows]]
        blocks.append((train_out, test_out))
    return blocks


def _keys(frame: pd.DataFrame, column) -> pd.Series:
    """Keys for a column, a digit key of a column, or a pair of columns joined with '|'.

    Numeric and boolean columns are used as they are (equal values, equal keys);
    other columns as strings.
    """
    if isinstance(column, DigitKey):
        return pd.Series(digit_key(frame[column[0]], column[1]), index=frame.index)
    if isinstance(column, tuple):
        return frame[column[0]].astype(str) + "|" + frame[column[1]].astype(str)
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str)


def _encoded_columns(report: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """(feature base name, source column, pair or digit key) for every target-encoded key."""
    return ([(c, c) for c in report["target_encoded"]]
            + [(f"{a}__{b}", (a, b)) for a, b in report["pairs"]]
            + [(f"{c}__{kind}", DigitKey(c, kind)) for c, kind in report.get("digit_keys", [])])


def nested_target_encodings(train_df: pd.DataFrame, test_df: pd.DataFrame, y: np.ndarray, fold_ids: np.ndarray,
                            task: str, n_classes: int, report: Dict[str, Any],
                            options: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Per-fold target-encoding blocks for the columns ``build_encodings`` encoded.

    Returns ``{"columns": [feature names], "train": [array per fold], "test": [array per fold]}``
    with columns in the same order and names as the single matrix, or None when there
    is nothing to nest (no target encodings, or ``nested: false``).
    """
    opts = {**DEFAULTS, **(options or {})}
    encoded = _encoded_columns(report)
    if not opts["nested"] or not encoded:
        return None
    width = n_classes if task == "multiclass" else 1
    n_folds = int(fold_ids.max()) + 1
    total = width * len(encoded)
    train_blocks = [np.empty((len(train_df), total), dtype=np.float32) for _ in range(n_folds)]
    test_blocks = [np.empty((len(test_df), total), dtype=np.float32) for _ in range(n_folds)]
    names: List[str] = []
    for position, (name, column) in enumerate(encoded):
        blocks = nested_target_encode(_keys(train_df, column), _keys(test_df, column), y, fold_ids,
                                      task, n_classes, float(opts["smoothing"]))
        names.extend([f"{name}_te"] if width == 1 else [f"{name}_te{k}" for k in range(width)])
        columns = slice(position * width, (position + 1) * width)
        for fold, (train_block, test_block) in enumerate(blocks):
            train_blocks[fold][:, columns] = train_block
            test_blocks[fold][:, columns] = test_block
    return {"columns": names, "train": train_blocks, "test": test_blocks}


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
    if opts["numeric_as_categorical"] == "all":
        numeric_cats = [c for c in numeric_columns if c in available]
    elif opts["numeric_as_categorical"] in (True, "auto"):
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

    kinds = list(opts["digit_keys"] or DIGIT_KINDS)
    unknown = [k for k in kinds if k not in DIGIT_KINDS]
    if unknown:
        raise ValueError(f"Unknown digit_keys {unknown} (known: {', '.join(DIGIT_KINDS)})")
    digit_keys: List[List[str]] = []
    for column in _digit_columns(train_df, numeric_columns, available, opts["digits"]):
        for kind in kinds:
            if len(np.unique(digit_key(train_df[column], kind))) > 1:  # constant keys say nothing
                digit_keys.append([column, kind])

    train_feats: Dict[str, np.ndarray] = {}
    test_feats: Dict[str, np.ndarray] = {}
    columns_to_encode = _encoded_columns({"target_encoded": te_columns, "pairs": pairs,
                                          "digit_keys": digit_keys})
    count_targets = [(c, c) for c in count_columns] + [(f"{a}__{b}", (a, b)) for a, b in pairs]

    for name, column in columns_to_encode:
        train_keys, test_keys = _keys(train_df, column), _keys(test_df, column)
        tr, te = target_encode(train_keys, test_keys, y, fold_ids, task, n_classes, float(opts["smoothing"]))
        for k in range(tr.shape[1]):
            suffix = "_te" if tr.shape[1] == 1 else f"_te{k}"
            train_feats[f"{name}{suffix}"] = tr[:, k].astype(np.float32)
            test_feats[f"{name}{suffix}"] = te[:, k].astype(np.float32)

    for name, column in count_targets:
        train_codes, test_codes, n_levels = _codes(_keys(train_df, column), _keys(test_df, column))
        counts = np.bincount(np.concatenate([train_codes, test_codes]), minlength=n_levels).astype(np.float32)
        train_feats[f"{name}_count"] = counts[train_codes]
        test_feats[f"{name}_count"] = counts[test_codes]

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
        "digit_keys": digit_keys,
        "nested": bool(opts["nested"]) and bool(columns_to_encode),
        "n_features": len(train_feats),
    }
    logger.info("Encodings: %d target-encoded, %d count-encoded, %d pairs, %d digit keys -> %d features",
                len(te_columns), len(count_columns), len(pairs), len(digit_keys), len(train_feats))
    return (pd.DataFrame(train_feats, index=train_df.index),
            pd.DataFrame(test_feats, index=test_df.index), report)
