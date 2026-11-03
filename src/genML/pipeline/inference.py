"""Shared inference heuristics used when a problem spec leaves a field unset.

Every function returns its decision together with a human-readable reason so
the resolved problem spec can record why each guess was made.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ID_NAME_PATTERNS = {"id", "ID", "Id", "index", "Index"}


def infer_task_type(y: Sequence) -> Tuple[str, str]:
    """Infer 'binary', 'multiclass' or 'regression' from target values.

    Follows AutoGluon's ``infer_problem_type`` rules: two distinct values are
    binary; non-numeric targets are multiclass; non-integer floats are
    regression; integer targets are multiclass only when they have few distinct
    values relative to the number of rows.
    """
    series = pd.Series(y)
    series = series[~series.isna()]
    if pd.api.types.is_numeric_dtype(series):
        series = series[np.isfinite(series.astype(float))]
    if len(series) == 0:
        return "binary", "empty target; defaulting to binary"

    n_rows = len(series)
    n_unique = int(series.nunique())

    if n_unique == 2:
        return "binary", "exactly 2 distinct target values"

    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "multiclass", f"non-numeric target with {n_unique} classes"

    values = series.astype(float).to_numpy()
    if not np.allclose(values, np.round(values)):
        return "regression", "target has non-integer values"

    threshold = 0.05 if n_rows > 1000 else 0.1
    unique_ratio = n_unique / n_rows
    if (unique_ratio <= threshold and n_unique <= 1000) or n_unique <= 20:
        return (
            "multiclass",
            f"integer target with {n_unique} distinct values (ratio {unique_ratio:.3f})",
        )
    return (
        "regression",
        f"integer target with many distinct values ({n_unique}, ratio {unique_ratio:.3f})",
    )


def infer_target_column(
    train_columns: Iterable[str],
    test_columns: Iterable[str],
    candidates: Optional[List[str]] = None,
) -> Tuple[Optional[str], str]:
    """Pick the target as the column present in train but not in test.

    Order follows the train column order so the choice is deterministic. When
    several columns qualify, the last one wins (Kaggle puts the target last).
    """
    test_set = set(test_columns)
    train_only = [col for col in train_columns if col not in test_set and col not in ID_NAME_PATTERNS]
    if len(train_only) == 1:
        return train_only[0], "only column present in train but not test"
    if len(train_only) > 1:
        return (
            train_only[-1],
            f"last of several train-only columns {train_only}; declare target in the spec",
        )
    if candidates:
        return candidates[0], "first analyzer target candidate (no train-only column)"
    return None, "no target could be inferred"
