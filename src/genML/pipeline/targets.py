"""Target preparation and its inverse, driven by the resolved problem.

- Classification targets are label-encoded to 0..k-1; the original labels are
  kept so predictions can be mapped back. For binary problems the positive
  class is encoded as 1.
- Regression targets get the transform their metric needs (log1p for RMSLE),
  or a logit transform when bounded to [0, 1] (existing behaviour).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOGIT_EPSILON = 1e-7


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def encode_classes(raw: pd.Series, task: str, positive_class: Optional[Any] = None) -> Tuple[np.ndarray, List[Any]]:
    """Encode class labels as integers; returns (codes, classes)."""
    classes = sorted(pd.unique(raw.dropna()), key=lambda v: (str(type(v)), v))
    if task == "binary" and positive_class is not None:
        matches = [c for c in classes if str(c) == str(positive_class)]
        if not matches:
            raise ValueError(f"positive_class {positive_class!r} not among target values {classes}")
        classes = [c for c in classes if c is not matches[0]] + [matches[0]]
    lookup = {value: index for index, value in enumerate(classes)}
    codes = raw.map(lookup).to_numpy()
    if np.isnan(codes.astype(float)).any():
        raise ValueError("Target has missing values; drop or impute them before training")
    return codes.astype(np.int64), [_python_scalar(c) for c in classes]


def prepare_target(raw: pd.Series, task: str, metric_transform: Optional[str] = None,
                   positive_class: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return the training target and a description of how to invert it."""
    if task in ("binary", "multiclass"):
        codes, classes = encode_classes(raw, task, positive_class)
        logger.info("Encoded %d classes: %s", len(classes), classes[:10])
        return codes, {"applied": False, "type": "label_encoding", "classes": classes}

    values = raw.to_numpy(dtype=float)
    if metric_transform == "log1p":
        if np.nanmin(values) < 0:
            raise ValueError("RMSLE needs a non-negative target")
        logger.info("Applying log1p target transform for an RMSLE metric")
        return np.log1p(values), {"applied": True, "type": "log1p"}

    target_min, target_max = float(np.nanmin(values)), float(np.nanmax(values))
    if target_min >= 0 and target_max <= 1 and target_max > target_min:
        logger.info("Bounded target in [%.3f, %.3f]: applying logit transform", target_min, target_max)
        clipped = np.clip(values, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
        return np.log(clipped / (1 - clipped)), {
            "applied": True,
            "type": "logit",
            "original_min": target_min,
            "original_max": target_max,
            "epsilon": LOGIT_EPSILON,
        }

    return values, {"applied": False, "type": None, "reason": "unbounded_target"}


def inverse_regression(predictions: np.ndarray, transform_info: Dict[str, Any]) -> np.ndarray:
    """Undo a regression target transform."""
    if not transform_info or not transform_info.get("applied"):
        return predictions
    kind = transform_info.get("type")
    if kind == "log1p":
        return np.expm1(predictions)
    if kind == "logit":
        values = 1 / (1 + np.exp(-predictions))
        epsilon = transform_info.get("epsilon", LOGIT_EPSILON)
        return np.clip(values, epsilon, 1 - epsilon)
    raise ValueError(f"Unknown target transform '{kind}'")


def decode_classes(codes: np.ndarray, classes: List[Any]) -> np.ndarray:
    """Map integer class codes back to the original labels."""
    lookup = np.asarray(classes, dtype=object)
    decoded = lookup[np.asarray(codes, dtype=int)]
    # Integer-valued float labels (0.0/1.0) are written as integers
    if classes and all(isinstance(c, float) and float(c).is_integer() for c in classes):
        return decoded.astype(np.int64)
    # Keep numeric labels numeric (e.g. Titanic expects 0/1 integers)
    try:
        return decoded.astype(type(classes[0])) if classes else decoded
    except (TypeError, ValueError):
        return decoded
