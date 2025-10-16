"""Helper module that captures optional third-party dependencies."""

from __future__ import annotations

try:
    import lightgbm as lgb  # type: ignore

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully when unavailable
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb  # type: ignore

    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully when unavailable
    cb = None
    CATBOOST_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor  # type: ignore
    import torch  # type: ignore

    TABNET_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully when unavailable
    TabNetClassifier = None
    TabNetRegressor = None
    torch = None
    TABNET_AVAILABLE = False

__all__ = [
    "lgb",
    "cb",
    "TabNetClassifier",
    "TabNetRegressor",
    "torch",
    "LIGHTGBM_AVAILABLE",
    "CATBOOST_AVAILABLE",
    "TABNET_AVAILABLE",
]
