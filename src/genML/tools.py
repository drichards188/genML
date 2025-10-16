"""Compatibility layer exposing the historical genML tools API."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import optuna  # re-exported for existing imports

from src.genML.pipeline import (
    engineer_features,
    generate_predictions,
    load_dataset,
    run_model_selection_advisor,
    train_model_pipeline,
)
from src.genML.pipeline import config
from src.genML.pipeline.ai_tuning import (
    build_ai_tuning_override_details,
    extract_override_values,
    load_ai_tuning_overrides,
    sanitize_override_value,
    save_ai_tuning_overrides,
)
from src.genML.pipeline.feature_engineering import (
    apply_ai_generated_features,
    build_derived_feature_context,
)
from src.genML.pipeline.model_advisor import (
    apply_model_selection_guidance,
    build_dataset_profile,
    detect_problem_type,
    get_available_model_summaries,
    load_model_selection_guidance,
    save_model_selection_guidance,
)
from src.genML.pipeline.optional_dependencies import (
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    TABNET_AVAILABLE,
    TabNetClassifier,
    TabNetRegressor,
    cb,
    lgb,
    torch,
)
from src.genML.pipeline.prediction import generate_predictions
from src.genML.pipeline.tuning import (
    optimize_catboost,
    optimize_linear_model,
    optimize_random_forest,
    optimize_tabnet,
    optimize_xgboost,
)
from src.genML.pipeline.utils import cleanup_memory

logger = logging.getLogger(__name__)

__all__ = [
    "apply_ai_generated_features",
    "apply_model_selection_guidance",
    "AI_ADVISORS_CONFIG",
    "AI_FEATURE_EPSILON",
    "AI_SUPPORTED_FEATURE_OPERATIONS",
    "AI_TUNING_OVERRIDE_PATH",
    "AI_TUNING_SUPPORTED_PARAMETERS",
    "build_ai_tuning_override_details",
    "build_dataset_profile",
    "build_derived_feature_context",
    "CATBOOST_AVAILABLE",
    "cleanup_memory",
    "detect_problem_type",
    "engineer_features",
    "extract_override_values",
    "generate_predictions",
    "get_available_model_summaries",
    "GPU_CONFIG",
    "LIGHTGBM_AVAILABLE",
    "load_ai_tuning_overrides",
    "load_dataset",
    "load_model_selection_guidance",
    "MEMORY_CONFIG",
    "optimize_catboost",
    "optimize_linear_model",
    "optimize_random_forest",
    "optimize_tabnet",
    "optimize_xgboost",
    "run_model_selection_advisor",
    "sanitize_override_value",
    "save_ai_tuning_overrides",
    "save_model_selection_guidance",
    "SUPPORTED_MODEL_SUMMARIES",
    "TABNET_AVAILABLE",
    "TabNetClassifier",
    "TabNetRegressor",
    "train_model_pipeline",
    "TUNING_CONFIG",
    "lgb",
    "cb",
    "torch",
    "optuna",
]

_PATH_ATTRIBUTES = [
    "OUTPUTS_DIR",
    "DATA_DIR",
    "FEATURES_DIR",
    "MODELS_DIR",
    "PREDICTIONS_DIR",
    "REPORTS_DIR",
]


def _sync_paths() -> None:
    for attr in _PATH_ATTRIBUTES:
        globals()[attr] = getattr(config, attr)


_sync_paths()


# Backwards-compatible re-exports of configuration dictionaries/constants.
TUNING_CONFIG = config.TUNING_CONFIG
GPU_CONFIG = config.GPU_CONFIG
MEMORY_CONFIG = config.MEMORY_CONFIG
AI_ADVISORS_CONFIG = config.AI_ADVISORS_CONFIG
AI_SUPPORTED_FEATURE_OPERATIONS = config.AI_SUPPORTED_FEATURE_OPERATIONS
AI_FEATURE_EPSILON = config.AI_FEATURE_EPSILON
AI_TUNING_OVERRIDE_PATH = config.AI_TUNING_OVERRIDE_PATH
AI_TUNING_SUPPORTED_PARAMETERS = config.AI_TUNING_SUPPORTED_PARAMETERS
SUPPORTED_MODEL_SUMMARIES = config.SUPPORTED_MODEL_SUMMARIES


class _ToolsModule(ModuleType):
    """Custom module type that keeps path assignments in sync with config."""

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - thin setter
        if name in _PATH_ATTRIBUTES:
            setattr(config, name, Path(value))
            super().__setattr__(name, Path(value))
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ToolsModule
