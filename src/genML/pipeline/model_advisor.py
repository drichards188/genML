"""AI-powered model advisor utilities."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.genML.gpu_utils import get_gpu_config
from src.genML.pipeline import config
from src.genML.pipeline.optional_dependencies import (
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    TABNET_AVAILABLE,
)

logger = logging.getLogger(__name__)


def build_dataset_profile(
    train_df: pd.DataFrame,
    metadata: Dict[str, Any],
    feature_names: List[str],
    problem_type: str,
) -> Dict[str, Any]:
    """Assemble a structured dataset description for the model advisor."""
    feature_names = feature_names or []
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train_df.columns if c not in numeric_cols]

    missing_ratios = (
        train_df.isnull().mean().sort_values(ascending=False).head(5).to_dict()
    )
    missing_summary = [
        {"feature": str(name), "missing_ratio": float(ratio)}
        for name, ratio in missing_ratios.items()
        if ratio > 0
    ]

    variance_summary: List[Dict[str, float]] = []
    if numeric_cols:
        var_series = train_df[numeric_cols].var().sort_values(ascending=False).head(5)
        variance_summary = [
            {"feature": str(name), "variance": float(value)}
            for name, value in var_series.items()
        ]

    target_distribution = metadata.get("target_distribution") or {}
    target_column = metadata.get("target_column")

    class_balance = None
    if (
        problem_type == "classification"
        and target_column
        and target_column in train_df.columns
    ):
        value_counts = train_df[target_column].value_counts(normalize=True)
        class_balance = [
            {"class": str(cls), "ratio": float(ratio)}
            for cls, ratio in value_counts.head(10).items()
        ]

    resource_info = get_gpu_config()

    profile = {
        "problem_type": problem_type,
        "row_count": int(train_df.shape[0]),
        "column_count_raw": int(train_df.shape[1]),
        "engineered_feature_count": int(len(feature_names)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "target_column": target_column,
        "target_distribution": target_distribution,
        "class_balance": class_balance,
        "missing_value_summary": missing_summary,
        "top_variance_features": variance_summary,
        "detected_domains": metadata.get("detected_domains", []),
        "ai_feature_summary": metadata.get("ai_feature_summary"),
        "ai_generated_features": metadata.get("ai_generated_features"),
        "sample_rows": train_df.head(3).to_dict("records"),
        "engineered_feature_samples": feature_names[:10],
        "resource_constraints": {
            "gpu_available": bool(resource_info.get("cuda_available")),
            "gpu_memory_gb": resource_info.get("gpu_memory_gb"),
            "cuml_available": bool(resource_info.get("cuml_available")),
            "xgboost_gpu_available": bool(resource_info.get("xgboost_gpu_available")),
        },
    }

    return profile


def load_model_selection_guidance() -> Dict[str, Any]:
    """Load previously generated model selection guidance if available."""
    guidance_path = config.REPORTS_DIR / "ai_model_selection.json"
    if not guidance_path.exists():
        return {}
    try:
        with open(guidance_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load model selection guidance: %s", exc)
        return {}


def save_model_selection_guidance(guidance: Dict[str, Any]) -> None:
    """Persist model advisor guidance to disk."""
    guidance_path = config.REPORTS_DIR / "ai_model_selection.json"
    try:
        guidance_path.parent.mkdir(parents=True, exist_ok=True)
        with open(guidance_path, "w") as f:
            json.dump(guidance, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save model selection guidance: %s", exc)


def apply_model_selection_guidance(
    models: Dict[str, Any],
    guidance: Dict[str, Any],
    base_tuning_list: List[str],
) -> Tuple[OrderedDict, Optional[List[str]], Dict[str, Any]]:
    """Reorder and filter the model dictionary based on AI guidance."""
    if not guidance or guidance.get("status") in {None, "failed"}:
        return OrderedDict(models), None, {"applied": False, "reason": "no_guidance"}

    excluded = set(guidance.get("excluded_models", []))
    recommended = guidance.get("recommended_models") or []
    available_names = list(models.keys())

    ordered_names = [
        name for name in recommended if name in models and name not in excluded
    ]
    ordered_names.extend(
        [
            name
            for name in available_names
            if name not in ordered_names and name not in excluded
        ]
    )

    filtered_models = OrderedDict(
        (name, models[name])
        for name in ordered_names
        if name in models and name not in excluded
    )

    if not filtered_models:
        filtered_models = OrderedDict(models)
        tuning_list = None
    else:
        if recommended:
            tuning_list = [
                name
                for name in recommended
                if name in base_tuning_list and name in filtered_models
            ]
        else:
            tuning_list = [
                name
                for name in ordered_names
                if name in base_tuning_list and name in filtered_models
            ]

    summary = {
        "applied": True,
        "recommended_order": ordered_names,
        "excluded_models": list(excluded),
        "guidance_status": guidance.get("status"),
        "confidence": guidance.get("confidence"),
        "per_model_rationale": guidance.get("per_model_rationale"),
        "global_risks": guidance.get("global_risks"),
        "notes": guidance.get("notes"),
    }

    return filtered_models, tuning_list, summary


def get_available_model_summaries(problem_type: str) -> Dict[str, Dict[str, Any]]:
    """Filter supported model summaries based on availability and task type."""
    summaries = {}

    for name, info in config.SUPPORTED_MODEL_SUMMARIES.items():
        if name == "LightGBM" and not LIGHTGBM_AVAILABLE:
            continue
        if name == "CatBoost" and not CATBOOST_AVAILABLE:
            continue
        if name == "TabNet" and not TABNET_AVAILABLE:
            continue
        if name == "Linear Regression" and problem_type == "classification":
            continue
        if name == "Logistic Regression" and problem_type == "regression":
            continue

        summaries[name] = info

    return summaries


def run_model_selection_advisor() -> str:
    """Execute the model selection advisor and persist its guidance."""
    try:
        logger.info("Running AI model selection advisor...")

        train_df = pd.read_pickle(config.DATA_DIR / "train_data.pkl")

        feature_report_path = config.REPORTS_DIR / "feature_engineering_report.json"
        if not feature_report_path.exists():
            raise FileNotFoundError("feature_engineering_report.json not found")

        with open(feature_report_path, "r") as f:
            feature_metadata = json.load(f)

        feature_names_file = config.FEATURES_DIR / "feature_names.txt"
        feature_names: List[str] = []
        if feature_names_file.exists():
            with open(feature_names_file, "r") as f:
                feature_names = [line.strip() for line in f.readlines() if line.strip()]

        problem_type = detect_problem_type()
        dataset_profile = build_dataset_profile(train_df, feature_metadata, feature_names, problem_type)

        supported_models = get_available_model_summaries(problem_type)

        historical_context = {}
        training_report_path = config.REPORTS_DIR / "model_training_report.json"
        if training_report_path.exists():
            try:
                with open(training_report_path, "r") as f:
                    historical_context = json.load(f)
            except Exception as exc:
                logger.warning("Could not load historical model report: %s", exc)

        from src.genML.ai_advisors import ModelSelectionAdvisor, OpenAIClient

        openai_client = OpenAIClient(config=config.AI_ADVISORS_CONFIG["openai_config"])
        advisor = ModelSelectionAdvisor(openai_client=openai_client)
        guidance = advisor.recommend_models(
            dataset_profile=dataset_profile,
            supported_models=supported_models,
            historical_context=historical_context,
        )

        guidance["problem_type"] = problem_type
        guidance["dataset_profile_digest"] = {
            "row_count": dataset_profile.get("row_count"),
            "engineered_feature_count": dataset_profile.get("engineered_feature_count"),
            "detected_domains": dataset_profile.get("detected_domains"),
        }

        save_model_selection_guidance(guidance)
        return json.dumps(guidance, indent=2)

    except Exception as exc:
        logger.warning("Model selection advisor failed: %s", exc)
        fallback = {
            "status": "failed",
            "error": str(exc),
        }
        save_model_selection_guidance(fallback)
        return json.dumps(fallback, indent=2)


def detect_problem_type() -> str:
    """
    Automatically detect whether this is a regression or classification problem.

    Returns:
        Either 'regression' or 'classification'
    """
    try:
        y_train = np.load(config.FEATURES_DIR / "y_train.npy")

        unique_values = len(np.unique(y_train))
        total_samples = len(y_train)
        unique_ratio = unique_values / total_samples

        is_integer_like = np.allclose(y_train, np.round(y_train))
        value_range = np.max(y_train) - np.min(y_train)

        if unique_values <= 20:
            problem_type = "classification"
            confidence = "high"
            reason = f"Only {unique_values} unique target values"
        elif unique_ratio < 0.05 and is_integer_like:
            problem_type = "classification"
            confidence = "medium"
            reason = f"Low unique ratio ({unique_ratio:.3f}) with integer-like values"
        elif unique_ratio > 0.1:
            problem_type = "regression"
            confidence = "high"
            reason = f"High unique ratio ({unique_ratio:.3f}) indicates continuous target"
        elif not is_integer_like:
            problem_type = "regression"
            confidence = "medium"
            reason = "Non-integer target values indicate regression"
        else:
            problem_type = "classification"
            confidence = "low"
            reason = "Ambiguous case - defaulting to classification"

        print(f"Problem type detected: {problem_type} (confidence: {confidence})")
        print(f"Reason: {reason}")
        print(f"Target statistics: {unique_values} unique values, {unique_ratio:.3f} ratio, range: {value_range:.3f}")

        return problem_type

    except Exception as exc:
        print(f"Error in problem type detection: {exc}")
        print("Defaulting to classification")
        return "classification"
