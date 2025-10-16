"""AI-driven hyperparameter override utilities."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from src.genML.pipeline import config

logger = logging.getLogger(__name__)


def sanitize_override_value(model: str, parameter: str, value: Any) -> Any:
    """Coerce AI-proposed tuning values into safe numeric/boolean types."""
    spec = config.AI_TUNING_SUPPORTED_PARAMETERS.get(model, {}).get(parameter)
    if not spec:
        raise ValueError(f"Unsupported parameter '{parameter}' for model '{model}'")

    value_type, min_val, max_val = spec

    if value_type == "int":
        cast_val = int(round(float(value)))
        if min_val is not None:
            cast_val = max(min_val, cast_val)
        if max_val is not None:
            cast_val = min(max_val, cast_val)
        return cast_val

    if value_type == "float":
        cast_val = float(value)
        if min_val is not None:
            cast_val = max(min_val, cast_val)
        if max_val is not None:
            cast_val = min(max_val, cast_val)
        return float(round(cast_val, 6))

    if value_type == "bool":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return bool(value)

    raise ValueError(f"Unsupported value type '{value_type}' for model '{model}' parameter '{parameter}'")


def build_ai_tuning_override_details(
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convert AI tuning recommendations into structured override details."""
    details: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for rec in recommendations or []:
        model = str(rec.get("model", "")).strip()
        parameter = str(rec.get("parameter", "")).strip()
        if not model or not parameter:
            continue

        if model not in config.AI_TUNING_SUPPORTED_PARAMETERS:
            logger.warning("Skipping AI tuning recommendation for unsupported model '%s'", model)
            continue
        if parameter not in config.AI_TUNING_SUPPORTED_PARAMETERS[model]:
            logger.warning("Skipping unsupported parameter '%s' for model '%s'", parameter, model)
            continue

        if "suggested_value" not in rec:
            logger.warning(
                "Skipping recommendation for %s/%s without suggested_value",
                model,
                parameter,
            )
            continue

        try:
            sanitized_value = sanitize_override_value(model, parameter, rec["suggested_value"])
        except Exception as exc:
            logger.warning("Could not sanitize AI tuning value for %s/%s: %s", model, parameter, exc)
            continue

        model_details = details.setdefault(model, {})
        model_details[parameter] = {
            "suggested_value": sanitized_value,
            "original_value": rec["suggested_value"],
            "confidence": rec.get("confidence"),
            "rationale": rec.get("rationale"),
            "source": rec.get("source"),
        }

    return details


def extract_override_values(details: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Return simple model->parameter->value mapping from override details."""
    overrides: Dict[str, Dict[str, Any]] = {}
    for model, params in (details or {}).items():
        overrides[model] = {param: config_val["suggested_value"] for param, config_val in params.items()}
    return overrides


def load_ai_tuning_overrides() -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """Load saved AI tuning overrides and metadata if available."""
    if not config.AI_TUNING_OVERRIDE_PATH.exists():
        return {}, {}

    try:
        with open(config.AI_TUNING_OVERRIDE_PATH, "r") as f:
            payload = json.load(f)
            return payload.get("details", {}), payload.get("metadata", {})
    except Exception as exc:
        logger.warning("Failed to load AI tuning overrides: %s", exc)
        return {}, {}


def save_ai_tuning_overrides(
    details: Dict[str, Dict[str, Dict[str, Any]]],
    metadata: Dict[str, Any],
) -> None:
    """Persist AI tuning overrides and metadata to disk."""
    try:
        config.AI_TUNING_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.AI_TUNING_OVERRIDE_PATH, "w") as f:
            json.dump({"details": details, "metadata": metadata}, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save AI tuning overrides: %s", exc)
