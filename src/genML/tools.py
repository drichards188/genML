"""
Generic ML Tools for Machine Learning Pipeline using CrewAI

This module contains the core data science functions that power the ML pipeline.
Each function represents a major stage in the machine learning workflow and is designed
to be modular, reusable, and well-documented. The functions handle data loading,
feature engineering, model training, and prediction generation.
"""
import copy
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime
import logging
import traceback
import gc  # For explicit garbage collection to prevent memory leaks

# Configure logging
# Note: Full logging setup is done in logging_config.py and initialized in main.py
# This just creates a logger instance for this module
logger = logging.getLogger(__name__)
import optuna
from optuna.samplers import TPESampler
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    cb = None
    CATBOOST_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TabNetRegressor = None
    TabNetClassifier = None
    TABNET_AVAILABLE = False

# Import GPU utilities for unified GPU acceleration support
from src.genML.gpu_utils import (
    get_gpu_config,
    get_linear_model_classifier,
    get_linear_model_regressor,
    get_random_forest_classifier,
    get_random_forest_regressor,
    get_xgboost_params,
    log_gpu_memory,
    get_gpu_memory_usage,
    is_cuml_available,
    is_xgboost_gpu_available,
    is_catboost_gpu_available
)

# Directory structure for organizing pipeline outputs
# This structure separates different types of artifacts for better organization and debugging
OUTPUTS_DIR = Path("outputs")                    # Root output directory
DATA_DIR = OUTPUTS_DIR / "data"                  # Processed datasets
FEATURES_DIR = OUTPUTS_DIR / "features"          # Engineered features and scalers
MODELS_DIR = OUTPUTS_DIR / "models"              # Trained models and training artifacts
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"    # Prediction files and detailed results
REPORTS_DIR = OUTPUTS_DIR / "reports"            # Analysis reports and metadata

# Ensure all output directories exist - critical for pipeline execution
# This prevents file writing errors during pipeline execution
for dir_path in [OUTPUTS_DIR, DATA_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# GPU detection is now handled by gpu_utils module
# The module automatically detects GPU on import and provides smart imports


# Hyperparameter Tuning Configuration
TUNING_CONFIG = {
    'enabled': True,                          # Enable/disable hyperparameter tuning
    'n_trials': 100,                         # Number of Optuna trials per model
    'timeout': 600,                          # Max seconds per model tuning (10 min)
    'n_jobs': -1,                            # Parallel jobs (-1 = all cores)
    'show_progress_bar': True,               # Show Optuna progress
    'models_to_tune': ['Random Forest', 'XGBoost', 'CatBoost', 'TabNet', 'Logistic Regression', 'Linear Regression']
}

# GPU Acceleration Configuration
GPU_CONFIG = {
    'force_cpu': False,                      # Force CPU mode even if GPU is available
    'enable_cuml': True,                     # Enable cuML GPU acceleration for sklearn models
    'enable_xgboost_gpu': True,              # Enable XGBoost GPU acceleration
    'log_gpu_memory': True,                  # Log GPU memory usage during training
    'gpu_memory_threshold_gb': 14.0,         # Warning threshold for GPU memory (GB)
}

# Memory Management Configuration (Prevents WSL2 crashes and memory leaks)
MEMORY_CONFIG = {
    'max_parallel_jobs': 1,                  # Limit parallel jobs to prevent memory exhaustion (WSL2-safe: serial execution)
    'enable_gc_between_trials': True,        # Force garbage collection between Optuna trials
    'enable_gpu_memory_cleanup': True,       # Clear GPU memory between trials (cuML)
    'cv_n_jobs_limit': 1,                    # Limit cross-validation parallel jobs (WSL2-safe: serial execution)
    'max_trees_random_forest': 100,          # Limit Random Forest max trees to prevent memory exhaustion (reduced from 150)
    'rf_cv_folds': 3,                        # Reduce CV folds for Random Forest to limit memory usage
    'aggressive_rf_cleanup': True,           # Enable aggressive cleanup for Random Forest (prevents cuML leaks)
}

# AI Advisors Configuration (OpenAI-powered intelligent analysis)
AI_ADVISORS_CONFIG = {
    'enabled': True,                         # Enable/disable AI advisors globally
    'feature_ideation': {
        'enabled': True,                     # Enable feature ideation advisor
        'sample_size': 100,                  # Number of rows to sample for analysis
        'save_report': True,                 # Save feature suggestions report
    },
    'error_analysis': {
        'enabled': True,                     # Enable error pattern analyzer
        'top_n_errors': 100,                 # Number of worst errors to analyze in detail
        'save_report': True,                 # Save error analysis report
    },
    'openai_config': {
        'model': 'gpt-4o-mini',              # OpenAI model to use
        'max_cost_per_run': 10.0,            # Maximum API cost per run (USD)
        'enable_cache': True,                # Enable response caching to save costs
        'cache_dir': 'outputs/ai_cache',     # Cache directory
    }
}

AI_SUPPORTED_FEATURE_OPERATIONS = {'ratio', 'difference', 'product', 'sum', 'log', 'binary_threshold'}
AI_FEATURE_EPSILON = 1e-6
AI_TUNING_OVERRIDE_PATH = REPORTS_DIR / 'ai_tuning_overrides.json'
AI_TUNING_SUPPORTED_PARAMETERS = {
    'CatBoost': {
        'iterations': ('int', 200, 2000),
        'learning_rate': ('float', 0.01, 0.3),
        'depth': ('int', 3, 10),
        'l2_leaf_reg': ('float', 1.0, 20.0),
        'subsample': ('float', 0.3, 1.0)
    },
    'XGBoost': {
        'n_estimators': ('int', 100, 2000),
        'learning_rate': ('float', 0.01, 0.3),
        'max_depth': ('int', 3, 12),
        'subsample': ('float', 0.3, 1.0),
        'colsample_bytree': ('float', 0.3, 1.0)
    },
    'Random Forest': {
        'n_estimators': ('int', 100, 1000),
        'max_depth': ('int', 5, 25),
        'min_samples_split': ('int', 2, 20),
        'min_samples_leaf': ('int', 1, 10)
    },
    'LightGBM': {
        'n_estimators': ('int', 100, 2000),
        'learning_rate': ('float', 0.01, 0.3),
        'num_leaves': ('int', 16, 256),
        'feature_fraction': ('float', 0.2, 1.0)
    },
    'Logistic Regression': {
        'C': ('float', 0.0001, 100.0)
    },
    'Linear Regression': {
        'fit_intercept': ('bool', None, None)
    }
}

SUPPORTED_MODEL_SUMMARIES: Dict[str, Dict[str, Any]] = {
    'Linear Regression': {
        'type': 'linear',
        'best_for': ['interpretable baseline', 'low-dimensional numeric data'],
        'limitations': ['struggles with non-linearity', 'sensitive to multicollinearity'],
        'resource_cost': 'very_low'
    },
    'Logistic Regression': {
        'type': 'linear',
        'best_for': ['binary classification', 'interpretable coefficients'],
        'limitations': ['assumes linear decision boundary', 'requires feature scaling'],
        'resource_cost': 'very_low'
    },
    'Random Forest': {
        'type': 'tree_ensemble',
        'best_for': ['mixed feature types', 'robustness to noise'],
        'limitations': ['larger memory usage', 'less suited for sparse high-dimensional data'],
        'resource_cost': 'medium'
    },
    'XGBoost': {
        'type': 'gradient_boosting',
        'best_for': ['tabular datasets', 'handling missing values'],
        'limitations': ['requires tuning to avoid overfitting', 'can be slow on very wide datasets'],
        'resource_cost': 'medium_high'
    },
    'CatBoost': {
        'type': 'gradient_boosting',
        'best_for': ['categorical-heavy datasets', 'strong default performance'],
        'limitations': ['GPU memory sensitive', 'longer training for very large datasets'],
        'resource_cost': 'medium_high'
    },
    'LightGBM': {
        'type': 'gradient_boosting',
        'best_for': ['large datasets', 'sparse features'],
        'limitations': ['sensitive to categorical preprocessing when GPU absent'],
        'resource_cost': 'medium'
    },
    'TabNet': {
        'type': 'neural_network',
        'best_for': ['datasets with complex interactions', 'GPU availability'],
        'limitations': ['requires large sample size', 'potentially unstable without tuning'],
        'resource_cost': 'high'
    }
}


def build_dataset_profile(
    train_df: pd.DataFrame,
    metadata: Dict[str, Any],
    feature_names: List[str],
    problem_type: str
) -> Dict[str, Any]:
    """Assemble a structured dataset description for the model advisor."""
    feature_names = feature_names or []
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train_df.columns if c not in numeric_cols]

    missing_ratios = (
        train_df.isnull()
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )
    missing_summary = [
        {
            'feature': str(name),
            'missing_ratio': float(ratio)
        }
        for name, ratio in missing_ratios.items()
        if ratio > 0
    ]

    variance_summary: List[Dict[str, float]] = []
    if numeric_cols:
        var_series = (
            train_df[numeric_cols]
            .var()
            .sort_values(ascending=False)
            .head(5)
        )
        variance_summary = [
            {'feature': str(name), 'variance': float(value)}
            for name, value in var_series.items()
        ]

    target_distribution = metadata.get('target_distribution') or {}
    target_column = metadata.get('target_column')

    class_balance = None
    if (
        problem_type == 'classification'
        and target_column
        and target_column in train_df.columns
    ):
        value_counts = train_df[target_column].value_counts(normalize=True)
        class_balance = [
            {'class': str(cls), 'ratio': float(ratio)}
            for cls, ratio in value_counts.head(10).items()
        ]

    resource_info = get_gpu_config()

    profile = {
        'problem_type': problem_type,
        'row_count': int(train_df.shape[0]),
        'column_count_raw': int(train_df.shape[1]),
        'engineered_feature_count': int(len(feature_names)),
        'numeric_feature_count': int(len(numeric_cols)),
        'categorical_feature_count': int(len(categorical_cols)),
        'target_column': target_column,
        'target_distribution': target_distribution,
        'class_balance': class_balance,
        'missing_value_summary': missing_summary,
        'top_variance_features': variance_summary,
        'detected_domains': metadata.get('detected_domains', []),
        'ai_feature_summary': metadata.get('ai_feature_summary'),
        'ai_generated_features': metadata.get('ai_generated_features'),
        'sample_rows': train_df.head(3).to_dict('records'),
        'engineered_feature_samples': feature_names[:10],
        'resource_constraints': {
            'gpu_available': bool(resource_info.get('cuda_available')),
            'gpu_memory_gb': resource_info.get('gpu_memory_gb'),
            'cuml_available': bool(resource_info.get('cuml_available')),
            'xgboost_gpu_available': bool(resource_info.get('xgboost_gpu_available'))
        }
    }

    return profile


def load_model_selection_guidance() -> Dict[str, Any]:
    """Load previously generated model selection guidance if available."""
    guidance_path = REPORTS_DIR / 'ai_model_selection.json'
    if not guidance_path.exists():
        return {}
    try:
        with open(guidance_path, 'r') as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load model selection guidance: {exc}")
        return {}


def save_model_selection_guidance(guidance: Dict[str, Any]) -> None:
    guidance_path = REPORTS_DIR / 'ai_model_selection.json'
    try:
        guidance_path.parent.mkdir(parents=True, exist_ok=True)
        with open(guidance_path, 'w') as f:
            json.dump(guidance, f, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to save model selection guidance: {exc}")


def apply_model_selection_guidance(
    models: Dict[str, Any],
    guidance: Dict[str, Any],
    base_tuning_list: List[str]
) -> Tuple[OrderedDict, Optional[List[str]], Dict[str, Any]]:
    """Reorder and filter the model dictionary based on AI guidance."""
    if not guidance or guidance.get('status') in {None, 'failed'}:
        return OrderedDict(models), None, {'applied': False, 'reason': 'no_guidance'}

    excluded = set(guidance.get('excluded_models', []))
    recommended = guidance.get('recommended_models') or []
    available_names = list(models.keys())

    ordered_names = [
        name for name in recommended
        if name in models and name not in excluded
    ]
    ordered_names.extend(
        [name for name in available_names if name not in ordered_names and name not in excluded]
    )

    filtered_models = OrderedDict(
        (name, models[name]) for name in ordered_names if name in models and name not in excluded
    )

    if not filtered_models:
        filtered_models = OrderedDict(models)
        tuning_list = None
    else:
        # Only include recommended models in tuning list if recommendations exist
        if recommended:
            tuning_list = [
                name for name in recommended
                if name in base_tuning_list and name in filtered_models
            ]
        else:
            # No recommendations, use all non-excluded models from base_tuning_list
            tuning_list = [
                name for name in ordered_names
                if name in base_tuning_list and name in filtered_models
            ]

    summary = {
        'applied': True,
        'recommended_order': ordered_names,
        'excluded_models': list(excluded),
        'guidance_status': guidance.get('status'),
        'confidence': guidance.get('confidence'),
        'per_model_rationale': guidance.get('per_model_rationale'),
        'global_risks': guidance.get('global_risks'),
        'notes': guidance.get('notes')
    }

    return filtered_models, tuning_list, summary


def _normalize_ai_feature_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize an AI-generated feature specification."""
    if not isinstance(spec, dict):
        raise ValueError("feature spec must be a dictionary")

    name = str(spec.get('name', '')).strip()
    if not name:
        raise ValueError("missing feature name")
    name = name.replace(' ', '_')

    operation = str(spec.get('operation', '')).strip().lower()
    if operation not in AI_SUPPORTED_FEATURE_OPERATIONS:
        raise ValueError(f"unsupported operation '{operation}'")

    inputs = spec.get('inputs', [])
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a non-empty list")
    inputs = [str(inp).strip() for inp in inputs if str(inp).strip()]
    if not inputs:
        raise ValueError("inputs cannot be empty strings")

    if operation in {'ratio', 'difference'} and len(inputs) != 2:
        raise ValueError(f"{operation} operation requires exactly 2 inputs")
    if operation == 'log' and len(inputs) != 1:
        raise ValueError("log operation requires exactly 1 input")
    if operation == 'binary_threshold' and len(inputs) != 1:
        raise ValueError("binary_threshold operation requires exactly 1 input")
    if operation in {'product', 'sum'} and len(inputs) < 2:
        raise ValueError(f"{operation} operation requires at least 2 inputs")

    parameters = spec.get('parameters') or {}
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a dictionary")

    if operation == 'binary_threshold':
        if 'threshold' not in parameters:
            raise ValueError("binary_threshold operation requires 'threshold' parameter")
        try:
            parameters['threshold'] = float(parameters['threshold'])
        except (TypeError, ValueError):
            raise ValueError("binary_threshold threshold must be a numeric value")

    expected_impact = str(spec.get('expected_impact', '')).strip().lower()
    rationale = str(spec.get('rationale', '')).strip()

    return {
        'name': name,
        'operation': operation,
        'inputs': inputs,
        'parameters': parameters,
        'expected_impact': expected_impact,
        'rationale': rationale
    }


def _resolve_unique_feature_name(base_name: str, existing: set) -> str:
    """Ensure the generated feature name does not clash with existing columns."""
    name = base_name
    suffix = 1
    while name in existing:
        name = f"{base_name}_{suffix}"
        suffix += 1
    return name


def _get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Extract a numeric pandas Series from a dataframe column."""
    if column not in df.columns:
        raise KeyError(f"column '{column}' not found in dataset")
    series = pd.to_numeric(df[column], errors='coerce')
    return series.fillna(0.0)


def _compute_ai_feature_series(operation: str, inputs: List[str], parameters: Dict[str, Any], source_df: pd.DataFrame) -> pd.Series:
    """Compute an engineered feature series based on the supported operations."""
    if operation == 'ratio':
        numerator = _get_numeric_series(source_df, inputs[0])
        denominator = _get_numeric_series(source_df, inputs[1])
        safe_denominator = denominator.where(np.abs(denominator) > AI_FEATURE_EPSILON, np.nan)
        result = numerator / safe_denominator
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif operation == 'difference':
        series_a = _get_numeric_series(source_df, inputs[0])
        series_b = _get_numeric_series(source_df, inputs[1])
        result = series_a - series_b
    elif operation == 'product':
        result = pd.Series(1.0, index=source_df.index, dtype=float)
        for column in inputs:
            result = result * _get_numeric_series(source_df, column)
    elif operation == 'sum':
        result = pd.Series(0.0, index=source_df.index, dtype=float)
        for column in inputs:
            result = result + _get_numeric_series(source_df, column)
    elif operation == 'log':
        base_series = _get_numeric_series(source_df, inputs[0])
        result = np.log1p(base_series.clip(lower=0.0))
    elif operation == 'binary_threshold':
        threshold = parameters.get('threshold', 0.0)
        base_series = _get_numeric_series(source_df, inputs[0])
        result = (base_series > threshold).astype(float)
    else:
        raise ValueError(f"operation '{operation}' not implemented")

    return result.astype(np.float32)


def apply_ai_generated_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    suggestions: Optional[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Apply AI-generated feature specifications to the engineered feature matrices.

    Returns:
        Tuple of (train_features, test_features, summary_dict)
    """
    summary: Dict[str, Any] = {
        'status': 'skipped',
        'attempted': 0,
        'successful': 0,
        'created_features': [],
        'failed_features': []
    }

    if not suggestions or suggestions.get('status') != 'success':
        summary['reason'] = 'no_valid_suggestions'
        return train_features, test_features, summary

    feature_specs = suggestions.get('engineered_features') or []
    if not feature_specs:
        summary['reason'] = 'no_engineered_features'
        return train_features, test_features, summary

    # Align indices to avoid mismatches when assigning new columns
    train_features = train_features.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    train_source = train_raw.reset_index(drop=True)
    test_source = test_raw.reset_index(drop=True)

    summary['attempted'] = len(feature_specs)
    existing_columns = set(train_features.columns)

    for spec in feature_specs:
        try:
            normalized = _normalize_ai_feature_spec(spec)
            feature_name = _resolve_unique_feature_name(normalized['name'], existing_columns)

            train_series = _compute_ai_feature_series(
                normalized['operation'],
                normalized['inputs'],
                normalized['parameters'],
                train_source
            )
            test_series = _compute_ai_feature_series(
                normalized['operation'],
                normalized['inputs'],
                normalized['parameters'],
                test_source
            )

            train_features[feature_name] = train_series
            test_features[feature_name] = test_series
            existing_columns.add(feature_name)

            summary['successful'] += 1
            summary['created_features'].append({
                'name': feature_name,
                'operation': normalized['operation'],
                'inputs': normalized['inputs'],
                'expected_impact': normalized['expected_impact'],
                'rationale': normalized['rationale']
            })
        except Exception as exc:
            failure_record = {
                'name': spec.get('name'),
                'operation': spec.get('operation'),
                'reason': str(exc)
            }
            summary['failed_features'].append(failure_record)
            logger.warning(f"Failed to apply AI-generated feature {failure_record['name']}: {exc}")

    if summary['successful'] > 0:
        summary['status'] = 'applied'
    else:
        summary['reason'] = summary.get('reason', 'no_features_applied')

    return train_features, test_features, summary


def sanitize_override_value(model: str, parameter: str, value: Any) -> Any:
    """Coerce AI-proposed tuning values into safe numeric/boolean types."""
    spec = AI_TUNING_SUPPORTED_PARAMETERS.get(model, {}).get(parameter)
    if not spec:
        raise ValueError(f"Unsupported parameter '{parameter}' for model '{model}'")

    value_type, min_val, max_val = spec

    if value_type == 'int':
        cast_val = int(round(float(value)))
        if min_val is not None:
            cast_val = max(min_val, cast_val)
        if max_val is not None:
            cast_val = min(max_val, cast_val)
        return cast_val

    if value_type == 'float':
        cast_val = float(value)
        if min_val is not None:
            cast_val = max(min_val, cast_val)
        if max_val is not None:
            cast_val = min(max_val, cast_val)
        return float(round(cast_val, 6))

    if value_type == 'bool':
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', '1', 'yes', 'y'}:
                return True
            if lowered in {'false', '0', 'no', 'n'}:
                return False
        return bool(value)

    raise ValueError(f"Unsupported value type '{value_type}' for model '{model}' parameter '{parameter}'")


def build_ai_tuning_override_details(recommendations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convert AI tuning recommendations into structured override details."""
    details: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for rec in recommendations or []:
        model = str(rec.get('model', '')).strip()
        parameter = str(rec.get('parameter', '')).strip()
        if not model or not parameter:
            continue

        if model not in AI_TUNING_SUPPORTED_PARAMETERS:
            logger.warning(f"Skipping AI tuning recommendation for unsupported model '{model}'")
            continue
        if parameter not in AI_TUNING_SUPPORTED_PARAMETERS[model]:
            logger.warning(f"Skipping unsupported parameter '{parameter}' for model '{model}'")
            continue

        if 'suggested_value' not in rec:
            logger.warning(f"Skipping recommendation for {model}/{parameter} without suggested_value")
            continue

        try:
            sanitized_value = sanitize_override_value(model, parameter, rec['suggested_value'])
        except Exception as exc:
            logger.warning(f"Could not sanitize AI tuning value for {model}/{parameter}: {exc}")
            continue

        rationale = str(rec.get('rationale', '')).strip()
        confidence = str(rec.get('confidence', '')).strip().lower()

        details.setdefault(model, {})[parameter] = {
            'value': sanitized_value,
            'confidence': confidence,
            'rationale': rationale,
            'suggested_value_raw': rec.get('suggested_value')
        }

    return details


def extract_override_values(details: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Flatten override details into model -> param -> value structure."""
    values: Dict[str, Dict[str, Any]] = {}
    for model, params in (details or {}).items():
        values[model] = {param: info.get('value') for param, info in params.items() if 'value' in info}
    return values


def load_ai_tuning_overrides() -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """Load persisted AI tuning overrides."""
    if not AI_TUNING_OVERRIDE_PATH.exists():
        return {}, {}
    try:
        with open(AI_TUNING_OVERRIDE_PATH, 'r') as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read AI tuning overrides: {exc}")
        return {}, {}

    details = data.get('overrides', {})
    return details, data


def save_ai_tuning_overrides(details: Dict[str, Dict[str, Dict[str, Any]]], metadata: Dict[str, Any]) -> None:
    """Persist AI tuning overrides to disk."""
    payload = {
        'generated_at': datetime.utcnow().isoformat(),
        'overrides': details,
        **metadata
    }
    AI_TUNING_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AI_TUNING_OVERRIDE_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved AI tuning overrides to {AI_TUNING_OVERRIDE_PATH}")

def load_dataset() -> str:
    """
    Load Titanic train and test datasets and return summary information.

    Returns:
        JSON string with dataset summary including shapes and preview
    """
    try:
        # Check for datasets in organized directory structure first
        # Priority: datasets/current/ -> datasets/active -> root directory
        dataset_paths = [
            'datasets/current/',
            'datasets/active/',
            '.'
        ]

        train_path = None
        test_path = None

        for base_path in dataset_paths:
            potential_train = os.path.join(base_path, 'train.csv')
            potential_test = os.path.join(base_path, 'test.csv')

            if os.path.exists(potential_train) and os.path.exists(potential_test):
                train_path = potential_train
                test_path = potential_test
                print(f"Using dataset from: {base_path}")
                break

        if not train_path or not test_path:
            return json.dumps({
                "error": "train.csv and test.csv files not found. Please place dataset files in datasets/current/ or project root directory.",
                "status": "failed"
            })

        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Detect target column by comparing train vs test columns
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        train_only_cols = train_cols - test_cols

        # Exclude common ID column names
        id_patterns = {'id', 'ID', 'Id', 'index', 'Index'}
        potential_targets = [col for col in train_only_cols if col not in id_patterns]
        target_col = potential_targets[0] if potential_targets else None

        # Calculate target statistics if target found
        target_rate = None
        if target_col and target_col in train_df.columns:
            try:
                # For binary/categorical targets
                if train_df[target_col].dtype in ['object', 'category'] or train_df[target_col].nunique() <= 20:
                    target_rate = train_df[target_col].value_counts(normalize=True).to_dict()
                else:
                    # For continuous targets, provide basic stats
                    target_rate = {
                        'mean': float(train_df[target_col].mean()),
                        'std': float(train_df[target_col].std()),
                        'min': float(train_df[target_col].min()),
                        'max': float(train_df[target_col].max())
                    }
            except:
                target_rate = None

        # Create summary
        summary = {
            "status": "success",
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "train_columns": list(train_df.columns),
            "test_columns": list(test_df.columns),
            "target_column": target_col,
            "missing_values": {
                "train": train_df.isnull().sum().to_dict(),
                "test": test_df.isnull().sum().to_dict()
            },
            "target_stats": target_rate,
            "train_head": train_df.head().to_dict('records'),
            "test_head": test_df.head().to_dict('records')
        }

        # Save datasets to organized folders
        train_df.to_pickle(DATA_DIR / 'train_data.pkl')
        test_df.to_pickle(DATA_DIR / 'test_data.pkl')

        # Save data exploration report
        with open(REPORTS_DIR / 'data_exploration_report.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return json.dumps(summary, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error loading data: {str(e)}",
            "status": "failed"
        })


def engineer_features() -> str:
    """
    Create engineered features from the loaded dataset using automated feature engineering.

    This function now uses the AutoFeatureEngine to automatically detect data types,
    apply appropriate feature processors, and perform intelligent feature selection.
    It can adapt to different datasets while maintaining the same interface.

    Key capabilities:
    - Automatic data type detection (numerical, categorical, text, datetime)
    - Domain-specific feature engineering strategies
    - Intelligent feature selection
    - Configurable processing based on dataset characteristics

    Returns:
        JSON string with feature engineering results and metadata
    """
    try:
        # Load preprocessed datasets from previous pipeline stage
        train_df = pd.read_pickle(DATA_DIR / 'train_data.pkl')
        test_df = pd.read_pickle(DATA_DIR / 'test_data.pkl')

        # Initialize the automated feature engineering engine
        from src.genML.features import AutoFeatureEngine

        # Configuration for feature engineering
        config = {
            'max_features': 200,  # Increased from 100 to allow more features (was too aggressive)
            'enable_feature_selection': True,
            'feature_importance_threshold': 0.001,  # Lowered from 0.01 to be less aggressive
            'manual_type_hints': {
                'num_lanes': 'numerical',
                'curvature': 'numerical',
                'speed_limit': 'numerical',
                'holiday': 'categorical',
                'num_reported_accidents': 'numerical'
            },
            'interaction_pairs': [
                # Generic numerical interactions that work across datasets
                # The feature engine will silently skip interactions if columns don't exist
                # Add dataset-specific interactions here if needed for competition optimization
            ],
            'numerical_config': {
                'enable_scaling': True,
                'enable_binning': True,
                'enable_polynomial': True,
                'polynomial_degree': 2,
                'n_bins': 5
            },
            'categorical_config': {
                'encoding_method': 'target',
                'enable_frequency': True,
                'max_categories': 20
            },
            'text_config': {
                'enable_basic_features': True,
                'enable_tfidf': False,  # Disabled for performance
                'enable_patterns': True
            },
            'feature_selection': {
                'max_features': 200,  # Increased from 100 to retain more features
                'enable_statistical_tests': True,
                'enable_model_based': True,
                'selection_strategy': 'union',
                'feature_importance_threshold': 0.001  # Lowered from 0.01 to be less aggressive
            }
        }

        # Create and fit the feature engine
        feature_engine = AutoFeatureEngine(config)

        # Analyze the training data to understand structure and domain
        analysis_results = feature_engine.analyze_data(train_df)

        # Fit the feature engineering pipeline
        # Detect target column by comparing train vs test columns (most reliable method)
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        train_only_cols = train_cols - test_cols

        # Exclude common ID column names
        id_patterns = {'id', 'ID', 'Id', 'index', 'Index'}
        potential_targets = [col for col in train_only_cols if col not in id_patterns]
        target_col = potential_targets[0] if potential_targets else None

        # Fallback to analysis-based detection if comparison fails
        if target_col is None and analysis_results.get('target_candidates'):
            target_col = analysis_results['target_candidates'][0]

        if target_col:
            print(f"Detected target column: {target_col}")
        else:
            print("Warning: Could not detect target column")

        feature_engine.fit(train_df, target_col)

        ai_feature_suggestions: Optional[Dict[str, Any]] = None

        # AI Advisor: Feature Ideation
        # Use AI to suggest domain-specific features based on current feature set
        if AI_ADVISORS_CONFIG['enabled'] and AI_ADVISORS_CONFIG['feature_ideation']['enabled']:
            try:
                print(f"\n{'='*60}")
                print("🤖 AI Feature Ideation Advisor")
                print(f"{'='*60}")
                print("Analyzing current features and suggesting improvements...")

                from src.genML.ai_advisors import FeatureIdeationAdvisor, OpenAIClient

                # Create OpenAI client with configuration
                openai_client = OpenAIClient(config=AI_ADVISORS_CONFIG['openai_config'])

                if openai_client.is_available():
                    # Get current feature list
                    current_features = [col for col in train_df.columns if col != target_col]

                    # Get feature importances if available from feature engine
                    feature_importances = None
                    if hasattr(feature_engine, 'feature_selector') and hasattr(feature_engine.feature_selector, 'feature_importances_'):
                        importance_dict = {}
                        for feat, imp in zip(current_features, feature_engine.feature_selector.feature_importances_):
                            importance_dict[feat] = float(imp)
                        feature_importances = importance_dict

                    # Create advisor and generate suggestions
                    advisor = FeatureIdeationAdvisor(openai_client=openai_client)

                    # Sample data for analysis (avoid sending too much data to API)
                    sample_size = AI_ADVISORS_CONFIG['feature_ideation']['sample_size']
                    df_sample = train_df.head(sample_size)

                    # Get detected domain from analysis results
                    detected_domain = analysis_results.get('domain_analysis', {}).get('detected_domains', [None])[0]

                    suggestions = advisor.suggest_features(
                        df_sample=df_sample,
                        current_features=current_features,
                        target_col=target_col,
                        detected_domain=detected_domain,
                        feature_importances=feature_importances
                    )

                    # Display suggestions summary
                    if suggestions.get('status') == 'success':
                        ai_feature_suggestions = suggestions
                        n_suggested = len(suggestions.get('engineered_features', []))
                        n_interactions = len(suggestions.get('interaction_suggestions', []))
                        n_transformations = len(suggestions.get('transformation_suggestions', []))

                        print(f"✅ AI Feature Ideation completed!")
                        print(f"   New features suggested: {n_suggested}")
                        print(f"   Interaction suggestions: {n_interactions}")
                        print(f"   Transformation suggestions: {n_transformations}")

                        # Display top priority features
                        if suggestions.get('priority_features'):
                            print(f"\n   Top Priority Features:")
                            for i, feat in enumerate(suggestions['priority_features'][:3], 1):
                                print(f"      {i}. {feat}")

                        # Save report if configured
                        if AI_ADVISORS_CONFIG['feature_ideation']['save_report']:
                            report_path = REPORTS_DIR / 'ai_feature_suggestions.json'
                            advisor.save_report(suggestions, str(report_path))
                            print(f"\n   📄 Report saved to: {report_path}")

                        # Display API usage stats
                        usage_stats = openai_client.get_usage_stats()
                        print(f"\n   💰 API Cost: ${usage_stats['total_cost']:.4f}")
                        print(f"   📊 Tokens: {usage_stats['total_tokens']:,} ({usage_stats['input_tokens']:,} in, {usage_stats['output_tokens']:,} out)")
                    else:
                        print(f"⚠️  AI Feature Ideation returned an error")
                else:
                    print("⚠️  OpenAI API not available - skipping feature ideation")
                    print("   Set OPENAI_API_KEY environment variable to enable AI advisors")

                print(f"{'='*60}\n")

            except Exception as e:
                logger.warning(f"AI Feature Ideation failed: {e}")
                print(f"⚠️  AI Feature Ideation failed: {e}")
                print("Continuing with existing features...\n")

        # Transform both training and test data
        train_features = feature_engine.transform(train_df)
        test_features = feature_engine.transform(test_df)

        ai_feature_summary = {
            'status': 'skipped',
            'attempted': 0,
            'successful': 0,
            'reason': 'not_evaluated'
        }

        if ai_feature_suggestions:
            train_features, test_features, ai_feature_summary = apply_ai_generated_features(
                train_df,
                test_df,
                train_features,
                test_features,
                ai_feature_suggestions
            )

            if ai_feature_summary['status'] == 'applied':
                created_names = [feat['name'] for feat in ai_feature_summary['created_features']]
                preview = ', '.join(created_names[:5]) if created_names else '(names unavailable)'
                print(f"   ✅ Applied {ai_feature_summary['successful']} AI-generated feature(s): {preview}")
                if ai_feature_summary['successful'] > 5:
                    remaining = ai_feature_summary['successful'] - 5
                    print(f"      ...and {remaining} more")
            elif ai_feature_summary.get('reason'):
                print(f"   ℹ️  AI feature application skipped: {ai_feature_summary['reason']}")
        else:
            ai_feature_summary = {
                'status': 'skipped',
                'attempted': 0,
                'successful': 0,
                'reason': 'no_ai_suggestions'
            }

        # Ensure purely numeric matrices for downstream numpy operations
        train_features = train_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        test_features = test_features.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Combine datasets to maintain original approach for downstream compatibility
        # Add a column to track which rows are train vs test
        train_features['_is_train'] = 1
        test_features['_is_train'] = 0

        combined_features = pd.concat([train_features, test_features], ignore_index=True)

        # Extract final feature matrix for compatibility with existing pipeline
        combined_features = combined_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        feature_columns = [col for col in combined_features.columns if col != '_is_train']
        train_len = len(train_df)

        train_features_final = combined_features.loc[combined_features['_is_train'] == 1, feature_columns].to_numpy(dtype=np.float32, copy=False)
        test_features_final = combined_features.loc[combined_features['_is_train'] == 0, feature_columns].to_numpy(dtype=np.float32, copy=False)

        # Get target variable
        if target_col and target_col in train_df.columns:
            train_target = train_df[target_col].values
        else:
            # Fallback: assume last column or binary target
            train_target = train_df.iloc[:, -1].values

        # Apply target transformation for bounded regression problems
        # If target is bounded in [0,1] (like probabilities), use logit transform
        target_min = np.min(train_target)
        target_max = np.max(train_target)
        apply_logit_transform = False

        if target_min >= 0 and target_max <= 1 and target_max > target_min:
            # Target is bounded in [0,1] - apply logit transformation
            # This helps models that assume unbounded targets
            apply_logit_transform = True
            print(f"Detected bounded target in [{target_min:.3f}, {target_max:.3f}]")
            print(f"Applying logit transformation: logit(y) = log(y / (1-y))")

            # Clip values to avoid log(0) and division by zero
            epsilon = 1e-7
            train_target_clipped = np.clip(train_target, epsilon, 1 - epsilon)

            # Apply logit transform
            train_target_transformed = np.log(train_target_clipped / (1 - train_target_clipped))

            # Save transformation parameters
            transform_info = {
                'applied': True,
                'type': 'logit',
                'original_min': float(target_min),
                'original_max': float(target_max),
                'epsilon': epsilon
            }
            joblib.dump(transform_info, FEATURES_DIR / 'target_transform.pkl')

            print(f"Transformed target range: [{np.min(train_target_transformed):.3f}, {np.max(train_target_transformed):.3f}]")

            # Use transformed target for training
            train_target = train_target_transformed
        else:
            # No transformation needed
            transform_info = {'applied': False}
            joblib.dump(transform_info, FEATURES_DIR / 'target_transform.pkl')
            print("Target transformation: None (unbounded target)")

        # Save processed features in the expected format
        np.save(FEATURES_DIR / 'X_train.npy', train_features_final)
        np.save(FEATURES_DIR / 'X_test.npy', test_features_final)
        np.save(FEATURES_DIR / 'y_train.npy', train_target)

        # Save a compatibility scaler (features are already scaled by the feature engine)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(train_features_final.shape[1])
        scaler.scale_ = np.ones(train_features_final.shape[1])
        joblib.dump(scaler, FEATURES_DIR / 'scaler.pkl')

        # Save feature names for downstream compatibility
        with open(FEATURES_DIR / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_columns))

        # Save comprehensive automated feature engineering report
        feature_engine.save_report(REPORTS_DIR / 'automated_feature_engineering_report.json')

        # Prepare metadata for return (compatible with existing pipeline interface)
        metadata = {
            "status": "success",
            "features_used": feature_columns,
            "train_shape": train_features_final.shape,
            "test_shape": test_features_final.shape,
            "feature_engineering_method": "automated",
            "total_features_generated": len(feature_columns),
            "detected_domains": analysis_results.get('domain_analysis', {}).get('detected_domains', []),
            "target_column": target_col,
            "ai_feature_summary": ai_feature_summary,
            "ai_generated_features": [feat['name'] for feat in ai_feature_summary.get('created_features', [])],
            "feature_stats": {
                "mean": np.nanmean(train_features_final, axis=0).tolist(),
                "std": np.nanstd(train_features_final, axis=0).tolist()
            },
            "target_transformation": {
                "applied": transform_info['applied'],
                "type": transform_info.get('type', None),
                "original_range": [transform_info.get('original_min'), transform_info.get('original_max')] if transform_info['applied'] else None,
                "transformed_range": [float(np.min(train_target)), float(np.max(train_target))]
            },
            "target_distribution": {
                "mean": float(np.mean(train_target)),
                "std": float(np.std(train_target)),
                "min": float(np.min(train_target)),
                "max": float(np.max(train_target))
            }
        }

        # Save feature engineering report for pipeline compatibility
        with open(REPORTS_DIR / 'feature_engineering_report.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return json.dumps(metadata, indent=2)

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Automated feature engineering crashed:\n{error_trace}")
        print("=== AUTOMATED FEATURE ENGINEERING TRACEBACK ===")
        print(error_trace)
        print("=== END TRACEBACK ===")
        return json.dumps({
            "error": f"Automated feature engineering failed: {str(e)}",
            "status": "failed"
        })


def get_available_model_summaries(problem_type: str) -> Dict[str, Dict[str, Any]]:
    """Filter supported model summaries based on availability and task type."""
    summaries = {}

    for name, info in SUPPORTED_MODEL_SUMMARIES.items():
        if name == 'LightGBM' and not LIGHTGBM_AVAILABLE:
            continue
        if name == 'CatBoost' and not CATBOOST_AVAILABLE:
            continue
        if name == 'TabNet' and not TABNET_AVAILABLE:
            continue
        if name == 'Linear Regression' and problem_type == 'classification':
            continue
        if name == 'Logistic Regression' and problem_type == 'regression':
            continue

        summaries[name] = info

    return summaries


def run_model_selection_advisor() -> str:
    """Execute the model selection advisor and persist its guidance."""
    try:
        logger.info("Running AI model selection advisor...")

        train_df = pd.read_pickle(DATA_DIR / 'train_data.pkl')

        feature_report_path = REPORTS_DIR / 'feature_engineering_report.json'
        if not feature_report_path.exists():
            raise FileNotFoundError("feature_engineering_report.json not found")

        with open(feature_report_path, 'r') as f:
            feature_metadata = json.load(f)

        feature_names_file = FEATURES_DIR / 'feature_names.txt'
        feature_names = []
        if feature_names_file.exists():
            with open(feature_names_file, 'r') as f:
                feature_names = [line.strip() for line in f.readlines() if line.strip()]

        problem_type = detect_problem_type()
        dataset_profile = build_dataset_profile(train_df, feature_metadata, feature_names, problem_type)

        supported_models = get_available_model_summaries(problem_type)

        historical_context = {}
        training_report_path = REPORTS_DIR / 'model_training_report.json'
        if training_report_path.exists():
            try:
                with open(training_report_path, 'r') as f:
                    historical_context = json.load(f)
            except Exception as exc:
                logger.warning(f"Could not load historical model report: {exc}")

        from src.genML.ai_advisors import ModelSelectionAdvisor, OpenAIClient

        openai_client = OpenAIClient(config=AI_ADVISORS_CONFIG['openai_config'])
        advisor = ModelSelectionAdvisor(openai_client=openai_client)
        guidance = advisor.recommend_models(
            dataset_profile=dataset_profile,
            supported_models=supported_models,
            historical_context=historical_context
        )

        guidance['problem_type'] = problem_type
        guidance['dataset_profile_digest'] = {
            'row_count': dataset_profile.get('row_count'),
            'engineered_feature_count': dataset_profile.get('engineered_feature_count'),
            'detected_domains': dataset_profile.get('detected_domains')
        }

        save_model_selection_guidance(guidance)
        return json.dumps(guidance, indent=2)

    except Exception as exc:
        logger.warning(f"Model selection advisor failed: {exc}")
        fallback = {
            'status': 'failed',
            'error': str(exc)
        }
        save_model_selection_guidance(fallback)
        return json.dumps(fallback, indent=2)


def detect_problem_type() -> str:
    """
    Automatically detect whether this is a regression or classification problem.

    Analyzes the target variable to determine the problem type based on:
    - Data type (continuous vs discrete)
    - Number of unique values
    - Value range and distribution

    Returns:
        Either 'regression' or 'classification'
    """
    try:
        # Load target variable
        y_train = np.load(FEATURES_DIR / 'y_train.npy')

        # Calculate key statistics for analysis
        unique_values = len(np.unique(y_train))
        total_samples = len(y_train)
        unique_ratio = unique_values / total_samples

        # Check data type characteristics
        is_integer_like = np.allclose(y_train, np.round(y_train))
        value_range = np.max(y_train) - np.min(y_train)

        # Decision logic for problem type detection
        # Classification indicators:
        # - Small number of unique values (< 20 or < 5% of samples)
        # - Integer-like values
        # - Small value range

        if unique_values <= 20:
            # Very few unique values - likely classification
            problem_type = 'classification'
            confidence = 'high'
            reason = f'Only {unique_values} unique target values'

        elif unique_ratio < 0.05 and is_integer_like:
            # Few unique values relative to sample size, and integer-like
            problem_type = 'classification'
            confidence = 'medium'
            reason = f'Low unique ratio ({unique_ratio:.3f}) with integer-like values'

        elif unique_ratio > 0.1:
            # Many unique values relative to sample size - likely regression
            problem_type = 'regression'
            confidence = 'high'
            reason = f'High unique ratio ({unique_ratio:.3f}) indicates continuous target'

        elif not is_integer_like:
            # Non-integer values - likely regression
            problem_type = 'regression'
            confidence = 'medium'
            reason = 'Non-integer target values indicate regression'

        else:
            # Edge case - default to classification for safety
            problem_type = 'classification'
            confidence = 'low'
            reason = 'Ambiguous case - defaulting to classification'

        # Log the decision for transparency
        print(f"Problem type detected: {problem_type} (confidence: {confidence})")
        print(f"Reason: {reason}")
        print(f"Target statistics: {unique_values} unique values, {unique_ratio:.3f} ratio, range: {value_range:.3f}")

        return problem_type

    except Exception as e:
        # Fallback to classification if detection fails
        print(f"Error in problem type detection: {e}")
        print("Defaulting to classification")
        return 'classification'


def optimize_random_forest(trial, X, y, problem_type, cv):
    """
    Optuna objective function for Random Forest hyperparameter optimization.
    Uses cuML RandomForest (GPU) if available, otherwise sklearn (CPU).

    Memory-safe implementation: Limits parallel jobs to prevent WSL2 crashes.
    Includes aggressive memory cleanup to prevent cuML memory leaks.

    Args:
        trial: Optuna trial object
        X: Training features
        y: Training target
        problem_type: 'regression' or 'classification'
        cv: Cross-validation splitter

    Returns:
        Mean cross-validation score
    """
    logger.info(f"[RF Trial {trial.number}] Starting Random Forest optimization trial")
    logger.info(f"[RF Trial {trial.number}] Input shapes: X={X.shape}, y={y.shape}")
    logger.info(f"[RF Trial {trial.number}] Problem type: {problem_type}")
    logger.info(f"[RF Trial {trial.number}] Using {'cuML (GPU)' if is_cuml_available() else 'sklearn (CPU)'}")

    try:
        # Suggest hyperparameters (compatible with both cuML and sklearn)
        logger.info(f"[RF Trial {trial.number}] Suggesting hyperparameters...")
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, MEMORY_CONFIG['max_trees_random_forest']),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 15),  # Reduced from 30 to prevent GPU OOM crashes
            'random_state': 42
        }
        logger.info(f"[RF Trial {trial.number}] Base params: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")

        # Add sklearn-specific parameters if using CPU (not supported by cuML)
        if not is_cuml_available():
            logger.info(f"[RF Trial {trial.number}] Adding sklearn-specific parameters...")
            params.update({
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
                'n_jobs': MEMORY_CONFIG['max_parallel_jobs']  # Limit parallel jobs to prevent memory exhaustion
            })
            logger.info(f"[RF Trial {trial.number}] Extended params: {params}")

        # Create model based on problem type using smart imports
        logger.info(f"[RF Trial {trial.number}] Creating Random Forest model...")
        if problem_type == 'regression':
            RandomForestRegressorClass = get_random_forest_regressor()
            logger.info(f"[RF Trial {trial.number}] Using {RandomForestRegressorClass.__module__}.{RandomForestRegressorClass.__name__}")
            model = RandomForestRegressorClass(**params)
            scoring = 'neg_mean_squared_error'
        else:
            RandomForestClassifierClass = get_random_forest_classifier()
            logger.info(f"[RF Trial {trial.number}] Using {RandomForestClassifierClass.__module__}.{RandomForestClassifierClass.__name__}")
            model = RandomForestClassifierClass(**params)
            scoring = 'accuracy'
        logger.info(f"[RF Trial {trial.number}] Model created successfully")

        # Evaluate with cross-validation (limit CV parallelism for memory-intensive Random Forest)
        cv_n_jobs = MEMORY_CONFIG['cv_n_jobs_limit'] if not is_cuml_available() else 1
        logger.info(f"[RF Trial {trial.number}] Starting cross-validation with {cv.get_n_splits()} folds, cv_n_jobs={cv_n_jobs}")
        logger.info(f"[RF Trial {trial.number}] Scoring metric: {scoring}")

        # Log GPU memory before CV if available
        if is_cuml_available():
            memory_gb = get_gpu_memory_usage()
            if memory_gb is not None:
                logger.info(f"[RF Trial {trial.number}] GPU memory before CV: {memory_gb:.2f}GB")

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=cv_n_jobs)
        logger.info(f"[RF Trial {trial.number}] Cross-validation completed")
        logger.info(f"[RF Trial {trial.number}] CV scores: {scores}")

        # Store result before cleanup
        mean_score = scores.mean()
        logger.info(f"[RF Trial {trial.number}] Mean score: {mean_score:.6f}")

        # Aggressive cleanup to prevent cuML memory leaks
        # cuML RandomForest stores tree metadata on CPU that isn't automatically freed
        logger.info(f"[RF Trial {trial.number}] Starting cleanup...")
        del model
        del scores

        # Force garbage collection immediately
        gc.collect()
        logger.info(f"[RF Trial {trial.number}] Garbage collection completed")

        # Clean up GPU memory if using cuML
        if is_cuml_available():
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                memory_gb = get_gpu_memory_usage()
                if memory_gb is not None:
                    logger.info(f"[RF Trial {trial.number}] GPU memory after cleanup: {memory_gb:.2f}GB")
                logger.info(f"[RF Trial {trial.number}] GPU memory cleanup completed")
            except Exception as e:
                logger.warning(f"[RF Trial {trial.number}] GPU cleanup failed: {e}")

        logger.info(f"[RF Trial {trial.number}] Trial completed successfully, returning score: {mean_score:.6f}")
        return mean_score

    except Exception as e:
        logger.error(f"[RF Trial {trial.number}] CRASHED with error: {str(e)}")
        logger.error(f"[RF Trial {trial.number}] Error type: {type(e).__name__}")
        logger.error(f"[RF Trial {trial.number}] Traceback:")
        import traceback as tb
        logger.error(tb.format_exc())
        raise


def optimize_xgboost(trial, X, y, problem_type, cv):
    """
    Optuna objective function for XGBoost hyperparameter optimization.
    Automatically uses GPU if available via gpu_utils.

    Args:
        trial: Optuna trial object
        X: Training features
        y: Training target
        problem_type: 'regression' or 'classification'
        cv: Cross-validation splitter

    Returns:
        Mean cross-validation score
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 7),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 1.0),
        'random_state': 42,
    }

    # Add GPU parameters if available
    gpu_params = get_xgboost_params()
    if gpu_params:
        params.update(gpu_params)

    # Create model based on problem type
    if problem_type == 'regression':
        model = xgb.XGBRegressor(**params)
        scoring = 'neg_mean_squared_error'
    else:
        model = xgb.XGBClassifier(**params, eval_metric='logloss')
        scoring = 'accuracy'

    # Evaluate with cross-validation
    # Note: XGBoost GPU will automatically transfer CPU data to GPU as needed
    # The warning about device mismatch is expected and can be ignored
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)  # n_jobs=1 for GPU

    # Clean up model to free memory immediately after scoring
    del model

    return scores.mean()


def optimize_linear_model(trial, X, y, problem_type, cv):
    """
    Optuna objective function for Linear/Logistic Regression hyperparameter optimization.
    Uses GPU-aware imports (cuML if available, sklearn fallback).

    Args:
        trial: Optuna trial object
        X: Training features
        y: Training target
        problem_type: 'regression' or 'classification'
        cv: Cross-validation splitter

    Returns:
        Mean cross-validation score
    """
    if problem_type == 'classification':
        # Logistic Regression hyperparameters
        params = {
            'max_iter': trial.suggest_int('lr_max_iter', 100, 2000),
            'random_state': 42
        }

        # Add sklearn-specific parameters if using CPU (not all supported by cuML)
        if not is_cuml_available():
            params.update({
                'C': trial.suggest_float('lr_C', 0.001, 10.0, log=True),
                'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2', 'elasticnet', 'none']),
                'solver': trial.suggest_categorical('lr_solver', ['lbfgs', 'saga']),
            })
            # Handle penalty-solver compatibility
            if params['penalty'] == 'elasticnet':
                params['solver'] = 'saga'
                params['l1_ratio'] = trial.suggest_float('lr_l1_ratio', 0.0, 1.0)
            elif params['penalty'] == 'l1':
                params['solver'] = 'saga'
            elif params['penalty'] == 'none':
                params['solver'] = 'lbfgs'

        LogisticRegressionClass = get_linear_model_classifier()
        model = LogisticRegressionClass(**params)
        scoring = 'accuracy'
    else:
        # Linear Regression hyperparameters (fewer to tune)
        params = {
            'fit_intercept': trial.suggest_categorical('linreg_fit_intercept', [True, False])
        }

        # Add sklearn-specific parameters
        if not is_cuml_available():
            params['positive'] = trial.suggest_categorical('linreg_positive', [True, False])

        LinearRegressionClass = get_linear_model_regressor()
        model = LinearRegressionClass(**params)
        scoring = 'neg_mean_squared_error'

    # Evaluate with cross-validation (limit parallelism for WSL2 stability)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)

    # Clean up model to free memory immediately after scoring
    del model

    return scores.mean()


def optimize_catboost(trial, X, y, problem_type, cv):
    """
    Optuna objective function for CatBoost hyperparameter optimization.
    CatBoost supports GPU acceleration natively.

    Args:
        trial: Optuna trial object
        X: Training features
        y: Training target
        problem_type: 'regression' or 'classification'
        cv: Cross-validation splitter

    Returns:
        Mean cross-validation score
    """
    use_gpu = is_catboost_gpu_available()

    # Suggest hyperparameters
    params = {
        'iterations': trial.suggest_int('cb_iterations', 300, 1200),
        'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('cb_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1.0, 20.0),
        'border_count': trial.suggest_int('cb_border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('cb_bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('cb_random_strength', 0.0, 10.0),
        'min_data_in_leaf': trial.suggest_int('cb_min_data_in_leaf', 1, 64),
        'subsample': trial.suggest_float('cb_subsample', 0.5, 1.0),
        'random_state': 42,
        'verbose': False,
        'allow_writing_files': False,
        'task_type': 'GPU' if use_gpu else 'CPU',
        'od_type': 'Iter',
        'od_wait': 30
    }
    if use_gpu:
        params['devices'] = '0'

    # Create model based on problem type
    if problem_type == 'regression':
        params.update({
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE'
        })
        model = cb.CatBoostRegressor(**params)
        scoring = 'neg_mean_squared_error'
    else:
        params.update({
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'auto_class_weights': 'Balanced'
        })
        model = cb.CatBoostClassifier(**params)
        scoring = 'accuracy'

    # Evaluate with cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)

    # Clean up model to free memory immediately after scoring
    del model

    return scores.mean()


def optimize_tabnet(trial, X, y, problem_type, cv):
    """
    Optuna objective function for TabNet hyperparameter optimization.
    TabNet is a neural network designed for tabular data with built-in GPU support.

    Args:
        trial: Optuna trial object
        X: Training features
        y: Training target
        problem_type: 'regression' or 'classification'
        cv: Cross-validation splitter

    Returns:
        Mean cross-validation score
    """
    # Suggest hyperparameters
    params = {
        'n_d': trial.suggest_int('tabnet_n_d', 8, 64),  # Width of decision prediction layer
        'n_a': trial.suggest_int('tabnet_n_a', 8, 64),  # Width of attention embedding
        'n_steps': trial.suggest_int('tabnet_n_steps', 3, 10),  # Number of steps in architecture
        'gamma': trial.suggest_float('tabnet_gamma', 1.0, 2.0),  # Feature reusage in attention
        'lambda_sparse': trial.suggest_float('tabnet_lambda_sparse', 1e-6, 1e-3, log=True),  # Sparsity regularization
        'optimizer_params': {
            'lr': trial.suggest_float('tabnet_lr', 1e-4, 1e-2, log=True)
        },
        'scheduler_params': {
            'step_size': 50,
            'gamma': 0.95
        },
        'mask_type': 'sparsemax',
        'seed': 42,
        'verbose': 0
    }

    # GPU configuration for TabNet
    if torch.cuda.is_available():
        params['device_name'] = 'cuda'
    else:
        params['device_name'] = 'cpu'

    # Create model based on problem type
    if problem_type == 'regression':
        # Use sklearn-style cross-validation but manually handle TabNet
        from sklearn.model_selection import cross_val_score
        from sklearn.base import BaseEstimator, RegressorMixin

        class TabNetRegressorWrapper(BaseEstimator, RegressorMixin):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model = None

            def fit(self, X, y):
                self.model = TabNetRegressor(**self.kwargs)
                # Convert to float32 for TabNet
                X_train = X.astype(np.float32)
                y_train = y.astype(np.float32).reshape(-1, 1)
                self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                return self

            def predict(self, X):
                X_test = X.astype(np.float32)
                return self.model.predict(X_test).flatten()

        model = TabNetRegressorWrapper(**params)
        scoring = 'neg_mean_squared_error'
    else:
        class TabNetClassifierWrapper(BaseEstimator):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model = None

            def fit(self, X, y):
                self.model = TabNetClassifier(**self.kwargs)
                X_train = X.astype(np.float32)
                y_train = y.astype(np.int64)
                self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                return self

            def predict(self, X):
                X_test = X.astype(np.float32)
                return self.model.predict(X_test)

        model = TabNetClassifierWrapper(**params)
        scoring = 'accuracy'

    # Evaluate with cross-validation
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        mean_score = scores.mean()
    except Exception as e:
        logger.warning(f"TabNet optimization failed: {e}")
        # Return very bad score to skip this trial
        mean_score = float('-inf') if problem_type == 'regression' else 0.0

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_score


def cleanup_memory(stage_name="", aggressive=False):
    """
    Force garbage collection and GPU memory cleanup to prevent memory leaks.
    Critical for preventing WSL2 crashes during long training runs.

    Args:
        stage_name: Name of the stage for logging purposes
        aggressive: If True, run multiple GC passes (useful for cuML leaks)
    """
    logger.info(f"[Cleanup] Starting cleanup for: {stage_name} (aggressive={aggressive})")

    # Log memory before cleanup
    if is_cuml_available():
        mem_before = get_gpu_memory_usage()
        if mem_before:
            logger.info(f"[Cleanup] GPU memory before cleanup: {mem_before:.2f}GB")

    if MEMORY_CONFIG['enable_gc_between_trials']:
        # Run garbage collection
        if aggressive:
            # Multiple GC passes to catch circular references and delayed cleanup
            logger.info(f"[Cleanup] Running aggressive garbage collection (3 passes)")
            for i in range(3):
                collected = gc.collect()
                logger.info(f"[Cleanup] GC pass {i+1}: collected {collected} objects")
        else:
            logger.info(f"[Cleanup] Running standard garbage collection")
            collected = gc.collect()
            logger.info(f"[Cleanup] GC collected {collected} objects")

    # Clean up GPU memory if using cuML
    if MEMORY_CONFIG['enable_gpu_memory_cleanup'] and is_cuml_available():
        try:
            import cupy as cp
            logger.info(f"[Cleanup] Freeing GPU memory pools (cuML)")
            # Clear unused GPU memory pools
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            # Log memory after cleanup
            mem_after = get_gpu_memory_usage()
            if mem_after:
                logger.info(f"[Cleanup] GPU memory after cleanup: {mem_after:.2f}GB")
                if mem_before and mem_after < mem_before:
                    freed = mem_before - mem_after
                    logger.info(f"[Cleanup] Freed {freed:.2f}GB of GPU memory")
        except Exception as e:
            logger.warning(f"[Cleanup] GPU memory cleanup failed: {e}")

    if CATBOOST_AVAILABLE:
        try:
            clear_cache = getattr(getattr(cb, '_catboost', None), 'clear_cache', None)
            if clear_cache:
                logger.info(f"[Cleanup] Clearing CatBoost internal cache")
                clear_cache()
        except Exception as e:
            logger.warning(f"[Cleanup] CatBoost cache cleanup failed: {e}")


def train_model_pipeline() -> str:
    """
    Train multiple ML models using cross-validation and select the best performer.

    This function implements a comprehensive model selection pipeline that:
    - Automatically detects problem type (regression vs classification)
    - Trains appropriate algorithms based on problem type
    - Uses cross-validation to get robust performance estimates
    - Prevents overfitting through proper validation methodology
    - Saves the best model for prediction generation

    Supports both regression and classification problems automatically.

    Returns:
        JSON string with model performance results and best model info
    """
    try:
        logger.info("="*80)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*80)

        # Load preprocessed features from feature engineering stage
        logger.info("Loading preprocessed features from feature engineering stage...")
        X_train = np.load(FEATURES_DIR / 'X_train.npy')
        y_train = np.load(FEATURES_DIR / 'y_train.npy')
        logger.info(f"Loaded training data: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        logger.info(f"Data types: X_train.dtype={X_train.dtype}, y_train.dtype={y_train.dtype}")

        # Automatically detect problem type
        logger.info("Detecting problem type...")
        problem_type = detect_problem_type()
        logger.info(f"Problem type detected: {problem_type}")

        # Get GPU configuration for logging
        gpu_config = get_gpu_config()
        logger.info(f"GPU Configuration: {gpu_config}")
        log_gpu_memory("Before Training")

        if LIGHTGBM_AVAILABLE:
            print("✅ LightGBM detected - adding LightGBM model to ensemble.")
        else:
            print("ℹ️ LightGBM not installed - install `lightgbm` to enable that model option.")

        if CATBOOST_AVAILABLE:
            print("✅ CatBoost detected - adding CatBoost model to ensemble.")
        else:
            print("ℹ️ CatBoost not installed - install `catboost` to enable that model option.")

        if TABNET_AVAILABLE:
            print("✅ TabNet detected - adding TabNet neural network to ensemble.")
        else:
            print("ℹ️ TabNet not installed - install `pytorch-tabnet` to enable that model option.")

        ai_override_details, ai_override_metadata = load_ai_tuning_overrides()
        ai_model_overrides = extract_override_values(ai_override_details)
        ai_override_updates: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if ai_model_overrides:
            override_summary = []
            for model_name, params in ai_model_overrides.items():
                if params:
                    formatted_params = ', '.join(f"{k}={v}" for k, v in params.items())
                    override_summary.append(f"{model_name}({formatted_params})")
            if override_summary:
                print(f"🤖 Applying AI tuning overrides: {', '.join(override_summary)}")
        else:
            ai_override_details = {}
            ai_override_metadata = {}

        original_models_to_tune = list(TUNING_CONFIG['models_to_tune'])

        # Get GPU-aware model classes using smart imports
        LinearRegressionClass = get_linear_model_regressor()
        LogisticRegressionClass = get_linear_model_classifier()
        RandomForestRegressorClass = get_random_forest_regressor()
        RandomForestClassifierClass = get_random_forest_classifier()

        def make_factory(constructor, base_params=None):
            """
            Create a factory that instantiates estimators with optional overrides.

            Using factories keeps heavy estimators out of long-lived scopes so they
            can be garbage collected promptly, which helps avoid GPU memory leaks.
            """
            base_params = base_params or {}

            def factory(overrides=None):
                params = base_params.copy()
                if overrides:
                    params.update(overrides)
                return constructor(**params)

            return factory

        def make_xgb_regressor_factory():
            def factory(overrides=None):
                params = {'random_state': 42}
                params.update(get_xgboost_params())
                if overrides:
                    params.update(overrides)
                return xgb.XGBRegressor(**params)

            return factory

        def make_xgb_classifier_factory():
            def factory(overrides=None):
                params = {'random_state': 42, 'eval_metric': 'logloss'}
                params.update(get_xgboost_params())
                if overrides:
                    params.update(overrides)
                return xgb.XGBClassifier(**params)

            return factory

        def make_catboost_factory(constructor):
            def factory(overrides=None):
                params = {
                    'iterations': 400,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'random_state': 42,
                    'verbose': False,
                    'allow_writing_files': False
                }
                if constructor == cb.CatBoostRegressor:
                    params.update({
                        'loss_function': 'RMSE',
                        'eval_metric': 'RMSE'
                    })
                else:
                    params.update({
                        'loss_function': 'Logloss',
                        'eval_metric': 'AUC'
                    })

                if is_catboost_gpu_available():
                    params.update({'task_type': 'GPU', 'devices': '0'})
                else:
                    params.update({'task_type': 'CPU'})

                if overrides:
                    params.update(overrides)
                return constructor(**params)

            return factory

        def make_tabnet_factory(constructor):
            """Factory for TabNet models with sklearn-compatible wrappers."""
            def factory(overrides=None):
                from sklearn.base import BaseEstimator, RegressorMixin

                # Default TabNet parameters
                params = {
                    'n_d': 32,
                    'n_a': 32,
                    'n_steps': 5,
                    'gamma': 1.5,
                    'lambda_sparse': 1e-4,
                    'optimizer_params': {'lr': 2e-3},
                    'scheduler_params': {'step_size': 50, 'gamma': 0.95},
                    'mask_type': 'sparsemax',
                    'seed': 42,
                    'verbose': 0,
                    'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
                if overrides:
                    params.update(overrides)

                # Create sklearn-compatible wrapper
                if constructor == TabNetRegressor:
                    class TabNetRegressorWrapper(BaseEstimator, RegressorMixin):
                        def __init__(self, **kwargs):
                            self.kwargs = kwargs
                            self.model = None

                        def fit(self, X, y):
                            self.model = TabNetRegressor(**self.kwargs)
                            X_train = X.astype(np.float32)
                            y_train = y.astype(np.float32).reshape(-1, 1)
                            self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                            return self

                        def predict(self, X):
                            X_test = X.astype(np.float32)
                            return self.model.predict(X_test).flatten()

                    return TabNetRegressorWrapper(**params)
                else:
                    class TabNetClassifierWrapper(BaseEstimator):
                        def __init__(self, **kwargs):
                            self.kwargs = kwargs
                            self.model = None

                        def fit(self, X, y):
                            self.model = TabNetClassifier(**self.kwargs)
                            X_train = X.astype(np.float32)
                            y_train = y.astype(np.int64)
                            self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                            return self

                        def predict(self, X):
                            X_test = X.astype(np.float32)
                            return self.model.predict(X_test)

                    return TabNetClassifierWrapper(**params)

            return factory

        # Define model ensemble based on problem type
        # Each model has different strengths and biases
        # Models automatically use GPU (cuML) if available, otherwise CPU (sklearn)
        if problem_type == 'regression':
            models = {
                'Linear Regression': make_factory(LinearRegressionClass),  # Linear, interpretable
                'Random Forest': make_factory(
                    RandomForestRegressorClass,
                    {'n_estimators': 100, 'random_state': 42}
                ),  # Non-linear, robust (GPU if cuML available)
                'XGBoost': make_xgb_regressor_factory()  # Gradient boosting, high performance (GPU if available)
            }
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = make_factory(
                    lgb.LGBMRegressor,
                    {
                        'random_state': 42,
                        'n_estimators': 200,
                        'learning_rate': 0.05,
                        'n_jobs': -1
                    }
                )
            if CATBOOST_AVAILABLE:
                models['CatBoost'] = make_catboost_factory(cb.CatBoostRegressor)
            if TABNET_AVAILABLE:
                models['TabNet'] = make_tabnet_factory(TabNetRegressor)  # Neural network for tabular data (GPU if available)
            # Use regular KFold for regression (no need to preserve class distribution)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            # Use regression scoring metric
            scoring_metric = 'neg_mean_squared_error'
            best_score = float('-inf')  # Higher (less negative) is better for MSE
        else:  # classification
            models = {
                'Logistic Regression': make_factory(
                    LogisticRegressionClass,
                    {'random_state': 42, 'max_iter': 1000}
                ),  # Linear, interpretable
                'Random Forest': make_factory(
                    RandomForestClassifierClass,
                    {'n_estimators': 100, 'random_state': 42}
                ),  # Non-linear, robust (GPU if cuML available)
                'XGBoost': make_xgb_classifier_factory()  # Gradient boosting, high performance (GPU if available)
            }
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = make_factory(
                    lgb.LGBMClassifier,
                    {
                        'random_state': 42,
                        'n_estimators': 200,
                        'learning_rate': 0.05,
                        'n_jobs': -1,
                        'objective': 'binary'
                    }
                )
            if CATBOOST_AVAILABLE:
                models['CatBoost'] = make_catboost_factory(cb.CatBoostClassifier)
            if TABNET_AVAILABLE:
                models['TabNet'] = make_tabnet_factory(TabNetClassifier)  # Neural network for tabular data (GPU if available)
            # Use StratifiedKFold to preserve class distribution in each fold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Use classification scoring metric
            scoring_metric = 'accuracy'
            best_score = 0  # Higher is better for accuracy

        model_selection_summary = {'applied': False}
        guidance_payload = load_model_selection_guidance()
        ordered_models, tuned_models_list, selection_summary = apply_model_selection_guidance(
            models,
            guidance_payload,
            original_models_to_tune
        )
        models = ordered_models
        if tuned_models_list is not None:
            TUNING_CONFIG['models_to_tune'] = tuned_models_list
        if selection_summary.get('applied'):
            model_selection_summary = selection_summary
            if selection_summary.get('recommended_order'):
                print(f"🤖 Model advisor recommended order: {selection_summary['recommended_order']}")
            if selection_summary.get('excluded_models'):
                print(f"   Skipping models per advisor guidance: {selection_summary['excluded_models']}")

        def evaluate_with_default_params(model_name, factory_fn, overrides=None):
            """
            Run cross-validation on the estimator built from the factory.
            The estimator is created inside this helper so it can be freed immediately.
            """
            import warnings

            estimator = factory_fn(overrides)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
                    # Use n_jobs=1 for XGBoost GPU (GPU parallelizes internally)
                    n_jobs_param = 1 if 'XGBoost' in model_name and is_xgboost_gpu_available() else -1
                    return cross_val_score(
                        estimator,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scoring_metric,
                        n_jobs=n_jobs_param
                    )
            finally:
                # Explicitly delete estimator before cleanup to release GPU memory
                del estimator

        # Model evaluation pipeline with Optuna hyperparameter tuning
        # Systematically evaluate each model using cross-validation with optimized hyperparameters
        results = {}
        best_model_name = None
        best_model_factory = None
        best_model_params = None
        best_model_object = None
        tuned_params = {}  # Store best hyperparameters for each model

        # Suppress Optuna's default logging to reduce clutter
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        for name, model_factory in models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            logger.info(f"[Training] Starting training for {name}")
            logger.info(f"[Training] Training data shape: X={X_train.shape}, y={y_train.shape}")

            model_override_values = ai_model_overrides.get(name, {})
            used_params = None
            model_mean_score = None
            model_std_score = 0.0
            model_scores_list = []
            tuned_flag = False

            # Check if this model should be tuned
            if TUNING_CONFIG['enabled'] and name in TUNING_CONFIG['models_to_tune']:
                print(f"🔧 Tuning hyperparameters for {name}...")
                print(f"   Strategy: Optuna Bayesian Optimization")
                print(f"   Trials: {TUNING_CONFIG['n_trials']}")
                logger.info(f"[Training] {name} - Hyperparameter tuning enabled with {TUNING_CONFIG['n_trials']} trials")

                # Create Optuna study for hyperparameter optimization
                study = optuna.create_study(
                    direction='maximize',  # Maximize score (works for both neg_mse and accuracy)
                    sampler=TPESampler(seed=42)
                )
                logger.info(f"[Training] {name} - Optuna study created")

                # Select appropriate optimization function
                if name == 'Random Forest':
                    logger.info(f"[Training] Random Forest - Entering hyperparameter optimization")
                    logger.info(f"[Training] Random Forest - Memory config: max_trees={MEMORY_CONFIG['max_trees_random_forest']}, cv_folds={MEMORY_CONFIG['rf_cv_folds']}, aggressive_cleanup={MEMORY_CONFIG['aggressive_rf_cleanup']}")
                    objective = lambda trial: optimize_random_forest(trial, X_train, y_train, problem_type, cv)
                elif name == 'XGBoost':
                    objective = lambda trial: optimize_xgboost(trial, X_train, y_train, problem_type, cv)
                elif name == 'CatBoost':
                    objective = lambda trial: optimize_catboost(trial, X_train, y_train, problem_type, cv)
                elif name == 'TabNet':
                    objective = lambda trial: optimize_tabnet(trial, X_train, y_train, problem_type, cv)
                elif name == 'Logistic Regression' or name == 'Linear Regression':
                    objective = lambda trial: optimize_linear_model(trial, X_train, y_train, problem_type, cv)
                else:
                    # Fallback for any other models
                    objective = None

                if objective:
                    # Run hyperparameter optimization
                    logger.info(f"[Training] {name} - Starting Optuna optimization (n_trials={TUNING_CONFIG['n_trials']}, timeout={TUNING_CONFIG['timeout']}s)")
                    try:
                        study.optimize(
                            objective,
                            n_trials=TUNING_CONFIG['n_trials'],
                            timeout=TUNING_CONFIG['timeout'],
                            show_progress_bar=TUNING_CONFIG['show_progress_bar'],
                            n_jobs=1  # Serial execution for stability
                        )
                        logger.info(f"[Training] {name} - Optimization completed successfully")
                    except Exception as e:
                        logger.error(f"[Training] {name} - Optimization FAILED: {str(e)}")
                        logger.error(f"[Training] {name} - Error type: {type(e).__name__}")
                        import traceback as tb
                        logger.error(f"[Training] {name} - Traceback:\n{tb.format_exc()}")
                        raise

                    # Clean up memory after optimization to prevent leaks
                    # Use aggressive cleanup for Random Forest to combat cuML memory leaks
                    is_rf = name == 'Random Forest'
                    logger.info(f"[Training] {name} - Starting memory cleanup (aggressive={is_rf})")
                    cleanup_memory(f"After {name} optimization", aggressive=is_rf)
                    logger.info(f"[Training] {name} - Memory cleanup completed")

                    # Extract best parameters
                    best_params = study.best_params
                    tuned_params[name] = best_params

                    # Remove model-specific prefixes from parameter names
                    prefix_map = {
                        'Random Forest': 'rf_',
                        'XGBoost': 'xgb_',
                        'CatBoost': 'cb_',
                        'TabNet': 'tabnet_',
                        'Logistic Regression': 'lr_',
                        'Linear Regression': 'linreg_'
                    }
                    prefix = prefix_map.get(name, '')
                    clean_params = {k.replace(prefix, ''): v for k, v in best_params.items()}

                    applied_overrides = model_override_values
                    if applied_overrides:
                        clean_params.update(applied_overrides)

                    # Use best score from optimization
                    best_trial_score = study.best_value
                    trial_scores = [trial.value for trial in study.trials if trial.value is not None]

                    print(f"   ✅ Optimization complete!")
                    print(f"   Best score: {best_trial_score:.6f}")
                    print(f"   Best params: {clean_params}")

                    model_mean_score = float(best_trial_score)
                    model_std_score = float(np.std(trial_scores[-5:])) if len(trial_scores) >= 5 else 0.0
                    model_scores_list = trial_scores[-5:]  # Last 5 trial scores
                    used_params = clean_params
                    tuned_flag = True

                    results[name] = {
                        'mean_score': model_mean_score,
                        'std_score': model_std_score,
                        'individual_scores': model_scores_list,
                        'best_params': clean_params,
                        'n_trials': len(study.trials),
                        'tuned': True,
                        'ai_overrides': applied_overrides
                    }

                    # Ensure Optuna study objects don't linger in memory
                    del study
                else:
                    # Model doesn't have tuning function, use default
                    print(f"   ⚠️  No tuning function available for {name}, using defaults")
                    scores = evaluate_with_default_params(name, model_factory, ai_model_overrides.get(name))
                    model_mean_score = float(scores.mean())
                    model_std_score = float(scores.std())
                    model_scores_list = scores.tolist()
                    results[name] = {
                        'mean_score': model_mean_score,
                        'std_score': model_std_score,
                        'individual_scores': model_scores_list,
                        'tuned': False,
                        'ai_overrides': model_override_values
                    }
                    print(f"   Score: {model_mean_score:.6f} (+/- {model_std_score:.6f})")

            else:
                # Model tuning disabled or not in tuning list - use defaults
                print(f"   Using default hyperparameters...")

                scores = evaluate_with_default_params(name, model_factory, model_override_values)
                model_mean_score = float(scores.mean())
                model_std_score = float(scores.std())
                model_scores_list = scores.tolist()

                results[name] = {
                    'mean_score': model_mean_score,
                    'std_score': model_std_score,
                    'individual_scores': model_scores_list,
                    'tuned': False,
                    'ai_overrides': model_override_values
                }

                print(f"   Score: {model_mean_score:.6f} (+/- {model_std_score:.6f})")

            if model_mean_score is not None and model_mean_score > best_score:
                best_score = model_mean_score
                best_model_name = name
                best_model_factory = model_factory
                best_model_params = used_params
                best_model_object = None

            # Clean up memory after each model evaluation
            # Use aggressive cleanup for Random Forest to combat cuML memory leaks
            is_rf = name == 'Random Forest'
            logger.info(f"[Training] {name} - Starting post-evaluation cleanup (aggressive={is_rf})")
            cleanup_memory(f"After {name} evaluation", aggressive=is_rf)
            logger.info(f"[Training] {name} - Post-evaluation cleanup completed")
            logger.info(f"[Training] {name} - Training completed")

        # Create stacking ensemble of top models
        print(f"\n{'='*60}")
        print(f"Creating Stacking Ensemble...")
        print(f"{'='*60}")

        # Sort models by score and select top performers for ensemble
        sorted_models = sorted(results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        top_models = [m for m in sorted_models[:min(3, len(sorted_models))]]  # Top 3 models

        if len(top_models) >= 2:
            print(f"Ensemble members (base models): {[m[0] for m in top_models]}")

            # Create ensemble estimators list
            ensemble_estimators = []

            for model_name, model_results in top_models:
                factory = models.get(model_name)
                if not factory:
                    logger.warning(f"[Ensemble] Missing factory for {model_name}, skipping")
                    continue

                overrides = model_results.get('best_params') if model_results.get('tuned') else None
                m = factory(overrides)

                ensemble_estimators.append((model_name.lower().replace(' ', '_'), m))

            if len(ensemble_estimators) < 2:
                print("Not enough valid models for ensembling after filtering.")
                cleanup_memory("After ensemble evaluation", aggressive=False)
            else:
                # Create stacking ensemble with meta-learner
                # Stacking learns optimal combination vs fixed weights in voting
                if problem_type == 'regression':
                    # Use Ridge as meta-learner for regression
                    meta_learner = Ridge(alpha=1.0, random_state=42)
                    ensemble = StackingRegressor(
                        estimators=ensemble_estimators,
                        final_estimator=meta_learner,
                        cv=5,  # Internal CV for generating meta-features
                        n_jobs=1  # Serial for GPU stability
                    )
                    print(f"Meta-learner: Ridge Regression")
                else:
                    # Use LogisticRegression as meta-learner for classification
                    LogisticClass = get_linear_model_classifier()
                    meta_learner = LogisticClass(random_state=42, max_iter=1000)
                    ensemble = StackingClassifier(
                        estimators=ensemble_estimators,
                        final_estimator=meta_learner,
                        cv=5,  # Internal CV for generating meta-features
                        n_jobs=1  # Serial for GPU stability
                    )
                    print(f"Meta-learner: Logistic Regression")

                # Evaluate stacking ensemble with cross-validation
                print("Evaluating stacking ensemble...")
                ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=1)
                ensemble_mean_score = ensemble_scores.mean()

                print(f"Stacking Ensemble CV Score: {ensemble_mean_score:.6f} (+/- {ensemble_scores.std():.6f})")

                # Add ensemble to results
                results['Stacking Ensemble'] = {
                    'mean_score': float(ensemble_mean_score),
                    'std_score': float(ensemble_scores.std()),
                    'individual_scores': ensemble_scores.tolist(),
                    'members': [m[0] for m in top_models],
                    'meta_learner': type(meta_learner).__name__,
                    'tuned': False
                }

                # Check if ensemble beats best individual model
                if ensemble_mean_score > best_score:
                    print(f"✅ Stacking Ensemble improves over best individual model!")
                    print(f"   Improvement: {((ensemble_mean_score - best_score) / abs(best_score) * 100):.2f}%")
                    best_score = ensemble_mean_score
                    best_model_name = 'Stacking Ensemble'
                    best_model_factory = None
                    best_model_params = None
                    best_model_object = ensemble
                else:
                    print(f"Individual model {best_model_name} still wins")
                    print(f"   Difference: {((best_score - ensemble_mean_score) / abs(best_score) * 100):.2f}%")
                    # Release ensemble estimators aggressively when not selected
                    del ensemble

                cleanup_memory("After ensemble evaluation", aggressive=False)
        else:
            print("Not enough models for ensembling (need at least 2)")

        # Train the selected model on full training dataset
        print(f"\n{'='*60}")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"   Score: {best_score:.6f}")
        if best_model_name in tuned_params:
            print(f"   Tuned Parameters: {tuned_params[best_model_name]}")
        print(f"{'='*60}\n")

        # Log GPU status for transparency
        if is_cuml_available():
            print(f"ℹ️  Using GPU acceleration (cuML) for {best_model_name}")
        elif is_xgboost_gpu_available() and 'XGBoost' in best_model_name:
            print(f"ℹ️  Using GPU acceleration (XGBoost) for {best_model_name}")
        else:
            print(f"ℹ️  Using CPU for {best_model_name}")

        if best_model_name == 'Stacking Ensemble':
            best_model = best_model_object
        else:
            if best_model_factory is None:
                raise RuntimeError(f"No factory available to instantiate best model '{best_model_name}'")

            overrides_for_best = ai_model_overrides.get(best_model_name, {})
            if best_model_params:
                final_params = best_model_params.copy()
                if overrides_for_best:
                    final_params.update(overrides_for_best)
            else:
                final_params = overrides_for_best if overrides_for_best else None

            best_model = best_model_factory(final_params)

        # This gives the model access to all available training data
        log_gpu_memory("Before Final Training")
        best_model.fit(X_train, y_train)
        log_gpu_memory("After Final Training")

        # AI Advisor: Error Pattern Analysis
        # Use AI to analyze prediction errors and identify improvement opportunities
        if AI_ADVISORS_CONFIG['enabled'] and AI_ADVISORS_CONFIG['error_analysis']['enabled']:
            try:
                print(f"\n{'='*60}")
                print("🤖 AI Error Pattern Analyzer")
                print(f"{'='*60}")
                print("Analyzing prediction errors to identify patterns...")

                from src.genML.ai_advisors import ErrorPatternAnalyzer, OpenAIClient

                # Create OpenAI client with configuration
                openai_client = OpenAIClient(config=AI_ADVISORS_CONFIG['openai_config'])

                if openai_client.is_available():
                    # Generate predictions on training data for error analysis
                    y_pred_train = best_model.predict(X_train)

                    # Load feature names for detailed analysis
                    feature_names_file = FEATURES_DIR / 'feature_names.txt'
                    if feature_names_file.exists():
                        with open(feature_names_file, 'r') as f:
                            feature_names = [line.strip() for line in f.readlines()]
                    else:
                        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

                    # Create analyzer and perform analysis
                    analyzer = ErrorPatternAnalyzer(openai_client=openai_client)

                    error_report = analyzer.analyze_errors(
                        X=X_train,
                        y_true=y_train,
                        y_pred=y_pred_train,
                        feature_names=feature_names,
                        model_type=best_model_name,
                        top_n_errors=AI_ADVISORS_CONFIG['error_analysis']['top_n_errors']
                    )

                    # Display analysis summary
                    if error_report.get('status') == 'success':
                        error_stats = error_report['error_statistics']
                        print(f"✅ AI Error Analysis completed!")
                        print(f"   Mean Absolute Error: {error_stats['mean_absolute_error']:.6f}")
                        print(f"   RMSE: {error_stats['rmse']:.6f}")
                        print(f"   Max Error: {error_stats['max_error']:.6f}")

                        # Display key findings from AI analysis
                        ai_suggestions = error_report.get('ai_suggestions', {})

                        if ai_suggestions.get('error_patterns_detected'):
                            print(f"\n   🔍 Key Error Patterns Detected:")
                            for i, pattern in enumerate(ai_suggestions['error_patterns_detected'][:3], 1):
                                print(f"      {i}. {pattern}")

                        if ai_suggestions.get('priority_actions'):
                            print(f"\n   💡 Priority Actions:")
                            for i, action in enumerate(ai_suggestions['priority_actions'][:3], 1):
                                print(f"      {i}. {action}")

                        # Display top feature-error correlations
                        feature_corrs = error_report.get('feature_error_correlations', {})
                        if feature_corrs:
                            print(f"\n   📊 Top Features Correlated with Errors:")
                            for i, (feat, corr) in enumerate(list(feature_corrs.items())[:5], 1):
                                print(f"      {i}. {feat}: {corr:+.3f}")

                        # Save report if configured
                        if AI_ADVISORS_CONFIG['error_analysis']['save_report']:
                            report_path = REPORTS_DIR / 'ai_error_analysis.json'
                            analyzer.save_report(error_report, str(report_path))
                            print(f"\n   📄 Report saved to: {report_path}")

                        # Display API usage stats
                        usage_stats = openai_client.get_usage_stats()
                        print(f"\n   💰 API Cost: ${usage_stats['total_cost']:.4f}")
                        print(f"   📊 Tokens: {usage_stats['total_tokens']:,} ({usage_stats['input_tokens']:,} in, {usage_stats['output_tokens']:,} out)")

                        new_override_details = build_ai_tuning_override_details(
                            ai_suggestions.get('tuning_recommendations', [])
                        )
                        if new_override_details:
                            merged_details = copy.deepcopy(ai_override_details)
                            for model_key, param_map in new_override_details.items():
                                merged_details.setdefault(model_key, {}).update(param_map)
                            ai_override_details = merged_details
                            ai_override_updates = new_override_details
                            ai_model_overrides = extract_override_values(ai_override_details)
                            print(f"\n   🔁 Recorded AI tuning overrides for: {', '.join(new_override_details.keys())}")
                    else:
                        print(f"⚠️  AI Error Analysis returned an error")
                else:
                    print("⚠️  OpenAI API not available - skipping error analysis")
                    print("   Set OPENAI_API_KEY environment variable to enable AI advisors")

                print(f"{'='*60}\n")

            except Exception as e:
                logger.warning(f"AI Error Analysis failed: {e}")
                print(f"⚠️  AI Error Analysis failed: {e}")
                print("Continuing with model saving...\n")

        # Save the trained model for prediction stage
        # Timestamped filename prevents overwrites and enables model versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'best_model_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl'

        if ai_override_updates:
            save_ai_tuning_overrides(
                ai_override_details,
                {
                    'problem_type': problem_type,
                    'best_model': best_model_name,
                    'best_score': float(best_score),
                    'model_filename': model_filename
                }
            )

        joblib.dump(best_model, MODELS_DIR / model_filename)

        # Save detailed cross-validation results for analysis
        # Provides transparency into model selection process
        cv_results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Mean_Score': [results[name]['mean_score'] for name in results.keys()],
            'Std_Score': [results[name]['std_score'] for name in results.keys()]
        })
        cv_results_df.to_csv(MODELS_DIR / 'cross_validation_results.csv', index=False)

        # Prepare results with problem type and GPU information
        model_results = {
            "status": "success",
            "problem_type": problem_type,
            "scoring_metric": scoring_metric,
            "best_model": best_model_name,
            "best_score": float(best_score),
            "all_results": results,
            "model_filename": model_filename,
            "timestamp": timestamp,
            "model_saved": str(MODELS_DIR / model_filename),
            "ai_model_selection_guidance": guidance_payload,
            "ai_model_selection_summary": model_selection_summary,
            "ai_tuning_overrides_loaded": ai_override_details,
            "ai_tuning_overrides_metadata": ai_override_metadata,
            "ai_tuning_overrides_updates": ai_override_updates,
            "ai_model_overrides_used": ai_model_overrides,
            "gpu_acceleration": {
                "cuml_available": is_cuml_available(),
                "xgboost_gpu_available": is_xgboost_gpu_available(),
                "gpu_used_for_best_model": is_cuml_available() or (is_xgboost_gpu_available() and 'XGBoost' in best_model_name),
                "gpu_info": gpu_config
            }
        }

        # Save model training report
        with open(REPORTS_DIR / 'model_training_report.json', 'w') as f:
            json.dump(model_results, f, indent=2)

        TUNING_CONFIG['models_to_tune'] = original_models_to_tune
        return json.dumps(model_results, indent=2)

    except Exception as e:
        try:
            TUNING_CONFIG['models_to_tune'] = original_models_to_tune
        except Exception:
            pass
        return json.dumps({
            "error": f"Error in model training: {str(e)}",
            "status": "failed"
        })


def generate_predictions() -> str:
    """
    Generate predictions using the trained model and create adaptive submission file.

    This final stage of the ML pipeline uses the best model selected during training
    to generate predictions on the test dataset. It automatically detects the expected
    submission format from sample submission files and creates properly formatted outputs.

    Key functions:
    - Load the best performing model from training stage
    - Generate predictions on preprocessed test data
    - Auto-detect submission format from sample submission files
    - Create properly formatted submission file matching expected format
    - Provide prediction confidence metrics and statistics

    Returns:
        JSON string with prediction results and submission file info
    """
    try:
        # Load preprocessed test features
        X_test = np.load(FEATURES_DIR / 'X_test.npy')

        # Load the most recently trained model
        # This ensures we use the latest model from the training stage
        model_files = list(MODELS_DIR.glob('best_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError("No trained model found in models directory")

        # Select the most recent model file (highest timestamp)
        latest_model = max(model_files, key=os.path.getctime)
        best_model = joblib.load(latest_model)

        # Load original test data for ID columns and submission formatting
        test_df = pd.read_pickle(DATA_DIR / 'test_data.pkl')

        # Generate model predictions on test data
        predictions = best_model.predict(X_test)                    # Predictions (class labels or continuous values)

        # Apply inverse target transformation if needed
        transform_info = joblib.load(FEATURES_DIR / 'target_transform.pkl')
        if transform_info['applied']:
            print(f"Applying inverse {transform_info['type']} transformation to predictions")
            print(f"Predictions range before inverse transform: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")

            # Inverse logit: y = exp(logit_y) / (1 + exp(logit_y))
            # Equivalent to: y = 1 / (1 + exp(-logit_y))  (more numerically stable)
            predictions = 1 / (1 + np.exp(-predictions))

            # Clip to original bounds for safety
            epsilon = transform_info['epsilon']
            predictions = np.clip(predictions, epsilon, 1 - epsilon)

            print(f"Predictions range after inverse transform: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")

        # Generate prediction probabilities/confidence scores if available
        # Only classification models have predict_proba method
        try:
            if hasattr(best_model, 'predict_proba'):
                prediction_probabilities = best_model.predict_proba(X_test)[:, 1]  # Confidence scores for positive class
            else:
                # For regression models, use the predictions themselves as "confidence"
                prediction_probabilities = predictions
        except Exception as e:
            # Fallback: use predictions as probabilities
            print(f"Warning: Could not generate prediction probabilities: {e}")
            prediction_probabilities = predictions

        # Use adaptive submission formatting based on sample submission files
        # This automatically detects the expected format and creates matching submissions
        from src.genML.submission_formatter import create_adaptive_submission

        submission_result = create_adaptive_submission(
            test_df=test_df,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            project_dir="."
        )

        submission_df = submission_result['submission_df']
        main_file = submission_result['main_file']
        timestamped_file = submission_result['timestamped_file']
        format_info = submission_result['format_info']

        # Save timestamped submission file to predictions directory as well
        # This maintains the existing pipeline structure for organization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predictions_submission_file = f'submission_{timestamp}.csv'
        submission_df.to_csv(PREDICTIONS_DIR / predictions_submission_file, index=False)

        # Save detailed predictions with confidence scores
        # Use detected column names for consistency
        id_col = format_info['id_column']
        detailed_predictions = pd.DataFrame({
            id_col: test_df[id_col] if id_col in test_df.columns else test_df.iloc[:, 0],
            'Prediction': predictions,
            'Confidence': prediction_probabilities  # Probability of positive class
        })
        detailed_predictions.to_csv(PREDICTIONS_DIR / f'detailed_predictions_{timestamp}.csv', index=False)

        # Compile comprehensive prediction results with format information
        submission_results = {
            "status": "success",
            "submission_file": main_file,                    # Main submission file
            "submission_filename": predictions_submission_file,  # Timestamped backup in predictions dir
            "timestamped_file": timestamped_file,            # Timestamped file in root
            "timestamp": timestamp,
            "model_used": str(latest_model),                 # Model file path for traceability
            "predictions_count": len(predictions),           # Total predictions made
            "predicted_positive": int(np.sum(predictions)),  # Count of positive predictions
            "predicted_negative": int(len(predictions) - np.sum(predictions)),  # Count of negative predictions
            "predicted_positive_rate": float(np.mean(predictions)),  # Overall prediction rate
            "confidence_stats": {                            # Model confidence analysis
                "min_confidence": float(prediction_probabilities.min()),
                "max_confidence": float(prediction_probabilities.max()),
                "mean_confidence": float(prediction_probabilities.mean())
            },
            "format_detection": {                            # Information about detected format
                "id_column": format_info['id_column'],
                "target_column": format_info['target_column'],
                "value_type": format_info['value_type'],
                "source_file": format_info['source_file'],
                "total_expected_rows": format_info['total_rows']
            },
            "submission_preview": submission_df.head(10).to_dict('records')  # Sample for verification
        }

        # Save comprehensive prediction report for analysis
        # Documents the prediction process and results for future reference
        with open(REPORTS_DIR / 'prediction_report.json', 'w') as f:
            json.dump(submission_results, f, indent=2)

        return json.dumps(submission_results, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error in prediction: {str(e)}",
            "status": "failed"
        })
