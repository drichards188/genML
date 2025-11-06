"""Configuration and shared constants for the genML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Directory structure for organizing pipeline outputs. The directories are created at import time
# so downstream stages can rely on them existing.
OUTPUTS_DIR = Path("outputs")
DATA_DIR = OUTPUTS_DIR / "data"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for directory in (OUTPUTS_DIR, DATA_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Global progress tracker instance (set by flow.py)
PROGRESS_TRACKER: Optional[Any] = None


# Hyperparameter Tuning Configuration
TUNING_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "n_trials": 100,
    "timeout": 600,
    "n_jobs": -1,
    "show_progress_bar": True,
    "models_to_tune": [
        "Random Forest",
        "XGBoost",
        "CatBoost",
        "TabNet",
        "Logistic Regression",
        "Linear Regression",
    ],
}

# GPU Acceleration Configuration
GPU_CONFIG: Dict[str, Any] = {
    "force_cpu": False,
    "enable_cuml": True,
    "enable_xgboost_gpu": True,
    "log_gpu_memory": True,
    "gpu_memory_threshold_gb": 14.0,
}

# Memory Management Configuration (Prevents WSL2 crashes and memory leaks)
MEMORY_CONFIG: Dict[str, Any] = {
    "max_parallel_jobs": 1,
    "enable_gc_between_trials": True,
    "enable_gpu_memory_cleanup": True,
    "cv_n_jobs_limit": 1,
    "max_trees_random_forest": 100,
    "rf_cv_folds": 3,
    "aggressive_rf_cleanup": True,
}

# Parallel Training Configuration
PARALLEL_TRAINING_CONFIG: Dict[str, Any] = {
    "enabled": True,  # Enable parallel training for CPU models
    "max_cpu_workers": 3,  # Max concurrent CPU models (0 = auto-detect, limited to 3)
    "cpu_optuna_n_jobs": 2,  # Optuna parallel trials for CPU models (0 = serial)
    "gpu_models_sequential": True,  # Always keep GPU models sequential to avoid OOM
}

# AI Advisors Configuration (OpenAI-powered intelligent analysis)
AI_ADVISORS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "feature_ideation": {
        "enabled": True,
        "sample_size": 100,
        "save_report": True,
    },
    "error_analysis": {
        "enabled": True,
        "top_n_errors": 100,
        "save_report": True,
    },
    "openai_config": {
        "model": "gpt-4o",  # Upgraded from gpt-4o-mini for better reasoning
        "max_cost_per_run": 10.0,
        "enable_cache": True,
        "cache_dir": "outputs/ai_cache",
    },
}

AI_SUPPORTED_FEATURE_OPERATIONS = {
    "ratio",
    "difference",
    "product",
    "sum",
    "log",
    "binary_threshold",
}
AI_FEATURE_EPSILON = 1e-6

AI_TUNING_OVERRIDE_PATH = REPORTS_DIR / "ai_tuning_overrides.json"

# Data Ingestion Configuration (for non-Kaggle datasets)
# Set INGESTION_CONFIG to None to use legacy CSV discovery mode
INGESTION_CONFIG: Optional[Dict[str, Any]] = None

# Example ingestion configurations (uncomment and customize as needed):

# # Example 1: PostgreSQL Database
# INGESTION_CONFIG = {
#     'data_source': {
#         'type': 'postgresql',
#         'connection_string': 'postgresql://user:password@localhost:5432/database',
#         'query': 'SELECT * FROM customers WHERE created_at >= \'2024-01-01\'',
#         # Or use table instead of query:
#         # 'table': 'customers',
#         # 'schema': 'public',
#     },
#     'target_column': 'churn',  # Column to predict
#     'id_column': 'customer_id',  # Optional: ID column for submissions
#     'split': {
#         'method': 'random',  # Options: 'random', 'time', 'custom'
#         'test_size': 0.2,
#         'stratify': True,  # Stratify split by target (for classification)
#         'random_state': 42,
#     },
#     'cleaning': {
#         'drop_duplicates': True,
#         'missing_strategy': 'auto',  # Options: 'auto', 'drop', 'none'
#         # Optional: Specify fill values for specific columns
#         # 'missing_fill_values': {'age': 0, 'name': 'Unknown'},
#     },
#     'validation': {
#         'required_columns': ['customer_id', 'churn'],
#         'column_types': {
#             'customer_id': 'int',
#             'churn': 'int',
#         },
#     },
#     'transformations': [
#         # Optional: Add custom transformations
#         # {'type': 'drop_columns', 'columns': ['internal_id']},
#         # {'type': 'rename_columns', 'mapping': {'old_name': 'new_name'}},
#     ],
# }

# # Example 2: MongoDB
# INGESTION_CONFIG = {
#     'data_source': {
#         'type': 'mongodb',
#         'connection_string': 'mongodb://localhost:27017/',
#         'database': 'mydb',
#         'collection': 'customers',
#         'query': {'status': 'active'},  # MongoDB query filter
#         'limit': 10000,  # Optional: limit number of documents
#     },
#     'target_column': 'churned',
#     'id_column': '_id',
#     'split': {
#         'method': 'random',
#         'test_size': 0.2,
#         'stratify': True,
#         'random_state': 42,
#     },
#     'cleaning': {
#         'drop_duplicates': True,
#         'missing_strategy': 'auto',
#     },
# }

# # Example 3: Time-based split for time series
# INGESTION_CONFIG = {
#     'data_source': {
#         'type': 'csv',
#         'file_path': 'data/sales_data.csv',
#     },
#     'target_column': 'revenue',
#     'split': {
#         'method': 'time',  # Time-based split
#         'time_column': 'date',
#         'test_size': 0.2,  # Most recent 20% as test
#     },
#     'cleaning': {
#         'drop_duplicates': False,
#         'missing_strategy': 'auto',
#     },
# }

AI_TUNING_SUPPORTED_PARAMETERS: Dict[str, Dict[str, Tuple[str, Any, Any]]] = {
    "CatBoost": {
        "iterations": ("int", 200, 2000),
        "learning_rate": ("float", 0.01, 0.3),
        "depth": ("int", 3, 10),
        "l2_leaf_reg": ("float", 1.0, 20.0),
        "subsample": ("float", 0.3, 1.0),
    },
    "XGBoost": {
        "n_estimators": ("int", 100, 2000),
        "learning_rate": ("float", 0.01, 0.3),
        "max_depth": ("int", 3, 12),
        "subsample": ("float", 0.3, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
    },
    "Random Forest": {
        "n_estimators": ("int", 100, 1000),
        "max_depth": ("int", 5, 25),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "LightGBM": {
        "n_estimators": ("int", 100, 2000),
        "learning_rate": ("float", 0.01, 0.3),
        "num_leaves": ("int", 16, 256),
        "feature_fraction": ("float", 0.2, 1.0),
    },
    "Logistic Regression": {
        "C": ("float", 0.0001, 100.0),
    },
    "Linear Regression": {
        "fit_intercept": ("bool", None, None),
    },
}

SUPPORTED_MODEL_SUMMARIES: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "type": "linear",
        "best_for": ["interpretable baseline", "low-dimensional numeric data"],
        "limitations": ["struggles with non-linearity", "sensitive to multicollinearity"],
        "resource_cost": "very_low",
    },
    "Logistic Regression": {
        "type": "linear",
        "best_for": ["binary classification", "interpretable coefficients"],
        "limitations": ["assumes linear decision boundary", "requires feature scaling"],
        "resource_cost": "very_low",
    },
    "Random Forest": {
        "type": "tree_ensemble",
        "best_for": ["mixed feature types", "robustness to noise"],
        "limitations": ["larger memory usage", "less suited for sparse high-dimensional data"],
        "resource_cost": "medium",
    },
    "XGBoost": {
        "type": "gradient_boosting",
        "best_for": ["tabular datasets", "handling missing values"],
        "limitations": [
            "requires tuning to avoid overfitting",
            "can be slow on very wide datasets",
        ],
        "resource_cost": "medium_high",
    },
    "CatBoost": {
        "type": "gradient_boosting",
        "best_for": ["categorical-heavy datasets", "strong default performance"],
        "limitations": ["GPU memory sensitive", "longer training for very large datasets"],
        "resource_cost": "medium_high",
    },
    "LightGBM": {
        "type": "gradient_boosting",
        "best_for": ["large datasets", "sparse features"],
        "limitations": ["sensitive to categorical preprocessing when GPU absent"],
        "resource_cost": "medium",
    },
    "TabNet": {
        "type": "neural_network",
        "best_for": ["datasets with complex interactions", "GPU availability"],
        "limitations": ["requires large sample size", "potentially unstable without tuning"],
        "resource_cost": "high",
    },
}
