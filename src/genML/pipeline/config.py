"""Configuration and shared constants for the genML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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

# Active run (src/genML/run_context.RunContext); its activate() repoints the dirs above
ACTIVE_RUN: Optional[Any] = None


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
        "LightGBM",
        "GLM-LightGBM",
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

# Frontier-model access (src/genML/llm) through the subscription CLIs. ``ask`` tries backends in
# this order. Codex is left out since 2026-11-17 (no longer signed in); its backend stays,
# so GENML_LLM_ORDER="codex-cli,claude-cli" (and second_opinion.ORDER) bring it back. Models:
# GENML_CLAUDE_MODEL / GENML_CODEX_MODEL.
LLM_CONFIG: Dict[str, Any] = {
    "order": ["claude-cli"],
    "claude_model": "opus",              # Claude Code alias for the latest Opus
    "claude_effort": "high",
    "codex_model": None,                 # None: the model in ~/.codex/config.toml
    "codex_effort": "high",
    "timeout_s": 900,
    "max_usd_per_call": 2.0,             # runaway guard for the Claude CLI (--max-budget-usd)
}
