"""
Generic ML Tools for Machine Learning Pipeline using CrewAI

This module contains the core data science functions that power the ML pipeline.
Each function represents a major stage in the machine learning workflow and is designed
to be modular, reusable, and well-documented. The functions handle data loading,
feature engineering, model training, and prediction generation.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
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
    is_xgboost_gpu_available
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
    'n_trials': 50,                          # Number of Optuna trials per model
    'timeout': 600,                          # Max seconds per model tuning (10 min)
    'n_jobs': -1,                            # Parallel jobs (-1 = all cores)
    'show_progress_bar': True,               # Show Optuna progress
    'models_to_tune': ['Random Forest', 'XGBoost', 'Logistic Regression', 'Linear Regression']
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
            'max_features': 100,
            'enable_feature_selection': True,
            'manual_type_hints': {
                'num_lanes': 'numerical',
                'curvature': 'numerical',
                'speed_limit': 'numerical',
                'holiday': 'categorical',
                'num_reported_accidents': 'numerical'
            },
            'interaction_pairs': [
                ('speed_limit', 'curvature'),
                ('num_lanes', 'speed_limit')
            ],
            'numerical_config': {
                'enable_scaling': True,
                'enable_binning': True,
                'enable_polynomial': True,
                'polynomial_degree': 2,
                'n_bins': 5
            },
            'categorical_config': {
                'encoding_method': 'label',
                'enable_frequency': True,
                'max_categories': 20
            },
            'text_config': {
                'enable_basic_features': True,
                'enable_tfidf': False,  # Disabled for performance
                'enable_patterns': True
            },
            'feature_selection': {
                'max_features': 100,
                'enable_statistical_tests': True,
                'enable_model_based': True,
                'selection_strategy': 'union'
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

        # Transform both training and test data
        train_features = feature_engine.transform(train_df)
        test_features = feature_engine.transform(test_df)

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
            "feature_stats": {
                "mean": np.nanmean(train_features_final, axis=0).tolist(),
                "std": np.nanstd(train_features_final, axis=0).tolist()
            },
            "target_distribution": {
                "positive_class": int(np.sum(train_target)),
                "negative_class": int(len(train_target) - np.sum(train_target)),
                "positive_rate": float(np.mean(train_target))
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

        # Get GPU-aware model classes using smart imports
        LinearRegressionClass = get_linear_model_regressor()
        LogisticRegressionClass = get_linear_model_classifier()
        RandomForestRegressorClass = get_random_forest_regressor()
        RandomForestClassifierClass = get_random_forest_classifier()

        # Define model ensemble based on problem type
        # Each model has different strengths and biases
        # Models automatically use GPU (cuML) if available, otherwise CPU (sklearn)
        if problem_type == 'regression':
            models = {
                'Linear Regression': LinearRegressionClass(),                                    # Linear, interpretable
                'Random Forest': RandomForestRegressorClass(n_estimators=100, random_state=42),  # Non-linear, robust (GPU if cuML available)
                'XGBoost': xgb.XGBRegressor(random_state=42, **get_xgboost_params())             # Gradient boosting, high performance (GPU if available)
            }
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = lgb.LGBMRegressor(
                    random_state=42,
                    n_estimators=200,
                    learning_rate=0.05,
                    n_jobs=-1
                )
            # Use regular KFold for regression (no need to preserve class distribution)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            # Use regression scoring metric
            scoring_metric = 'neg_mean_squared_error'
            best_score = float('-inf')  # Higher (less negative) is better for MSE
        else:  # classification
            models = {
                'Logistic Regression': LogisticRegressionClass(random_state=42, max_iter=1000),  # Linear, interpretable
                'Random Forest': RandomForestClassifierClass(n_estimators=100, random_state=42), # Non-linear, robust (GPU if cuML available)
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', **get_xgboost_params())  # Gradient boosting, high performance (GPU if available)
            }
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = lgb.LGBMClassifier(
                    random_state=42,
                    n_estimators=200,
                    learning_rate=0.05,
                    n_jobs=-1,
                    objective='binary'
                )
            # Use StratifiedKFold to preserve class distribution in each fold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Use classification scoring metric
            scoring_metric = 'accuracy'
            best_score = 0  # Higher is better for accuracy

        # Model evaluation pipeline with Optuna hyperparameter tuning
        # Systematically evaluate each model using cross-validation with optimized hyperparameters
        results = {}
        best_model_name = None
        best_model = None
        tuned_params = {}  # Store best hyperparameters for each model

        # Suppress Optuna's default logging to reduce clutter
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        for name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            logger.info(f"[Training] Starting training for {name}")
            logger.info(f"[Training] Training data shape: X={X_train.shape}, y={y_train.shape}")

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
                        'Logistic Regression': 'lr_',
                        'Linear Regression': 'linreg_'
                    }
                    prefix = prefix_map.get(name, '')
                    clean_params = {k.replace(prefix, ''): v for k, v in best_params.items()}

                    # Create tuned model using GPU-aware imports
                    if problem_type == 'regression':
                        if name == 'Random Forest':
                            model = RandomForestRegressorClass(**clean_params)
                        elif name == 'XGBoost':
                            model = xgb.XGBRegressor(**clean_params, **get_xgboost_params())
                        elif name == 'Linear Regression':
                            model = LinearRegressionClass(**clean_params)
                    else:
                        if name == 'Random Forest':
                            model = RandomForestClassifierClass(**clean_params)
                        elif name == 'XGBoost':
                            model = xgb.XGBClassifier(**clean_params, eval_metric='logloss', **get_xgboost_params())
                        elif name == 'Logistic Regression':
                            model = LogisticRegressionClass(**clean_params)

                    # Use best score from optimization
                    best_trial_score = study.best_value
                    trial_scores = [trial.value for trial in study.trials if trial.value is not None]

                    print(f"   ✅ Optimization complete!")
                    print(f"   Best score: {best_trial_score:.6f}")
                    print(f"   Best params: {clean_params}")

                    results[name] = {
                        'mean_score': float(best_trial_score),
                        'std_score': float(np.std(trial_scores[-5:])) if len(trial_scores) >= 5 else 0.0,
                        'individual_scores': trial_scores[-5:],  # Last 5 trial scores
                        'best_params': clean_params,
                        'n_trials': len(study.trials),
                        'tuned': True
                    }

                    if best_trial_score > best_score:
                        best_score = best_trial_score
                        best_model_name = name
                        best_model = model
                else:
                    # Model doesn't have tuning function, use default
                    print(f"   ⚠️  No tuning function available for {name}, using defaults")
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_metric)
                    results[name] = {
                        'mean_score': float(scores.mean()),
                        'std_score': float(scores.std()),
                        'individual_scores': scores.tolist(),
                        'tuned': False
                    }

                    if scores.mean() > best_score:
                        best_score = scores.mean()
                        best_model_name = name
                        best_model = model

            else:
                # Model tuning disabled or not in tuning list - use defaults
                print(f"   Using default hyperparameters...")

                # Suppress XGBoost GPU device mismatch warnings for cleaner output
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
                    # Use n_jobs=1 for XGBoost GPU (GPU parallelizes internally)
                    n_jobs_param = 1 if 'XGBoost' in name and is_xgboost_gpu_available() else -1
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=n_jobs_param)

                results[name] = {
                    'mean_score': float(scores.mean()),
                    'std_score': float(scores.std()),
                    'individual_scores': scores.tolist(),
                    'tuned': False
                }

                print(f"   Score: {scores.mean():.6f} (+/- {scores.std():.6f})")

                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model_name = name
                    best_model = model

            # Clean up memory after each model evaluation
            # Use aggressive cleanup for Random Forest to combat cuML memory leaks
            is_rf = name == 'Random Forest'
            logger.info(f"[Training] {name} - Starting post-evaluation cleanup (aggressive={is_rf})")
            cleanup_memory(f"After {name} evaluation", aggressive=is_rf)
            logger.info(f"[Training] {name} - Post-evaluation cleanup completed")
            logger.info(f"[Training] {name} - Training completed")

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

        # This gives the model access to all available training data
        log_gpu_memory("Before Final Training")
        best_model.fit(X_train, y_train)
        log_gpu_memory("After Final Training")

        # Save the trained model for prediction stage
        # Timestamped filename prevents overwrites and enables model versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'best_model_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
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

        return json.dumps(model_results, indent=2)

    except Exception as e:
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
