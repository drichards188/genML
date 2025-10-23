"""Model training pipeline orchestrator."""

from __future__ import annotations

import copy
import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict
import inspect

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', message='.*`gpu_hist` is deprecated.*', module='xgboost')
warnings.filterwarnings('ignore', message='.*gpu_hist.*')
warnings.filterwarnings('ignore', message='Parameters: { \"predictor\" } are not used.')
warnings.filterwarnings('ignore', message='default bootstrap type.*', module='catboost')
warnings.filterwarnings('ignore', message='X does not have valid feature names, but .*LGBM', module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='catboost')

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from src.genML.gpu_utils import (
    get_gpu_config,
    get_linear_model_classifier,
    get_linear_model_regressor,
    get_random_forest_classifier,
    get_random_forest_regressor,
    get_xgboost_params,
    is_catboost_gpu_available,
    is_cuml_available,
    is_xgboost_gpu_available,
    log_gpu_memory,
    is_lightgbm_gpu_available,
    get_lightgbm_params,
)
from src.genML.pipeline import config
from src.genML.pipeline.ai_tuning import (
    build_ai_tuning_override_details,
    extract_override_values,
    load_ai_tuning_overrides,
    save_ai_tuning_overrides,
)
from src.genML.pipeline.model_advisor import (
    apply_model_selection_guidance,
    detect_problem_type,
    load_model_selection_guidance
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
from src.genML.pipeline.tuning import (
    optimize_catboost,
    optimize_linear_model,
    optimize_random_forest,
    optimize_tabnet,
    optimize_xgboost,
)
from src.genML.pipeline.utils import cleanup_memory


logger = logging.getLogger(__name__)


def _compute_adaptive_tuning_limits(
    shape: tuple[int, ...],
) -> dict[str, Any]:
    """
    Derive dataset-aware overrides for Optuna tuning so small tabular datasets
    do not spend excessive time in hyperparameter optimization loops.
    """
    base_trials = config.TUNING_CONFIG["n_trials"]
    base_timeout = config.TUNING_CONFIG["timeout"]

    n_samples = int(shape[0]) if shape else 0
    n_features = int(shape[1]) if len(shape) > 1 else 0

    overrides: dict[str, Any] = {
        "global": {
            "n_trials": base_trials,
            "timeout": base_timeout,
        },
        "per_model": {},
        "notes": [],
    }

    if n_samples == 0:
        return overrides

    # Heuristic caps for compact datasets where long searches bring diminishing returns.
    very_small_dataset = n_samples <= 2000
    small_dataset = 2000 < n_samples <= 5000 or (n_samples <= 5000 and n_features <= 50)

    if very_small_dataset:
        overrides["global"]["n_trials"] = min(base_trials, 25)
        overrides["global"]["timeout"] = min(base_timeout, 240)
        overrides["per_model"]["CatBoost"] = {
            "n_trials": min(overrides["global"]["n_trials"], 15),
            "timeout": min(overrides["global"]["timeout"], 180),
        }
        overrides["per_model"]["TabNet"] = {
            "n_trials": min(overrides["global"]["n_trials"], 10),
            "timeout": min(overrides["global"]["timeout"], 180),
        }
        overrides["notes"].append(
            f"Detected small dataset (n={n_samples}, features={n_features}); capping tuning to "
            f"{overrides['global']['n_trials']} trials with CatBoost limited to "
            f"{overrides['per_model']['CatBoost']['n_trials']} trials."
        )
    elif small_dataset:
        overrides["global"]["n_trials"] = min(base_trials, 40)
        overrides["global"]["timeout"] = min(base_timeout, 360)
        overrides["per_model"]["CatBoost"] = {
            "n_trials": min(overrides["global"]["n_trials"], 20),
            "timeout": min(overrides["global"]["timeout"], 240),
        }
        overrides["notes"].append(
            f"Detected mid-sized dataset (n={n_samples}, features={n_features}); reducing tuning to "
            f"{overrides['global']['n_trials']} trials overall and CatBoost to "
            f"{overrides['per_model']['CatBoost']['n_trials']} trials."
        )

    return overrides


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
        X_train = np.load(config.FEATURES_DIR / 'X_train.npy')
        y_train = np.load(config.FEATURES_DIR / 'y_train.npy')
        logger.info(f"Loaded training data: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        logger.info(f"Data types: X_train.dtype={X_train.dtype}, y_train.dtype={y_train.dtype}")

        n_samples = int(X_train.shape[0])
        n_features = int(X_train.shape[1]) if X_train.ndim > 1 else 0
        lightgbm_n_jobs_default = -1 if n_samples > 1500 else config.MEMORY_CONFIG["max_parallel_jobs"]
        lightgbm_gpu_ready = (
            LIGHTGBM_AVAILABLE
            and is_lightgbm_gpu_available()
            and not config.GPU_CONFIG.get('force_cpu', False)
        )

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

        original_models_to_tune = list(config.TUNING_CONFIG['models_to_tune'])

        # Get GPU-aware model classes using smart imports
        LinearRegressionClass = get_linear_model_regressor()
        LogisticRegressionClass = get_linear_model_classifier()
        RandomForestRegressorClass = get_random_forest_regressor()
        RandomForestClassifierClass = get_random_forest_classifier()

        tuning_overrides = _compute_adaptive_tuning_limits(X_train.shape)
        global_tuning_trials = tuning_overrides["global"]["n_trials"]
        global_tuning_timeout = tuning_overrides["global"]["timeout"]
        per_model_tuning_limits: Dict[str, Dict[str, Any]] = tuning_overrides["per_model"]

        for note in tuning_overrides.get("notes", []):
            print(f"⚙️ {note}")

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
                params = {'random_state': 42, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
                params.update(get_xgboost_params())
                if overrides:
                    params.update(overrides)
                return xgb.XGBRegressor(**params)

            return factory

        def make_xgb_classifier_factory():
            def factory(overrides=None):
                params = {'random_state': 42, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
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
                is_regressor = constructor == cb.CatBoostRegressor
                if is_regressor:
                    params.update({
                        'loss_function': 'RMSE',
                        'eval_metric': 'RMSE'
                    })
                else:
                    params.update({
                        'loss_function': 'Logloss',
                        'eval_metric': 'AUC'
                    })

                if is_catboost_gpu_available() and not config.GPU_CONFIG['force_cpu']:
                    params.update({'task_type': 'GPU', 'devices': '0'})
                    default_bootstrap = 'Poisson'
                else:
                    params.update({'task_type': 'CPU', 'thread_count': config.MEMORY_CONFIG['max_parallel_jobs']})
                    default_bootstrap = 'Bernoulli'

                if overrides:
                    params.update(overrides)

                subsample = params.get('subsample')
                bootstrap_type = params.get('bootstrap_type')
                if subsample is not None and subsample < 0.999:
                    if not bootstrap_type or str(bootstrap_type).lower() == 'bayesian':
                        params['bootstrap_type'] = default_bootstrap
                elif not bootstrap_type:
                    params['bootstrap_type'] = default_bootstrap

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
                lightgbm_base_params = {
                    'random_state': 42,
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'n_jobs': lightgbm_n_jobs_default,
                    'verbosity': -1,
                }
                if n_samples <= 1500:
                    lightgbm_base_params.update(
                        {
                            'n_estimators': 120,
                            'min_child_samples': 5,
                            'min_gain_to_split': 0.0,
                            'feature_pre_filter': False,
                        }
                    )
                    print(
                        f"ℹ️ Adjusting LightGBM for small dataset (n={n_samples}, features={n_features}) "
                        "to avoid degenerate splits and excessive CPU usage."
                    )
                if not lightgbm_gpu_ready:
                    print("ℹ️ LightGBM GPU build not detected; training will run on CPU.")
                models['LightGBM'] = make_factory(lgb.LGBMRegressor, lightgbm_base_params)
                if lightgbm_gpu_ready:
                    lgb_gpu_params = get_lightgbm_params()

                    def lgb_factory_with_gpu(overrides=None):
                        params = lightgbm_base_params.copy()
                        params.update(lgb_gpu_params)
                        if overrides:
                            params.update(overrides)
                        return lgb.LGBMRegressor(**params)

                    models['LightGBM'] = lgb_factory_with_gpu
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
            logistic_base_params: Dict[str, Any] = {'max_iter': 1000}
            try:
                logistic_signature = inspect.signature(LogisticRegressionClass.__init__)
                if 'random_state' in logistic_signature.parameters:
                    logistic_base_params['random_state'] = 42
            except (ValueError, TypeError):
                # Some constructors (e.g., C extensions) may not expose signatures; fall back to safe defaults
                pass

            models = {
                'Logistic Regression': make_factory(
                    LogisticRegressionClass,
                    logistic_base_params
                ),  # Linear, interpretable
                'Random Forest': make_factory(
                    RandomForestClassifierClass,
                    {'n_estimators': 100, 'random_state': 42}
                ),  # Non-linear, robust (GPU if cuML available)
                'XGBoost': make_xgb_classifier_factory()  # Gradient boosting, high performance (GPU if available)
            }
            if LIGHTGBM_AVAILABLE:
                lightgbm_base_params = {
                    'random_state': 42,
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'n_jobs': lightgbm_n_jobs_default,
                    'objective': 'binary',
                    'verbosity': -1,
                }
                if n_samples <= 1500:
                    lightgbm_base_params.update(
                        {
                            'n_estimators': 120,
                            'min_child_samples': 5,
                            'min_gain_to_split': 0.0,
                            'feature_pre_filter': False,
                        }
                    )
                    print(
                        f"ℹ️ Adjusting LightGBM for small dataset (n={n_samples}, features={n_features}) "
                        "to avoid degenerate splits and excessive CPU usage."
                    )
                if not lightgbm_gpu_ready:
                    print("ℹ️ LightGBM GPU build not detected; training will run on CPU.")
                models['LightGBM'] = make_factory(lgb.LGBMClassifier, lightgbm_base_params)
                if lightgbm_gpu_ready:
                    lgb_gpu_params = get_lightgbm_params()

                    def lgb_clf_factory_with_gpu(overrides=None):
                        params = lightgbm_base_params.copy()
                        params.update(lgb_gpu_params)
                        if overrides:
                            params.update(overrides)
                        return lgb.LGBMClassifier(**params)

                    models['LightGBM'] = lgb_clf_factory_with_gpu
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
            config.TUNING_CONFIG['models_to_tune'] = tuned_models_list
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

        # Track overall model training progress
        total_models = len(models)
        models_completed = 0

        for name, model_factory in models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            logger.info(f"[Training] Starting training for {name}")
            logger.info(f"[Training] Training data shape: X={X_train.shape}, y={y_train.shape}")

            # Track model start in progress tracker
            if config.PROGRESS_TRACKER:
                config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

            model_override_values = ai_model_overrides.get(name, {})
            used_params = None
            model_mean_score = None
            model_std_score = 0.0
            model_scores_list = []
            tuned_flag = False

            # Check if this model should be tuned
            if config.TUNING_CONFIG['enabled'] and name in config.TUNING_CONFIG['models_to_tune']:
                print(f"🔧 Tuning hyperparameters for {name}...")
                print(f"   Strategy: Optuna Bayesian Optimization")
                model_limit = per_model_tuning_limits.get(name, {})
                model_n_trials = int(model_limit.get("n_trials", global_tuning_trials))
                model_timeout = int(model_limit.get("timeout", global_tuning_timeout))
                print(f"   Trials: {model_n_trials}")
                print(f"   Timeout: {model_timeout}s")
                if model_n_trials < config.TUNING_CONFIG['n_trials'] or model_timeout < config.TUNING_CONFIG['timeout']:
                    print("   (Adaptive tuning limits applied for dataset size)")
                logger.info(f"[Training] {name} - Hyperparameter tuning enabled with {model_n_trials} trials")

                # Track model start with tuning info
                if config.PROGRESS_TRACKER:
                    config.PROGRESS_TRACKER.track_model_start(name, total_trials=model_n_trials)

                # Create Optuna study for hyperparameter optimization
                study = optuna.create_study(
                    direction='maximize',  # Maximize score (works for both neg_mse and accuracy)
                    sampler=TPESampler(seed=42)
                )
                logger.info(f"[Training] {name} - Optuna study created")

                # Select appropriate optimization function
                if name == 'Random Forest':
                    logger.info(f"[Training] Random Forest - Entering hyperparameter optimization")
                    logger.info(f"[Training] Random Forest - Memory config: max_trees={config.MEMORY_CONFIG['max_trees_random_forest']}, cv_folds={config.MEMORY_CONFIG['rf_cv_folds']}, aggressive_cleanup={config.MEMORY_CONFIG['aggressive_rf_cleanup']}")
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
                    logger.info(f"[Training] {name} - Starting Optuna optimization (n_trials={model_n_trials}, timeout={model_timeout}s)")

                    # Define callback to track trial progress
                    def trial_callback(study, trial):
                        if config.PROGRESS_TRACKER and trial.value is not None:
                            config.PROGRESS_TRACKER.track_model_trial(name, trial.number + 1, trial.value)

                    try:
                        study.optimize(
                            objective,
                            n_trials=model_n_trials,
                            timeout=model_timeout,
                            show_progress_bar=config.TUNING_CONFIG['show_progress_bar'],
                            n_jobs=1,  # Serial execution for stability
                            callbacks=[trial_callback]  # Track trial progress
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

                    # Track model completion
                    if config.PROGRESS_TRACKER:
                        config.PROGRESS_TRACKER.track_model_complete(
                            name,
                            mean_score=model_mean_score,
                            std_score=model_std_score,
                            best_params=clean_params
                        )

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

                    # Track model completion
                    if config.PROGRESS_TRACKER:
                        config.PROGRESS_TRACKER.track_model_complete(
                            name,
                            mean_score=model_mean_score,
                            std_score=model_std_score
                        )

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

                # Track model start (no tuning)
                if config.PROGRESS_TRACKER:
                    config.PROGRESS_TRACKER.track_model_start(name, total_trials=None)

                scores = evaluate_with_default_params(name, model_factory, model_override_values)
                model_mean_score = float(scores.mean())
                model_std_score = float(scores.std())
                model_scores_list = scores.tolist()

                # Track model completion
                if config.PROGRESS_TRACKER:
                    config.PROGRESS_TRACKER.track_model_complete(
                        name,
                        mean_score=model_mean_score,
                        std_score=model_std_score
                    )

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

            # Update model training progress
            models_completed += 1
            if config.PROGRESS_TRACKER:
                config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

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
                    meta_params: Dict[str, Any] = {'max_iter': 1000}
                    try:
                        logistic_sig = inspect.signature(LogisticClass.__init__)
                        if 'random_state' in logistic_sig.parameters:
                            meta_params['random_state'] = 42
                    except (ValueError, TypeError):
                        pass
                    meta_learner = LogisticClass(**meta_params)
                    ensemble = StackingClassifier(
                        estimators=ensemble_estimators,
                        final_estimator=meta_learner,
                        cv=5,  # Internal CV for generating meta-features
                        n_jobs=1  # Serial for GPU stability
                    )
                    print(f"Meta-learner: Logistic Regression")

                # Evaluate stacking ensemble with cross-validation
                print("Evaluating stacking ensemble...")
                try:
                    ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=1)
                    ensemble_mean_score = ensemble_scores.mean()
                except Exception as exc:
                    logger.warning(f"[Ensemble] Stacking evaluation failed: {exc}")
                    print(f"⚠️  Stacking ensemble evaluation failed: {exc}")
                    cleanup_memory("After failed ensemble evaluation", aggressive=False)
                else:
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
            print(f"\u2139\ufe0f  Using GPU acceleration (cuML) for {best_model_name}")
        elif is_xgboost_gpu_available() and 'XGBoost' in best_model_name:
            print(f"\u2139\ufe0f  Using GPU acceleration (XGBoost) for {best_model_name}")
        elif is_lightgbm_gpu_available() and 'LightGBM' in best_model_name:
            print(f"\u2139\ufe0f  Using GPU acceleration (LightGBM) for {best_model_name}")
        else:
            print(f"\u2139\ufe0f  Using CPU for {best_model_name}")

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
        if config.AI_ADVISORS_CONFIG['enabled'] and config.AI_ADVISORS_CONFIG['error_analysis']['enabled']:
            try:
                print(f"\n{'='*60}")
                print("🤖 AI Error Pattern Analyzer")
                print(f"{'='*60}")
                print("Analyzing prediction errors to identify patterns...")

                from src.genML.ai_advisors import ErrorPatternAnalyzer, OpenAIClient

                # Create OpenAI client with configuration
                openai_client = OpenAIClient(config=config.AI_ADVISORS_CONFIG['openai_config'])

                if openai_client.is_available():
                    # Generate predictions on training data for error analysis
                    y_pred_train = best_model.predict(X_train)

                    # Load feature names for detailed analysis
                    feature_names_file = config.FEATURES_DIR / 'feature_names.txt'
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
                        top_n_errors=config.AI_ADVISORS_CONFIG['error_analysis']['top_n_errors']
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
                        if config.AI_ADVISORS_CONFIG['error_analysis']['save_report']:
                            report_path = config.REPORTS_DIR / 'ai_error_analysis.json'
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

        joblib.dump(best_model, config.MODELS_DIR / model_filename)

        # Save detailed cross-validation results for analysis
        # Provides transparency into model selection process
        cv_results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Mean_Score': [results[name]['mean_score'] for name in results.keys()],
            'Std_Score': [results[name]['std_score'] for name in results.keys()]
        })
        cv_results_df.to_csv(config.MODELS_DIR / 'cross_validation_results.csv', index=False)

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
            "model_saved": str(config.MODELS_DIR / model_filename),
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
        with open(config.REPORTS_DIR / 'model_training_report.json', 'w') as f:
            json.dump(model_results, f, indent=2)

        config.TUNING_CONFIG['models_to_tune'] = original_models_to_tune
        return json.dumps(model_results, indent=2)

    except Exception as e:
        try:
            config.TUNING_CONFIG['models_to_tune'] = original_models_to_tune
        except Exception:
            pass
        return json.dumps({
            "error": f"Error in model training: {str(e)}",
            "status": "failed"
        })
