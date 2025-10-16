"""Model training pipeline orchestrator."""

from __future__ import annotations

import copy
import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict

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
            if config.TUNING_CONFIG['enabled'] and name in config.TUNING_CONFIG['models_to_tune']:
                print(f"🔧 Tuning hyperparameters for {name}...")
                print(f"   Strategy: Optuna Bayesian Optimization")
                print(f"   Trials: {config.TUNING_CONFIG['n_trials']}")
                logger.info(f"[Training] {name} - Hyperparameter tuning enabled with {config.TUNING_CONFIG['n_trials']} trials")

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
                    logger.info(f"[Training] {name} - Starting Optuna optimization (n_trials={config.TUNING_CONFIG['n_trials']}, timeout={config.TUNING_CONFIG['timeout']}s)")
                    try:
                        study.optimize(
                            objective,
                            n_trials=config.TUNING_CONFIG['n_trials'],
                            timeout=config.TUNING_CONFIG['timeout'],
                            show_progress_bar=config.TUNING_CONFIG['show_progress_bar'],
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
