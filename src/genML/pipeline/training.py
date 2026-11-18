"""Model training pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Callable, Optional, Tuple, List

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', message='.*`gpu_hist` is deprecated.*', module='xgboost')
warnings.filterwarnings('ignore', message='.*gpu_hist.*')
warnings.filterwarnings('ignore', message='Parameters: { \"predictor\" } are not used.')
warnings.filterwarnings('ignore', message='default bootstrap type.*', module='catboost')
warnings.filterwarnings('ignore', message='X does not have valid feature names, but .*LGBM', module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='catboost')
warnings.filterwarnings('ignore', message='.*Default metric period is 5 because AUC.*')
warnings.filterwarnings('ignore', message='.*AUC is/are not implemented for GPU.*')

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from src.genML.gpu_utils import (
    get_gpu_config,
    get_linear_model_classifier,
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
from src.genML.pipeline import config, context
from src.genML.pipeline.ensemble import hill_climb, stack
from src.genML.pipeline.inference import infer_task_type
from src.genML.pipeline.oof import (ExperimentStore, FoldFeatures, _with_seed, fit_oof, predict_matrix, run_folds,
                                    tuning_fold_indices)
from src.genML.pipeline.targets import inverse_regression
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
    GBDT_KINDS,
    TREE_COUNT_PARAM,
    fold_cv_score,
    model_search_space,
    optimize_gbdt,
    optimize_linear_model,
    optimize_random_forest,
    optimize_tabnet,
    suggest_gbdt_params,
)
from src.genML.pipeline.utils import cleanup_memory
from src.genML.pipeline.neural import NEURAL_MODELS


logger = logging.getLogger(__name__)


def run_seed(budget: Optional[Dict[str, Any]]) -> int:
    """First seed of a run: 42 plus ``budget.seed`` (0 by default). Only trusted code sets budget.seed
    (repeated baseline runs for the noise measurement); experiments cannot."""
    return 42 + int((budget or {}).get("seed", 0))


def detect_problem_type() -> str:
    """'regression' or 'classification', from the stored training target (y_train.npy)."""
    try:
        y_train = np.load(config.FEATURES_DIR / "y_train.npy")
        task, reason = infer_task_type(y_train)
        problem_type = "regression" if task == "regression" else "classification"
        logger.info("Problem type detected: %s (%s; %s)", problem_type, task, reason)
        return problem_type
    except Exception as exc:
        print(f"Error in problem type detection: {exc}")
        print("Defaulting to classification")
        return "classification"


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


def _categorize_models_by_resource(
    models: Dict[str, Callable],
    problem_type: str
) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
    """
    Categorize models into CPU-only and GPU-capable pools.

    GPU models are kept sequential to avoid OOM errors.
    CPU models can be trained in parallel safely.

    Returns:
        Tuple of (cpu_models, gpu_models) dictionaries
    """
    cpu_models = {}
    gpu_models = {}

    # Models that use GPU acceleration
    gpu_model_names = {'XGBoost', 'CatBoost', 'LightGBM', 'TabNet', *NEURAL_MODELS}

    for name, factory in models.items():
        if name in gpu_model_names:
            # Check if GPU is actually available for this model
            if name == 'XGBoost' and is_xgboost_gpu_available():
                gpu_models[name] = factory
            elif name == 'CatBoost' and is_catboost_gpu_available():
                gpu_models[name] = factory
            elif name == 'LightGBM' and is_lightgbm_gpu_available():
                gpu_models[name] = factory
            elif name in NEURAL_MODELS:
                gpu_models[name] = factory  # one at a time: they share the GPU (or all CPU cores)
            elif name == 'TabNet' and TABNET_AVAILABLE:
                # TabNet can use GPU if torch has CUDA
                try:
                    if torch.cuda.is_available():
                        gpu_models[name] = factory
                    else:
                        cpu_models[name] = factory
                except:
                    cpu_models[name] = factory
            else:
                # GPU not available, treat as CPU model
                cpu_models[name] = factory
        else:
            cpu_models[name] = factory

    return cpu_models, gpu_models


def _build_cv(problem, problem_type: str):
    """Cross-validation splitter from the resolved problem (fixed seed per problem)."""
    cv_config = dict(problem.cv) if problem is not None else {}
    n_splits = int(cv_config.get("n_splits", 5))
    random_state = int(cv_config.get("random_state", 42))
    shuffle = bool(cv_config.get("shuffle", True))
    method = cv_config.get("method") or ("kfold" if problem_type == "regression" else "stratified_kfold")
    if method == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if method == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if method in ("group_kfold", "stratified_group_kfold", "time_blocks"):
        return None  # replaced by the run's own folds once they are loaded (PredefinedSplit)
    raise ValueError(f"Unsupported cv.method '{method}'")


def _filter_models(models: Dict[str, Any], requested) -> Dict[str, Any]:
    """Keep only the models a problem spec asks for, in the spec's order."""
    if not requested:
        return models
    lookup = {name.lower(): name for name in models}
    selected = OrderedDict()
    for name in requested:
        key = lookup.get(str(name).lower())
        if key is None:
            logger.warning("Model '%s' from the problem spec is not available; skipping", name)
            continue
        selected[key] = models[key]
    if not selected:
        raise ValueError(f"None of the requested models {requested} are available: {list(models)}")
    return selected


def _run_oof_stage(problem, task: str, results: Dict[str, Any], models: Dict[str, Callable],
                   params_for: Callable[[str, Dict[str, Any]], Any],
                   X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """Retrain candidates on fixed folds, store OOF/test predictions, choose final predictions.

    Every model is scored on its out-of-fold predictions with the problem's own
    metric (on the original target scale for regression). The run's final test
    predictions come from the best single model or from the ensemble of this
    run's models, whichever has the better OOF score.
    """
    metric = problem.metric_obj
    X_test = np.load(config.FEATURES_DIR / 'X_test.npy')
    transform_info = joblib.load(config.FEATURES_DIR / 'target_transform.pkl')
    n_classes = len(problem.classes) if problem.is_classification else 0

    if task == 'regression':
        y_eval = inverse_regression(y_train.astype(float), transform_info)

        def native_score(pred):
            return metric.score(y_eval, inverse_regression(pred, transform_info), task)
    else:
        def native_score(pred):
            return metric.score(y_train, pred, task)

    run = config.ACTIVE_RUN
    problem_dir = run.problem_dir if run is not None else config.OUTPUTS_DIR
    run_id = run.run_id if run is not None else datetime.now().strftime('%Y%m%d_%H%M%S')
    fold_ids, signature = run_folds(y_train, problem.cv, problem_dir, config.FEATURES_DIR)
    store = ExperimentStore(problem_dir)
    store.save_targets(signature, y_train, transform_info)
    first_seed = run_seed(problem.budget)
    seeds = list(range(first_seed, first_seed + int(problem.budget.get('seeds', 1))))
    tracker = config.PROGRESS_TRACKER
    fold_features = FoldFeatures.load(config.FEATURES_DIR)

    candidates = sorted(
        [(name, res) for name, res in results.items() if name in models and res.get('mean_score') is not None],
        key=lambda item: item[1]['mean_score'], reverse=True,
    )
    limit = problem.budget.get('oof_models')
    if limit:
        candidates = candidates[:int(limit)]

    print(f"\n{'='*60}")
    print(f"Out-of-fold stage: {len(candidates)} model(s) x {int(fold_ids.max()) + 1} folds x {len(seeds)} seed(s)")
    print(f"Folds {signature} shared by every run of '{problem.name}'")
    print(f"{'='*60}")

    singles = []
    for name, res in candidates:
        params = params_for(name, res)
        if tracker:
            tracker.track_model_start(f"{name} (OOF)", total_trials=None)
        try:
            oof, test_pred = fit_oof(models[name], params, X_train, y_train, X_test, fold_ids,
                                     task, n_classes, seeds, fold_features=fold_features)
        except Exception as exc:
            logger.warning(f"[OOF] {name} failed: {exc}")
            print(f"⚠️  OOF for {name} failed: {exc}")
            continue
        score = native_score(oof)
        experiment = store.save(name, run_id, oof, test_pred, {
            "model": name, "params": params, "metric": problem.metric, "oof_score": score,
            "greater_is_better": metric.greater_is_better, "fold_signature": signature,
            "n_features": int(X_train.shape[1]), "seeds": seeds, "task": task,
            "nested_encoding": fold_features is not None,
        })
        results[name]['oof_score'] = score
        results[name]['experiment_id'] = experiment.experiment_id
        singles.append({"name": name, "score": score, "oof": oof, "test": test_pred,
                        "experiment_id": experiment.experiment_id})
        print(f"   {name}: OOF {problem.metric} = {score:.6f}")
        if tracker:
            tracker.track_model_complete(f"{name} (OOF)", mean_score=float(metric.signed(score)),
                                         std_score=0.0, best_params=params)
        cleanup_memory(f"After OOF {name}", aggressive=False)

    if not singles:
        raise RuntimeError("No model produced out-of-fold predictions")

    best_single = max(singles, key=lambda item: metric.signed(item["score"]))
    final = {"name": best_single["name"], "score": best_single["score"], "test": best_single["test"],
             "source": "single", "experiment_id": best_single["experiment_id"]}
    ensemble_info = None

    method = problem.ensemble.get('method', 'stack')
    if method not in ('stack', 'hill_climb', 'none'):
        raise ValueError(f"Unknown ensemble.method '{method}' (use stack, hill_climb or none)")
    if method != 'none' and len(singles) >= 2:
        members = sorted(singles, key=lambda item: metric.signed(item["score"]), reverse=True)
        members = members[:int(problem.ensemble.get('top', 10))]
        oofs, tests = [m["oof"] for m in members], [m["test"] for m in members]
        if tracker:
            tracker.track_model_start('Stacking Ensemble', total_trials=None)
        if method == 'stack':
            blend_oof, blend_test, ensemble_info = stack(
                y_train, oofs, tests, fold_ids, task, n_classes,
                strength=float(problem.ensemble.get('strength', 1.0)))
        else:
            blend_oof, blend_test, ensemble_info = hill_climb(
                oofs, tests, lambda pred: metric.signed(native_score(pred)))
        ensemble_score = native_score(blend_oof)
        ensemble_info.update({"members": [m["name"] for m in members], "oof_score": ensemble_score})
        ensemble_name = f"Ensemble ({method})"
        experiment = store.save(f"ensemble_{method}", run_id, blend_oof, blend_test, {
            "model": ensemble_name, "metric": problem.metric, "oof_score": ensemble_score,
            "greater_is_better": metric.greater_is_better, "fold_signature": signature,
            "members": [m["experiment_id"] for m in members], "ensemble": ensemble_info, "task": task,
        })
        results[ensemble_name] = {
            'mean_score': float(metric.signed(ensemble_score)), 'std_score': 0.0,
            'oof_score': ensemble_score, 'members': ensemble_info["members"],
            'experiment_id': experiment.experiment_id, 'tuned': False,
        }
        print(f"   {ensemble_name} of {len(members)}: OOF {problem.metric} = {ensemble_score:.6f}")
        if tracker:
            tracker.track_model_complete('Stacking Ensemble', mean_score=float(metric.signed(ensemble_score)),
                                         std_score=0.0, best_params={'members': ensemble_info["members"]})
        if metric.signed(ensemble_score) > metric.signed(best_single["score"]):
            final = {"name": ensemble_name, "score": ensemble_score, "test": blend_test,
                     "source": "ensemble", "experiment_id": experiment.experiment_id}

    np.save(config.FEATURES_DIR / 'final_test_predictions.npy', final["test"])
    summary = {
        "final_name": final["name"],
        "final_score": final["score"],
        "final_source": final["source"],
        "final_experiment_id": final["experiment_id"],
        "metric": problem.metric,
        "greater_is_better": metric.greater_is_better,
        "fold_signature": signature,
        "seeds": seeds,
        "single_scores": {m["name"]: m["score"] for m in singles},
        "ensemble": ensemble_info,
    }
    if (problem.cv or {}).get("method") == "time_blocks":
        # The last block is predicted only by models trained on earlier periods: an honest
        # forward-in-time score to read next to the k-fold OOF score
        last = int(fold_ids.max())
        rows = np.where(fold_ids == last)[0]
        oof_final = store.get(final["experiment_id"]).oof
        if task == 'regression':
            holdout = metric.score(y_eval[rows], inverse_regression(oof_final[rows], transform_info), task)
        else:
            holdout = metric.score(y_train[rows], oof_final[rows], task)
        summary["forward_holdout"] = {"fold": last, "rows": int(len(rows)), "score": float(holdout)}
        print(f"   Forward holdout (latest block, {len(rows)} rows): {problem.metric} = {holdout:.6f}")
    with open(config.REPORTS_DIR / 'final_predictions.json', 'w') as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"🏆 Final predictions: {final['name']} (OOF {problem.metric} = {final['score']:.6f})")
    return summary


def _train_single_model(
    name: str,
    model_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    problem_type: str,
    cv: Any,
    scoring_metric: str,
    tuning_enabled: bool,
    models_to_tune: List[str],
    per_model_tuning_limits: Dict[str, Dict[str, Any]],
    global_tuning_trials: int,
    global_tuning_timeout: int,
    base_optuna_n_trials: int,
    base_optuna_timeout: int,
    is_cpu_pool: bool,
    fold_ids: np.ndarray = None,
    fold_features: FoldFeatures = None,
    tuning_folds: List[int] = None,
    gbdt_budget: Dict[str, Any] = None,
    seed: int = 42,
) -> Tuple[str, Dict[str, Any]]:
    """
    Train a single model with hyperparameter tuning.

    ``seed`` (42 plus ``budget.seed``) drives the Optuna sampler and the models built while tuning, so
    repeated runs of the same settings with different seeds measure run-to-run noise.

    This function is designed to be called in parallel for CPU models
    or sequentially for GPU models.

    Returns:
        Tuple of (model_name, results_dict)
    """
    if seed != 42:
        base_factory = model_factory

        def model_factory(overrides=None):
            return _with_seed(base_factory, overrides, seed)

    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    logger.info(f"[Training] Starting training for {name}")
    logger.info(f"[Training] Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Track model start in progress tracker
    if config.PROGRESS_TRACKER:
        config.PROGRESS_TRACKER.track_model_start(name)

    used_params = None
    model_mean_score = None
    model_std_score = 0.0
    model_scores_list = []
    tuned_flag = False

    # Helper function for default evaluation
    def evaluate_with_default_params(overrides=None):
        import warnings
        if fold_ids is not None:
            # Shared folds with each fold's nested encodings (no cross-fold leakage)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
                # Neural and foundation models are screened on the tuning folds only: the OOF
                # stage refits every fold anyway, and each fit costs minutes
                screen_folds = (tuning_folds if name in NEURAL_MODELS and tuning_folds
                                else range(int(fold_ids.max()) + 1))
                _, fold_scores, _ = fold_cv_score(
                    model_factory, overrides, X_train, y_train, fold_ids,
                    screen_folds, scoring_metric, fold_features)
            return np.asarray(fold_scores)
        estimator = model_factory(overrides)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
                # For CPU pool models, we can use parallel CV
                # For GPU models, use n_jobs=1 (GPU parallelizes internally)
                n_jobs_param = -1 if is_cpu_pool else 1
                return cross_val_score(
                    estimator,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring_metric,
                    n_jobs=n_jobs_param
                )
        finally:
            del estimator

    # Check if this model should be tuned
    if tuning_enabled and name in models_to_tune:
        print(f"🔧 Tuning hyperparameters for {name}...")
        print(f"   Strategy: Optuna Bayesian Optimization")
        model_limit = per_model_tuning_limits.get(name, {})
        model_n_trials = int(model_limit.get("n_trials", global_tuning_trials))
        model_timeout = int(model_limit.get("timeout", global_tuning_timeout))
        print(f"   Trials: {model_n_trials}")
        print(f"   Timeout: {model_timeout}s")
        if model_n_trials < base_optuna_n_trials or model_timeout < base_optuna_timeout:
            print("   (Adaptive tuning limits applied for dataset size)")
        logger.info(f"[Training] {name} - Hyperparameter tuning enabled with {model_n_trials} trials")

        # Track model start with tuning info
        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_model_start(name, total_trials=model_n_trials)

        # Create Optuna study for hyperparameter optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=seed)
        )
        logger.info(f"[Training] {name} - Optuna study created")

        # Select appropriate optimization function
        if name == 'Random Forest':
            logger.info(f"[Training] Random Forest - Entering hyperparameter optimization")
            objective = lambda trial: optimize_random_forest(trial, X_train, y_train, problem_type, cv, scoring=scoring_metric)
        elif name in GBDT_KINDS and fold_ids is not None:
            gbdt_kind = GBDT_KINDS[name]
            print(f"   Tuning folds: {list(tuning_folds)} (early stopping picks the tree count)")
            gbdt_search_space = model_search_space(name, gbdt_budget)
            gbdt_max_trees = (gbdt_budget or {}).get('max_trees')
            if (gbdt_budget or {}).get('search_space', {}).get(name) or gbdt_max_trees:
                print(f"   Search space from budget: {(gbdt_budget or {}).get('search_space', {}).get(name)}"
                      f", max trees {gbdt_max_trees or 'default'}")
            objective = lambda trial: optimize_gbdt(trial, gbdt_kind, model_factory, X_train, y_train,
                                                    fold_ids, tuning_folds, scoring_metric, fold_features,
                                                    space=gbdt_search_space, max_trees=gbdt_max_trees)
        elif name == 'TabNet':
            objective = lambda trial: optimize_tabnet(trial, X_train, y_train, problem_type, cv, scoring=scoring_metric)
        elif name == 'Logistic Regression' or name == 'Linear Regression':
            objective = lambda trial: optimize_linear_model(trial, X_train, y_train, problem_type, cv, scoring=scoring_metric)
        else:
            objective = None

        if objective:
            # Run hyperparameter optimization
            logger.info(f"[Training] {name} - Starting Optuna optimization (n_trials={model_n_trials}, timeout={model_timeout}s)")

            # Define callback to track trial progress
            def trial_callback(study, trial):
                if config.PROGRESS_TRACKER and trial.value is not None:
                    config.PROGRESS_TRACKER.track_model_trial(name, trial.number + 1, trial.value)

            # For CPU pool models, allow parallel trials based on config
            # For GPU models, keep serial to avoid memory issues
            cpu_optuna_n_jobs = config.PARALLEL_TRAINING_CONFIG.get('cpu_optuna_n_jobs', 2)
            optuna_n_jobs = cpu_optuna_n_jobs if is_cpu_pool else 1

            try:
                study.optimize(
                    objective,
                    n_trials=model_n_trials,
                    timeout=model_timeout,
                    show_progress_bar=config.TUNING_CONFIG['show_progress_bar'],
                    n_jobs=optuna_n_jobs,
                    callbacks=[trial_callback]
                )
                logger.info(f"[Training] {name} - Optimization completed successfully")
            except Exception as e:
                logger.error(f"[Training] {name} - Optimization FAILED: {str(e)}")
                raise

            # Clean up memory after optimization
            is_rf = name == 'Random Forest'
            logger.info(f"[Training] {name} - Starting memory cleanup (aggressive={is_rf})")
            cleanup_memory(f"After {name} optimization", aggressive=is_rf)
            logger.info(f"[Training] {name} - Memory cleanup completed")

            # Extract best parameters
            best_params = study.best_params

            # Remove model-specific prefixes from parameter names
            prefix_map = {
                'Random Forest': 'rf_',
                'XGBoost': 'xgb_',
                'LightGBM': 'lgb_',
                'CatBoost': 'cb_',
                'TabNet': 'tabnet_',
                'Logistic Regression': 'lr_',
                'Linear Regression': 'linreg_'
            }
            prefix = prefix_map.get(name, '')
            if name in GBDT_KINDS and fold_ids is not None:
                # Full parameter set (including fixed entries) plus the early-stopped tree count
                gbdt_kind = GBDT_KINDS[name]
                clean_params = suggest_gbdt_params(optuna.trial.FixedTrial(best_params), gbdt_kind,
                                                   model_search_space(name, gbdt_budget))
                n_trees = study.best_trial.user_attrs.get("n_trees")
                if n_trees:
                    clean_params[TREE_COUNT_PARAM[gbdt_kind]] = int(n_trees)
            else:
                clean_params = {k.replace(prefix, ''): v for k, v in best_params.items()}

            # Use best score from optimization
            best_trial_score = study.best_value
            trial_scores = [trial.value for trial in study.trials if trial.value is not None]

            print(f"   ✅ Optimization complete!")
            print(f"   Best score: {best_trial_score:.6f}")
            print(f"   Best params: {clean_params}")

            model_mean_score = float(best_trial_score)
            model_std_score = float(np.std(trial_scores[-5:])) if len(trial_scores) >= 5 else 0.0
            model_scores_list = trial_scores[-5:]
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

            results = {
                'mean_score': model_mean_score,
                'std_score': model_std_score,
                'individual_scores': model_scores_list,
                'best_params': clean_params,
                'n_trials': len(study.trials),
                'tuned': True,
            }

            del study
        else:
            # Model doesn't have tuning function, use default
            print(f"   ⚠️  No tuning function available for {name}, using defaults")
            scores = evaluate_with_default_params()
            model_mean_score = float(scores.mean())
            model_std_score = float(scores.std())
            model_scores_list = scores.tolist()

            if config.PROGRESS_TRACKER:
                config.PROGRESS_TRACKER.track_model_complete(
                    name,
                    mean_score=model_mean_score,
                    std_score=model_std_score
                )

            results = {
                'mean_score': model_mean_score,
                'std_score': model_std_score,
                'individual_scores': model_scores_list,
                'tuned': False,
            }
            print(f"   Score: {model_mean_score:.6f} (+/- {model_std_score:.6f})")
    else:
        # Model tuning disabled or not in tuning list
        print(f"   Using default hyperparameters...")

        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_model_start(name, total_trials=None)

        scores = evaluate_with_default_params()
        model_mean_score = float(scores.mean())
        model_std_score = float(scores.std())
        model_scores_list = scores.tolist()

        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_model_complete(
                name,
                mean_score=model_mean_score,
                std_score=model_std_score
            )

        results = {
            'mean_score': model_mean_score,
            'std_score': model_std_score,
            'individual_scores': model_scores_list,
            'tuned': False,
        }
        print(f"   Score: {model_mean_score:.6f} (+/- {model_std_score:.6f})")

    # Store best params if tuned
    if tuned_flag and used_params:
        results['best_params'] = used_params

    # Clean up memory after each model evaluation
    is_rf = name == 'Random Forest'
    logger.info(f"[Training] {name} - Starting post-evaluation cleanup (aggressive={is_rf})")
    cleanup_memory(f"After {name} evaluation", aggressive=is_rf)
    logger.info(f"[Training] {name} - Post-evaluation cleanup completed")
    logger.info(f"[Training] {name} - Training completed")

    return name, results


def load_feature_names(features_dir) -> List[str]:
    """Feature names saved next to X_train.npy (empty if the file is missing)."""
    from pathlib import Path

    path = Path(features_dir) / "feature_names.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_model_factories(problem_type: str, task: str, n_samples: int, n_features: int,
                          feature_names: Optional[List[str]] = None,
                          model_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Callable]:
    """Model factories for a problem: name -> factory(overrides) returning an unfitted estimator.

    Shared by the training stage and by diagnostics that refit a run's models
    (``diagnostics.seed_noise_test``). ``feature_names`` lets GLM-LightGBM find the
    target-encoded columns (it uses their logits). ``model_params`` (``budget.model_params``)
    fixes parameters of the neural and foundation models (``pipeline/neural.py``)."""
    is_multiclass = task == "multiclass"
    lightgbm_n_jobs_default = -1 if n_samples > 1500 else config.MEMORY_CONFIG["max_parallel_jobs"]
    lightgbm_gpu_ready = (
        LIGHTGBM_AVAILABLE
        and is_lightgbm_gpu_available()
        and not config.GPU_CONFIG.get('force_cpu', False)
    )

    # Get GPU-aware model classes using smart imports
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

    def make_scaled_factory(constructor, base_params=None):
        """Linear models see standardized columns, as they do when tuned (tuning.optimize_linear_model)."""
        inner = make_factory(constructor, base_params)

        def factory(overrides=None):
            return make_pipeline(StandardScaler(), inner(overrides))

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
            params = {'random_state': 42}
            if is_multiclass:
                params.update({'objective': 'multi:softprob', 'eval_metric': 'mlogloss'})
            else:
                params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss'})
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
            is_gpu = is_catboost_gpu_available() and not config.GPU_CONFIG['force_cpu']

            if is_regressor:
                params.update({
                    'loss_function': 'RMSE',
                    'eval_metric': 'RMSE'
                })
            else:
                # Use GPU-compatible metrics when on GPU (AUC not supported on GPU)
                if is_multiclass:
                    params.update({'loss_function': 'MultiClass', 'eval_metric': 'MultiClass'})
                else:
                    eval_metric = 'Logloss' if is_gpu else 'AUC'
                    params.update({
                        'loss_function': 'Logloss',
                        'eval_metric': eval_metric
                    })

            if is_gpu:
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
                from sklearn.base import ClassifierMixin

                class TabNetClassifierWrapper(ClassifierMixin, BaseEstimator):
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs
                        self.model = None

                    def fit(self, X, y):
                        self.model = TabNetClassifier(**self.kwargs)
                        X_train = X.astype(np.float32)
                        y_train = y.astype(np.int64)
                        self.classes_ = np.unique(y_train)
                        self.model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024)
                        return self

                    def predict(self, X):
                        X_test = X.astype(np.float32)
                        return self.model.predict(X_test)

                    def predict_proba(self, X):
                        return self.model.predict_proba(X.astype(np.float32))

                return TabNetClassifierWrapper(**params)

        return factory

    # Define model ensemble based on problem type
    # Each model has different strengths and biases
    # Models automatically use GPU (cuML) if available, otherwise CPU (sklearn)
    if problem_type == 'regression':
        models = {
            'Linear Regression': make_scaled_factory(Ridge, {'random_state': 42}),  # Linear, interpretable
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
    else:  # classification
        # cuML uses different parameter names than sklearn
        # cuML: seed, sklearn: random_state
        logistic_base_params: Dict[str, Any] = {'max_iter': 1000}

        # Check if this is cuML (Cython classes don't expose signatures properly)
        class_module = LogisticRegressionClass.__module__
        if 'cuml' in class_module:
            # cuML uses 'seed' instead of 'random_state'
            # Note: cuML LogisticRegression doesn't support random_state parameter
            pass  # No seed parameter needed for cuML LogisticRegression
        else:
            # sklearn uses 'random_state'
            logistic_base_params['random_state'] = 42

        models = {
            'Logistic Regression': make_scaled_factory(
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
                'objective': 'multiclass' if is_multiclass else 'binary',
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
            if not is_multiclass:
                from src.genML.pipeline.glm_boost import GLMBoostClassifier, te_column_indices

                glm_te_columns = te_column_indices(feature_names)
                glm_gpu_params = get_lightgbm_params() if lightgbm_gpu_ready else {}

                def glm_lgb_factory(overrides=None):
                    params = {**lightgbm_base_params, **glm_gpu_params, **(overrides or {})}
                    return GLMBoostClassifier(te_columns=glm_te_columns, **params)

                models['GLM-LightGBM'] = glm_lgb_factory
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = make_catboost_factory(cb.CatBoostClassifier)
        if TABNET_AVAILABLE:
            models['TabNet'] = make_tabnet_factory(TabNetClassifier)  # Neural network for tabular data (GPU if available)
    from src.genML.pipeline.neural import neural_factories

    models.update(neural_factories(task, model_params))  # RealMLP, TabM, TabPFN when installed
    return models


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

        # Automatically detect problem type
        problem = context.current_problem()
        if problem is not None:
            problem_type = problem.problem_type
            task = problem.task
            logger.info(f"Problem '{problem.name}': {task}, metric {problem.metric}")
        else:
            logger.info("No resolved problem; detecting problem type...")
            problem_type = detect_problem_type()
            task = (
                "regression" if problem_type == "regression"
                else ("binary" if len(np.unique(y_train)) <= 2 else "multiclass")
            )
            logger.info(f"Problem type detected: {problem_type} ({task})")

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

        original_models_to_tune = list(config.TUNING_CONFIG['models_to_tune'])


        tuning_overrides = _compute_adaptive_tuning_limits(X_train.shape)
        global_tuning_trials = tuning_overrides["global"]["n_trials"]
        global_tuning_timeout = tuning_overrides["global"]["timeout"]
        per_model_tuning_limits: Dict[str, Dict[str, Any]] = tuning_overrides["per_model"]

        for note in tuning_overrides.get("notes", []):
            print(f"⚙️ {note}")

        models = build_model_factories(problem_type, task, n_samples, n_features,
                                       feature_names=load_feature_names(config.FEATURES_DIR),
                                       model_params=((problem.budget or {}).get('model_params')
                                                     if problem is not None else None))

        # Metric, CV and budget come from the resolved problem spec
        cv = _build_cv(problem, problem_type)
        if problem is not None:
            scoring_metric = problem.scoring
            metric_name = problem.metric
        else:
            scoring_metric = 'neg_mean_squared_error' if problem_type == 'regression' else 'accuracy'
            metric_name = 'mse' if problem_type == 'regression' else 'accuracy'
        print(f"Optimizing for: {metric_name} (CV scores are higher-is-better)")
        best_score = float('-inf')

        # Shared folds (same as the OOF stage) and per-fold nested encodings, if any
        cv_config = dict(problem.cv) if problem is not None else {
            "method": "kfold" if problem_type == "regression" else "stratified_kfold",
            "n_splits": 5, "shuffle": True, "random_state": 42}
        run = config.ACTIVE_RUN
        problem_dir = run.problem_dir if run is not None else config.OUTPUTS_DIR
        fold_ids, _ = run_folds(y_train, cv_config, problem_dir, config.FEATURES_DIR)
        if cv is None:  # grouped / time folds come from the data, so sklearn gets them as-is
            from sklearn.model_selection import PredefinedSplit

            cv = PredefinedSplit(fold_ids)
        fold_features = FoldFeatures.load(config.FEATURES_DIR)
        n_folds = int(fold_ids.max()) + 1
        budget_for_folds = problem.budget if problem is not None else {}
        default_tuning_folds = n_folds if n_samples <= 200_000 else min(2, n_folds)
        # time_blocks tune on the latest periods, other methods on the first folds
        tuning_folds = tuning_fold_indices(cv_config, n_folds,
                                           int(budget_for_folds.get('tuning_folds', default_tuning_folds)))
        if fold_features is not None:
            print(f"Nested target encoding: {len(fold_features.columns)} columns swapped per fold")

        models = _filter_models(models, problem.models if problem is not None else None)
        models_to_tune = list(config.TUNING_CONFIG['models_to_tune'])
        tuning_enabled = config.TUNING_CONFIG['enabled']
        budget = problem.budget if problem is not None else {}
        if 'n_trials' in budget:
            global_tuning_trials = int(budget['n_trials'])
            per_model_tuning_limits = {}
        if 'timeout_per_model' in budget:
            global_tuning_timeout = int(budget['timeout_per_model'])
            per_model_tuning_limits = {}
        if budget.get('tuning') is False:
            tuning_enabled = False
        if budget:
            print(f"Budget from problem spec: {budget}")

        # Model evaluation pipeline with Optuna hyperparameter tuning
        # Systematically evaluate each model using cross-validation with optimized hyperparameters
        results = {}
        best_model_name = None

        # Suppress Optuna's default logging to reduce clutter
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Track overall model training progress
        total_models = len(models)
        models_completed = 0

        # Categorize models into CPU and GPU pools for parallel/sequential execution
        cpu_models, gpu_models = _categorize_models_by_resource(models, problem_type)

        # Check if parallel training is enabled
        parallel_enabled = config.PARALLEL_TRAINING_CONFIG.get('enabled', True)
        max_cpu_workers_config = config.PARALLEL_TRAINING_CONFIG.get('max_cpu_workers', 3)

        print(f"\n{'='*60}")
        print(f"Model Training Strategy:")
        if parallel_enabled and len(cpu_models) > 1:
            print(f"  CPU Models ({len(cpu_models)}): {list(cpu_models.keys())} - Will train in parallel")
        else:
            print(f"  CPU Models ({len(cpu_models)}): {list(cpu_models.keys())} - Will train sequentially")
        print(f"  GPU Models ({len(gpu_models)}): {list(gpu_models.keys())} - Will train sequentially")
        print(f"{'='*60}")

        # Train CPU models in parallel using ThreadPoolExecutor
        if cpu_models and parallel_enabled and len(cpu_models) > 1:
            # Determine max workers - limit to avoid overloading system
            max_workers = min(len(cpu_models), max_cpu_workers_config) if max_cpu_workers_config > 0 else min(len(cpu_models), 3)
            print(f"\n🚀 Training CPU models in parallel (max {max_workers} concurrent)...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all CPU model training tasks
                future_to_model = {}
                for name, model_factory in cpu_models.items():
                    future = executor.submit(
                        _train_single_model,
                        name=name,
                        model_factory=model_factory,
                        X_train=X_train,
                        y_train=y_train,
                        problem_type=problem_type,
                        cv=cv,
                        scoring_metric=scoring_metric,
                        tuning_enabled=tuning_enabled,
                        models_to_tune=models_to_tune,
                        seed=run_seed(budget),
                        per_model_tuning_limits=per_model_tuning_limits,
                        global_tuning_trials=global_tuning_trials,
                        global_tuning_timeout=global_tuning_timeout,
                        base_optuna_n_trials=config.TUNING_CONFIG['n_trials'],
                        base_optuna_timeout=config.TUNING_CONFIG['timeout'],
                        is_cpu_pool=True,
                        fold_ids=fold_ids,
                        fold_features=fold_features,
                        tuning_folds=tuning_folds,
                        gbdt_budget=budget,
                    )
                    future_to_model[future] = name

                # Collect results as they complete
                for future in as_completed(future_to_model):
                    name = future_to_model[future]
                    try:
                        model_name, model_results = future.result()
                        results[model_name] = model_results


                        # Update progress
                        models_completed += 1
                        if config.PROGRESS_TRACKER:
                            config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

                        print(f"✅ {model_name} training completed")
                    except Exception as e:
                        logger.error(f"CPU model {name} training failed: {e}")
                        print(f"❌ {name} training failed: {e}")
                        # Continue with other models
                        models_completed += 1
                        if config.PROGRESS_TRACKER:
                            config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

        elif cpu_models:
            # Parallel training disabled or only 1 CPU model - train sequentially
            print(f"\n⚙️  Training CPU models sequentially...")

            for name, model_factory in cpu_models.items():
                try:
                    model_name, model_results = _train_single_model(
                        name=name,
                        model_factory=model_factory,
                        X_train=X_train,
                        y_train=y_train,
                        problem_type=problem_type,
                        cv=cv,
                        scoring_metric=scoring_metric,
                        tuning_enabled=tuning_enabled,
                        models_to_tune=models_to_tune,
                        seed=run_seed(budget),
                        per_model_tuning_limits=per_model_tuning_limits,
                        global_tuning_trials=global_tuning_trials,
                        global_tuning_timeout=global_tuning_timeout,
                        base_optuna_n_trials=config.TUNING_CONFIG['n_trials'],
                        base_optuna_timeout=config.TUNING_CONFIG['timeout'],
                        is_cpu_pool=True,
                        fold_ids=fold_ids,
                        fold_features=fold_features,
                        tuning_folds=tuning_folds,
                        gbdt_budget=budget,
                    )
                    results[model_name] = model_results


                    # Update progress
                    models_completed += 1
                    if config.PROGRESS_TRACKER:
                        config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

                    print(f"✅ {model_name} training completed")
                except Exception as e:
                    logger.error(f"CPU model {name} training failed: {e}")
                    print(f"❌ {name} training failed: {e}")
                    # Continue with other models
                    models_completed += 1
                    if config.PROGRESS_TRACKER:
                        config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

        # Train GPU models sequentially to avoid OOM
        if gpu_models:
            print(f"\n🎮 Training GPU models sequentially to avoid memory issues...")

            for name, model_factory in gpu_models.items():
                try:
                    model_name, model_results = _train_single_model(
                        name=name,
                        model_factory=model_factory,
                        X_train=X_train,
                        y_train=y_train,
                        problem_type=problem_type,
                        cv=cv,
                        scoring_metric=scoring_metric,
                        tuning_enabled=tuning_enabled,
                        models_to_tune=models_to_tune,
                        seed=run_seed(budget),
                        per_model_tuning_limits=per_model_tuning_limits,
                        global_tuning_trials=global_tuning_trials,
                        global_tuning_timeout=global_tuning_timeout,
                        base_optuna_n_trials=config.TUNING_CONFIG['n_trials'],
                        base_optuna_timeout=config.TUNING_CONFIG['timeout'],
                        is_cpu_pool=False,
                        fold_ids=fold_ids,
                        fold_features=fold_features,
                        tuning_folds=tuning_folds,
                        gbdt_budget=budget,
                    )
                    results[model_name] = model_results


                    # Update progress
                    models_completed += 1
                    if config.PROGRESS_TRACKER:
                        config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

                    print(f"✅ {model_name} training completed")
                except Exception as e:
                    logger.error(f"GPU model {name} training failed: {e}")
                    print(f"❌ {name} training failed: {e}")
                    # Continue with other models
                    models_completed += 1
                    if config.PROGRESS_TRACKER:
                        config.PROGRESS_TRACKER.update_model_training_stage(models_completed, total_models)

        if problem is None:
            from src.genML.pipeline.dataset import load_stage_table

            train_df, test_df = load_stage_table("train"), load_stage_table("test")
            problem = context.ensure_problem(train_df, test_df)

        def params_for(name, model_result):
            params = dict(model_result.get('best_params') or {}) if model_result.get('tuned') else {}
            return params or None

        oof_summary = _run_oof_stage(problem, task, results, models, params_for, X_train, y_train)
        best_model_name = oof_summary["final_name"]
        best_score = oof_summary["final_score"]

        # Optional: refit the best single model on all rows and predict with it
        best_model = None
        if problem.budget.get('refit_full') and oof_summary["final_source"] == "single":
            print(f"Refitting {best_model_name} on all {len(y_train)} rows (budget.refit_full)")
            best_model = models[best_model_name](params_for(best_model_name, results[best_model_name]))
            best_model.fit(X_train, y_train)
            X_test_full = np.load(config.FEATURES_DIR / 'X_test.npy')
            n_classes = len(problem.classes) if problem.is_classification else 0
            np.save(config.FEATURES_DIR / 'final_test_predictions.npy',
                    predict_matrix(best_model, X_test_full, task, n_classes))

        # Save the trained model for prediction stage
        # Timestamped filename prevents overwrites and enables model versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = (
            f'best_model_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl' if best_model is not None else None
        )

        if best_model is not None:
            joblib.dump(best_model, config.MODELS_DIR / model_filename)

        # Save detailed cross-validation results for analysis
        # Provides transparency into model selection process
        cv_results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Mean_Score': [results[name]['mean_score'] for name in results.keys()],
            'Std_Score': [results[name]['std_score'] for name in results.keys()],
            'OOF_Score': [results[name].get('oof_score') for name in results.keys()],
        })
        cv_results_df.to_csv(config.MODELS_DIR / 'cross_validation_results.csv', index=False)

        # Prepare results with problem type and GPU information
        model_results = {
            "status": "success",
            "problem_type": problem_type,
            "scoring_metric": metric_name,
            "task": task,
            "best_model": best_model_name,
            "best_score": float(best_score),
            "greater_is_better": oof_summary["greater_is_better"],
            "score_source": "out-of-fold",
            "final_predictions": oof_summary,
            "all_results": results,
            "model_filename": model_filename,
            "timestamp": timestamp,
            "model_saved": str(config.MODELS_DIR / model_filename) if model_filename else None,
            "gpu_acceleration": {
                "cuml_available": is_cuml_available(),
                "xgboost_gpu_available": is_xgboost_gpu_available(),
                "gpu_used_for_best_model": is_cuml_available() or (is_xgboost_gpu_available() and 'XGBoost' in best_model_name),
                "gpu_info": gpu_config
            }
        }

        # Save model training report
        with open(config.REPORTS_DIR / 'model_training_report.json', 'w') as f:
            json.dump(model_results, f, indent=2, default=str)

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
