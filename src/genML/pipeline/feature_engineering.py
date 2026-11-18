"""Feature engineering stage and AI-assisted feature utilities."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.genML.pipeline import config, context
from src.genML.pipeline.targets import prepare_target

logger = logging.getLogger(__name__)


def _add_encodings(problem, train_df, test_df, train_features, test_features, analysis_results, options):
    """Append fold-safe target/count/pair encodings (see features/encoding.py)."""
    from src.genML.features.encoding import build_encodings, nested_target_encodings
    from src.genML.pipeline.oof import run_folds

    y_model, _ = prepare_target(
        train_df[problem.target], task=problem.task,
        metric_transform=problem.metric_obj.target_transform, positive_class=problem.positive_class,
    )
    run = config.ACTIVE_RUN
    problem_dir = run.problem_dir if run is not None else config.OUTPUTS_DIR
    fold_ids, _ = run_folds(y_model, problem.cv, problem_dir, config.FEATURES_DIR, train_df)

    excluded = {problem.target, problem.id_column, *problem.drop_columns}
    declared_cats = [c for c, role in problem.column_types.items() if role == "categorical"]
    categorical = [c for c in list(dict.fromkeys(declared_cats + analysis_results.get("categorical_columns", [])))
                   if c not in excluded]
    numeric = [c for c in analysis_results.get("numerical_columns", []) if c not in excluded]
    n_classes = int(train_df[problem.target].nunique()) if problem.task == "multiclass" else 0

    enc_train, enc_test, report = build_encodings(
        train_df, test_df, y_model, fold_ids, problem.task, n_classes, categorical, numeric, options
    )
    fold_blocks = nested_target_encodings(
        train_df, test_df, y_model, fold_ids, problem.task, n_classes, report, options
    )
    enc_train = enc_train.set_axis(train_features.index)
    enc_test = enc_test.set_axis(test_features.index)
    return (pd.concat([train_features, enc_train], axis=1),
            pd.concat([test_features, enc_test], axis=1), report, fold_blocks)


def _raw_features(train_df: pd.DataFrame, test_df: pd.DataFrame, problem) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Original columns only: numbers as they are, other columns as integer codes.

    ``feature_engineering.base: raw`` uses this instead of the automated transforms
    (scaling, log, squares, bins). Tree models gain nothing from monotone transforms
    of a column, and the duplicates dilute column sampling.
    """
    excluded = {problem.target, problem.id_column, *problem.drop_columns}
    columns = [c for c in train_df.columns if c in test_df.columns and c not in excluded]
    train_out, test_out = pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)
    for column in columns:
        role = problem.column_types.get(column)
        series = train_df[column]
        if role != "categorical" and (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
            train_out[column] = series.astype(np.float32)  # models get float32 anyway; halves memory
            test_out[column] = test_df[column].astype(np.float32)
        else:
            values = pd.concat([series, test_df[column]]).astype(str)
            codes = {value: code for code, value in enumerate(sorted(values.unique()))}
            train_out[column] = series.astype(str).map(codes).astype(np.float32)
            test_out[column] = test_df[column].astype(str).map(codes).astype(np.float32)
    return train_out, test_out


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def engineer_features() -> str:
    """
    Create engineered features from the loaded dataset using automated feature engineering.

    This function now uses the AutoFeatureEngine to automatically detect data types,
    apply appropriate feature processors, and perform intelligent feature selection.
    It can adapt to different datasets while maintaining the same interface.

    Returns:
        JSON string with feature engineering results and metadata
    """
    try:
        from src.genML.pipeline.dataset import load_stage_table

        train_df, test_df = load_stage_table("train"), load_stage_table("test")

        from src.genML.features import AutoFeatureEngine

        config_dict = {
            "max_features": 200,
            "enable_feature_selection": True,
            "feature_importance_threshold": 0.001,
            "manual_type_hints": {},
            "interaction_pairs": [],
            "numerical_config": {
                "enable_scaling": True,
                "enable_binning": True,
                "enable_polynomial": True,
                "polynomial_degree": 2,
                "n_bins": 5,
            },
            "categorical_config": {
                "encoding_method": "target",
                "enable_frequency": True,
                "max_categories": 20,
            },
            "text_config": {
                "enable_basic_features": True,
                "enable_tfidf": False,
                "enable_patterns": True,
            },
            "feature_selection": {
                "max_features": 200,
                "enable_statistical_tests": True,
                "enable_model_based": True,
                "selection_strategy": "union",
                "feature_importance_threshold": 0.001,
            },
        }

        problem = context.ensure_problem(train_df, test_df)
        target_col = problem.target

        # Column roles declared in the spec; drop columns and the ID are excluded
        type_hints = dict(problem.column_types)
        for column in problem.drop_columns:
            type_hints[column] = "id"
        if problem.id_column and problem.id_column in train_df.columns:
            type_hints[problem.id_column] = "id"
        config_dict["manual_type_hints"] = type_hints
        fe_overrides = dict(problem.feature_engineering)
        encoding_options = fe_overrides.pop("encodings", None)
        base_mode = fe_overrides.pop("base", "auto")

        # Label-free derived columns, composite keys, group statistics, drift filter (features/derived.py)
        from src.genML.features.derived import DERIVED_KEYS
        from src.genML.features.derived import apply as apply_derived
        from src.genML.pipeline.oof import cv_column, needs_column, run_folds

        derived_settings = {key: fe_overrides.pop(key) for key in DERIVED_KEYS if key in fe_overrides}
        derived_report = None
        if derived_settings:
            protected = [c for c in (target_col, problem.id_column, cv_column(problem.cv)) if c]
            derived_report = apply_derived(train_df, test_df, derived_settings, protected, target=target_col)
            for part, value in derived_report.items():
                if part == "drift_filter":
                    print(f"Drift filter: dropped {len(value['dropped'])} column(s) {value['dropped'][:10]}")
                elif part == "group_stats":
                    print(f"Group statistics: {len(value['columns'])} columns")
                else:
                    print(f"{part}: {value}")
        if base_mode not in ("auto", "raw"):
            raise ValueError(f"feature_engineering.base must be 'auto' or 'raw', not {base_mode!r}")
        config_dict = _deep_merge(config_dict, fe_overrides)

        feature_engine = AutoFeatureEngine(config_dict)
        analysis_results = feature_engine.analyze_data(train_df)

        # Track progress: Data analysis complete
        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_stage_progress(2, 20, "Analyzing data types and patterns")

        print(f"Target column: {target_col} ({problem.task}, metric {problem.metric})")

        # Track progress: Fitting feature engine
        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_stage_progress(2, 40, "Fitting feature transformations")

        if base_mode == "raw":
            train_features, test_features = _raw_features(train_df, test_df, problem)
            print(f"Base features: {train_features.shape[1]} original columns (feature_engineering.base: raw)")
        else:
            feature_engine.fit(train_df, target_col)

        # Track progress: Feature engine fitted
        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_stage_progress(2, 60, "Transforming features")

        if base_mode != "raw":
            train_features = feature_engine.transform(train_df)
            test_features = feature_engine.transform(test_df)
        feature_columns = list(train_features.columns)

        # Track progress: Features transformed
        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_stage_progress(2, 80, "Finalizing feature matrices")

        # Align test features to training feature order to avoid column mismatches.
        for column in feature_columns:
            if column not in test_features.columns:
                test_features[column] = 0.0
        test_features = test_features[feature_columns]

        encoding_report = None
        fold_blocks = None
        if encoding_options:
            train_features, test_features, encoding_report, fold_blocks = _add_encodings(
                problem, train_df, test_df, train_features, test_features, analysis_results, encoding_options
            )
            feature_columns = list(train_features.columns)
            print(f"Added {encoding_report['n_features']} leak-safe encoding features")

        # Straight to float32 arrays (no combined train+test copy: large tables run out of memory)
        train_features_final = train_features[feature_columns].to_numpy(dtype=np.float32, copy=False)
        test_features_final = test_features[feature_columns].to_numpy(dtype=np.float32, copy=False)

        train_target, transform_info = prepare_target(
            train_df[target_col],
            task=problem.task,
            metric_transform=problem.metric_obj.target_transform,
            positive_class=problem.positive_class,
        )
        joblib.dump(transform_info, config.FEATURES_DIR / "target_transform.pkl")
        if needs_column(problem.cv):  # record this run's grouped / time folds for the training stage
            run = config.ACTIVE_RUN
            run_folds(train_target, problem.cv, run.problem_dir if run is not None else config.OUTPUTS_DIR,
                      config.FEATURES_DIR, train_df)
        if transform_info.get("classes") is not None:
            problem.classes = transform_info["classes"]
            context.set_active_problem(problem)
        print(f"Target preparation: {transform_info.get('type') or 'none'}")

        # Track progress: Saving features
        if config.PROGRESS_TRACKER:
            config.PROGRESS_TRACKER.track_stage_progress(2, 95, "Saving feature matrices")

        np.save(config.FEATURES_DIR / "X_train.npy", train_features_final)
        np.save(config.FEATURES_DIR / "X_test.npy", test_features_final)
        if fold_blocks is not None:
            from src.genML.pipeline.oof import FoldFeatures

            FoldFeatures.save(
                config.FEATURES_DIR,
                [feature_columns.index(name) for name in fold_blocks["columns"]],
                fold_blocks["columns"], fold_blocks["train"], fold_blocks["test"],
            )
            print(f"Nested target encoding: {len(fold_blocks['columns'])} columns x "
                  f"{len(fold_blocks['train'])} fold blocks")
        np.save(config.FEATURES_DIR / "y_train.npy", train_target)

        scaler = StandardScaler()
        scaler.mean_ = np.zeros(train_features_final.shape[1])
        scaler.scale_ = np.ones(train_features_final.shape[1])
        joblib.dump(scaler, config.FEATURES_DIR / "scaler.pkl")

        with open(config.FEATURES_DIR / "feature_names.txt", "w") as f:
            f.write("\n".join(feature_columns))

        if base_mode != "raw":
            feature_engine.save_report(config.REPORTS_DIR / "automated_feature_engineering_report.json")

        metadata = {
            "status": "success",
            "features_used": feature_columns,
            "train_shape": train_features_final.shape,
            "test_shape": test_features_final.shape,
            "feature_engineering_method": "raw" if base_mode == "raw" else "automated",
            "nested_target_encoding": fold_blocks is not None,
            "total_features_generated": len(feature_columns),
            "target_column": target_col,
            "encodings": encoding_report,
            "derived": derived_report,
            "feature_stats": {
                "mean": np.nanmean(train_features_final, axis=0).tolist(),
                "std": np.nanstd(train_features_final, axis=0).tolist(),
            },
            "target_transformation": {
                "applied": transform_info["applied"],
                "type": transform_info.get("type"),
                "original_range": (
                    [transform_info.get("original_min"), transform_info.get("original_max")]
                    if transform_info["applied"]
                    else None
                ),
                "transformed_range": [float(np.min(train_target)), float(np.max(train_target))],
            },
            "target_distribution": {
                "mean": float(np.mean(train_target)),
                "std": float(np.std(train_target)),
                "min": float(np.min(train_target)),
                "max": float(np.max(train_target)),
            },
        }

        with open(config.REPORTS_DIR / "feature_engineering_report.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return json.dumps(metadata, indent=2)

    except Exception as exc:  # pragma: no cover - full pipeline tests cover primary paths
        error_trace = traceback.format_exc()
        logger.error("Feature engineering failed: %s\n%s", exc, error_trace)
        failure_payload = {
            "status": "failed",
            "error": str(exc),
            "traceback": error_trace,
        }
        return json.dumps(failure_payload, indent=2)
