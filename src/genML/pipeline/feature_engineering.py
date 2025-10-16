"""Feature engineering stage and AI-assisted feature utilities."""

from __future__ import annotations

import json
import logging
import traceback
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.genML.pipeline import config

logger = logging.getLogger(__name__)


def _normalize_ai_feature_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize an AI-generated feature specification."""
    if not isinstance(spec, dict):
        raise ValueError("feature spec must be a dictionary")

    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError("missing feature name")
    name = name.replace(" ", "_")

    operation = str(spec.get("operation", "")).strip().lower()
    if operation not in config.AI_SUPPORTED_FEATURE_OPERATIONS:
        raise ValueError(f"unsupported operation '{operation}'")

    inputs = spec.get("inputs", [])
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a non-empty list")
    inputs = [str(inp).strip() for inp in inputs if str(inp).strip()]
    if not inputs:
        raise ValueError("inputs cannot be empty strings")

    if operation in {"ratio", "difference"} and len(inputs) != 2:
        raise ValueError(f"{operation} operation requires exactly 2 inputs")
    if operation == "log" and len(inputs) != 1:
        raise ValueError("log operation requires exactly 1 input")
    if operation == "binary_threshold" and len(inputs) != 1:
        raise ValueError("binary_threshold operation requires exactly 1 input")
    if operation in {"product", "sum"} and len(inputs) < 2:
        raise ValueError(f"{operation} operation requires at least 2 inputs")

    parameters = spec.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a dictionary")

    if operation == "binary_threshold":
        if "threshold" not in parameters:
            raise ValueError("binary_threshold operation requires 'threshold' parameter")
        try:
            parameters["threshold"] = float(parameters["threshold"])
        except (TypeError, ValueError):
            raise ValueError("binary_threshold threshold must be a numeric value")

    expected_impact = str(spec.get("expected_impact", "")).strip().lower()
    rationale = str(spec.get("rationale", "")).strip()

    return {
        "name": name,
        "operation": operation,
        "inputs": inputs,
        "parameters": parameters,
        "expected_impact": expected_impact,
        "rationale": rationale,
    }


def _resolve_unique_feature_name(base_name: str, existing: set[str]) -> str:
    """Generate a unique feature name avoiding collisions in current feature set."""
    if base_name not in existing:
        return base_name

    index = 2
    while True:
        candidate = f"{base_name}_{index}"
        if candidate not in existing:
            return candidate
        index += 1


def _get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Retrieve column as float series, handling missing values."""
    if column not in df.columns:
        raise ValueError(f"Missing column '{column}' for AI feature")
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        try:
            series = pd.to_numeric(series, errors="coerce")
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to coerce column '{column}' to numeric: {exc}") from exc
    return series.astype(float)


def _compute_ai_feature_series(
    operation: str,
    inputs: List[str],
    parameters: Dict[str, Any],
    source_df: pd.DataFrame,
) -> pd.Series:
    """Compute a new feature according to AI specification."""
    if operation == "ratio":
        numerator = _get_numeric_series(source_df, inputs[0])
        denominator = _get_numeric_series(source_df, inputs[1])
        denominator = denominator.replace({0: config.AI_FEATURE_EPSILON})
        return numerator / denominator

    if operation == "difference":
        first = _get_numeric_series(source_df, inputs[0])
        second = _get_numeric_series(source_df, inputs[1])
        return first - second

    if operation == "product":
        series_list = [_get_numeric_series(source_df, col) for col in inputs]
        result = series_list[0].copy()
        for series in series_list[1:]:
            result *= series
        return result

    if operation == "sum":
        series_list = [_get_numeric_series(source_df, col) for col in inputs]
        total = series_list[0].copy()
        for series in series_list[1:]:
            total += series
        return total

    if operation == "log":
        series = _get_numeric_series(source_df, inputs[0])
        return np.log(series.replace({0: config.AI_FEATURE_EPSILON}).abs())

    if operation == "binary_threshold":
        threshold = parameters.get("threshold", 0)
        series = _get_numeric_series(source_df, inputs[0])
        return (series > threshold).astype(int)

    raise ValueError(f"Unsupported AI feature operation '{operation}'")


def build_derived_feature_context(
    df_sample: pd.DataFrame,
    target_col: Optional[str],
    feature_importances: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """Build summary context for AI feature ideation advisor."""
    context: Dict[str, Any] = {}

    if df_sample.empty:
        return context

    numeric_columns = [
        col for col in df_sample.columns if col != target_col and pd.api.types.is_numeric_dtype(df_sample[col])
    ]
    categorical_columns = [
        col for col in df_sample.columns if col != target_col and not pd.api.types.is_numeric_dtype(df_sample[col])
    ]

    context["numeric_columns"] = numeric_columns[:20]
    context["categorical_columns"] = categorical_columns[:20]

    if numeric_columns:
        variance_series = df_sample[numeric_columns].var().dropna().sort_values(ascending=False)
        top_variance_cols = variance_series.head(6).index.tolist()
        context["high_variance_numeric"] = {col: float(variance_series[col]) for col in top_variance_cols}

        ratio_pairs = [list(pair) for pair in combinations(top_variance_cols, 2)]
        context["candidate_ratio_pairs"] = ratio_pairs[:5]
        context["candidate_difference_pairs"] = ratio_pairs[:5]

        try:
            corr_matrix = df_sample[numeric_columns].corr().abs()
            tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr_pairs = (
                corr_matrix.where(tri_mask).stack().sort_values(ascending=False).head(5).items()
            )
            context["high_correlation_pairs"] = [
                {"features": [feat_a, feat_b], "abs_correlation": float(value)}
                for (feat_a, feat_b), value in high_corr_pairs
            ]
        except Exception:  # pragma: no cover - correlation failures are rare and non-fatal
            context["high_correlation_pairs"] = []

    if target_col and target_col in df_sample.columns and numeric_columns:
        target_series = df_sample[target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            correlations = (
                df_sample[numeric_columns]
                .corrwith(target_series)
                .dropna()
                .abs()
                .sort_values(ascending=False)
                .head(5)
            )
            context["target_correlations"] = {feature: float(score) for feature, score in correlations.items()}
        else:
            grouped_means: Dict[str, Dict[str, float]] = {}
            target_groups = target_series.astype(str)
            for col in numeric_columns[:5]:
                try:
                    means = df_sample.groupby(target_groups)[col].mean()
                    grouped_means[col] = {str(group): float(value) for group, value in means.items()}
                except Exception:
                    continue
            if grouped_means:
                context["target_group_means"] = grouped_means

    if feature_importances:
        sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        context["top_feature_importances"] = [
            {"feature": name, "importance": float(score)} for name, score in sorted_importances[:5]
        ]

        untouched_numeric = [col for col in numeric_columns if col not in feature_importances]
        if untouched_numeric:
            context["untouched_numeric_columns"] = untouched_numeric[:5]

    return context


def apply_ai_generated_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    suggestions: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Apply AI-generated feature specifications to the engineered feature matrices.

    Returns:
        Tuple of (train_features, test_features, summary_dict)
    """
    summary: Dict[str, Any] = {
        "status": "skipped",
        "attempted": 0,
        "successful": 0,
        "created_features": [],
        "failed_features": [],
    }

    if not suggestions or suggestions.get("status") != "success":
        summary["reason"] = "no_valid_suggestions"
        return train_features, test_features, summary

    feature_specs = suggestions.get("engineered_features") or []
    if not feature_specs:
        summary["reason"] = "no_engineered_features"
        return train_features, test_features, summary

    train_features = train_features.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    train_source = train_raw.reset_index(drop=True)
    test_source = test_raw.reset_index(drop=True)

    summary["attempted"] = len(feature_specs)
    existing_columns = set(train_features.columns)

    for spec in feature_specs:
        try:
            normalized = _normalize_ai_feature_spec(spec)
            feature_name = _resolve_unique_feature_name(normalized["name"], existing_columns)

            missing_inputs = [
                column
                for column in normalized["inputs"]
                if column not in train_source.columns or column not in test_source.columns
            ]
            if missing_inputs:
                failure_record = {
                    "name": normalized["name"],
                    "operation": normalized["operation"],
                    "reason": f"missing source columns: {', '.join(missing_inputs)}",
                }
                summary["failed_features"].append(failure_record)
                logger.debug(
                    "Skipping AI feature %s due to missing inputs: %s",
                    normalized["name"],
                    ", ".join(missing_inputs),
                )
                continue

            train_series = _compute_ai_feature_series(
                normalized["operation"],
                normalized["inputs"],
                normalized["parameters"],
                train_source,
            )
            test_series = _compute_ai_feature_series(
                normalized["operation"],
                normalized["inputs"],
                normalized["parameters"],
                test_source,
            )

            train_features[feature_name] = train_series
            test_features[feature_name] = test_series
            existing_columns.add(feature_name)

            summary["successful"] += 1
            summary["created_features"].append(
                {
                    "name": feature_name,
                    "operation": normalized["operation"],
                    "inputs": normalized["inputs"],
                    "expected_impact": normalized["expected_impact"],
                    "rationale": normalized["rationale"],
                }
            )
        except Exception as exc:
            failure_record = {
                "name": spec.get("name"),
                "operation": spec.get("operation"),
                "reason": str(exc),
            }
            summary["failed_features"].append(failure_record)
            logger.warning("Failed to apply AI-generated feature %s: %s", failure_record["name"], exc)

    if summary["successful"] > 0:
        summary["status"] = "applied"
    else:
        summary["reason"] = summary.get("reason", "no_features_applied")

    return train_features, test_features, summary


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
        train_df = pd.read_pickle(config.DATA_DIR / "train_data.pkl")
        test_df = pd.read_pickle(config.DATA_DIR / "test_data.pkl")

        from src.genML.features import AutoFeatureEngine

        config_dict = {
            "max_features": 200,
            "enable_feature_selection": True,
            "feature_importance_threshold": 0.001,
            "manual_type_hints": {
                "num_lanes": "numerical",
                "curvature": "numerical",
                "speed_limit": "numerical",
                "holiday": "categorical",
                "num_reported_accidents": "numerical",
            },
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

        feature_engine = AutoFeatureEngine(config_dict)
        analysis_results = feature_engine.analyze_data(train_df)

        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        train_only_cols = train_cols - test_cols
        id_patterns = {"id", "ID", "Id", "index", "Index"}
        potential_targets = [col for col in train_only_cols if col not in id_patterns]
        target_col = potential_targets[0] if potential_targets else None

        if target_col is None and analysis_results.get("target_candidates"):
            target_col = analysis_results["target_candidates"][0]

        if target_col:
            print(f"Detected target column: {target_col}")
        else:
            print("Warning: Could not detect target column")

        feature_engine.fit(train_df, target_col)

        ai_feature_suggestions: Optional[Dict[str, Any]] = None

        if config.AI_ADVISORS_CONFIG["enabled"] and config.AI_ADVISORS_CONFIG["feature_ideation"]["enabled"]:
            try:
                print("\n" + "=" * 60)
                print("🤖 AI Feature Ideation Advisor")
                print("=" * 60)
                print("Analyzing current features and suggesting improvements...")

                from src.genML.ai_advisors import FeatureIdeationAdvisor, OpenAIClient

                openai_client = OpenAIClient(config=config.AI_ADVISORS_CONFIG["openai_config"])

                if openai_client.is_available():
                    current_features = [col for col in train_df.columns if col != target_col]

                    feature_importances = None
                    if hasattr(feature_engine, "feature_selector") and hasattr(
                        feature_engine.feature_selector, "feature_importances_"
                    ):
                        importance_dict = {
                            feat: float(imp)
                            for feat, imp in zip(
                                current_features, feature_engine.feature_selector.feature_importances_
                            )
                        }
                        feature_importances = importance_dict

                    advisor = FeatureIdeationAdvisor(openai_client=openai_client)

                    sample_size = config.AI_ADVISORS_CONFIG["feature_ideation"]["sample_size"]
                    df_sample = train_df.head(sample_size)

                    detected_domain = analysis_results.get("domain_analysis", {}).get("detected_domains", [None])[0]

                    derived_context = build_derived_feature_context(
                        df_sample=df_sample,
                        target_col=target_col,
                        feature_importances=feature_importances,
                    )

                    if derived_context.get("candidate_ratio_pairs"):
                        ratio_preview = [
                            " / ".join(pair) for pair in derived_context["candidate_ratio_pairs"][:3]
                        ]
                        print(f"   Candidate ratio pairs: {', '.join(ratio_preview)}")

                    if derived_context.get("target_correlations"):
                        top_corr = list(derived_context["target_correlations"].items())[:3]
                        corr_preview = [f"{feat} ({score:.2f})" for feat, score in top_corr]
                        if corr_preview:
                            print(f"   Strong target correlations: {', '.join(corr_preview)}")

                    ai_feature_suggestions = advisor.generate_feature_ideas(
                        dataset_summary={
                            "target_column": target_col,
                            "problem_type": analysis_results.get("problem_type"),
                            "detected_domain": detected_domain,
                            "feature_importances": feature_importances,
                            "sample_rows": df_sample.to_dict("records"),
                            "derived_context": derived_context,
                        },
                        existing_features=current_features,
                    )

                    if ai_feature_suggestions.get("status") == "success":
                        print("AI Feature Advisor successfully generated suggestions.")
                        print(
                            json.dumps(
                                ai_feature_suggestions.get("summary", {}) or {},
                                indent=2,
                            )
                        )
                    else:
                        print("AI Feature Advisor did not produce successful suggestions.")
                else:
                    print("OpenAI client unavailable; skipping feature ideation.")
            except Exception as exc:  # pragma: no cover - advisor failures do not break pipeline
                logger.warning("AI Feature Ideation Advisor failed: %s", exc)
                ai_feature_suggestions = None

        train_features, test_features = feature_engine.transform(train_df, test_df)
        feature_columns = list(train_features.columns)
        feature_metadata = feature_engine.get_metadata()

        if ai_feature_suggestions:
            train_features, test_features, ai_feature_summary = apply_ai_generated_features(
                train_raw=train_df,
                test_raw=test_df,
                train_features=train_features,
                test_features=test_features,
                suggestions=ai_feature_suggestions,
            )
        else:
            ai_feature_summary = {
                "status": "skipped",
                "reason": "advisor_disabled" if not config.AI_ADVISORS_CONFIG["enabled"] else "no_suggestions",
                "created_features": [],
            }

        combined_features = pd.concat(
            [
                train_features.assign(_is_train=1),
                test_features.assign(_is_train=0),
            ],
            ignore_index=True,
        )

        train_features_final = combined_features.loc[combined_features["_is_train"] == 1, feature_columns].to_numpy(
            dtype=np.float32, copy=False
        )
        test_features_final = combined_features.loc[combined_features["_is_train"] == 0, feature_columns].to_numpy(
            dtype=np.float32, copy=False
        )

        if target_col and target_col in train_df.columns:
            train_target = train_df[target_col].values
        else:
            train_target = train_df.iloc[:, -1].values

        target_min = np.min(train_target)
        target_max = np.max(train_target)
        apply_logit_transform = False

        if target_min >= 0 and target_max <= 1 and target_max > target_min:
            apply_logit_transform = True
            print(f"Detected bounded target in [{target_min:.3f}, {target_max:.3f}]")
            print("Applying logit transformation: logit(y) = log(y / (1-y))")

            epsilon = 1e-7
            train_target_clipped = np.clip(train_target, epsilon, 1 - epsilon)
            train_target_transformed = np.log(train_target_clipped / (1 - train_target_clipped))

            transform_info = {
                "applied": True,
                "type": "logit",
                "original_min": float(target_min),
                "original_max": float(target_max),
                "epsilon": epsilon,
            }
            joblib.dump(transform_info, config.FEATURES_DIR / "target_transform.pkl")
            train_target = train_target_transformed
        else:
            transform_info = {"applied": False}
            joblib.dump(transform_info, config.FEATURES_DIR / "target_transform.pkl")
            print("Target transformation: None (unbounded target)")

        np.save(config.FEATURES_DIR / "X_train.npy", train_features_final)
        np.save(config.FEATURES_DIR / "X_test.npy", test_features_final)
        np.save(config.FEATURES_DIR / "y_train.npy", train_target)

        scaler = StandardScaler()
        scaler.mean_ = np.zeros(train_features_final.shape[1])
        scaler.scale_ = np.ones(train_features_final.shape[1])
        joblib.dump(scaler, config.FEATURES_DIR / "scaler.pkl")

        with open(config.FEATURES_DIR / "feature_names.txt", "w") as f:
            f.write("\n".join(feature_columns))

        feature_engine.save_report(config.REPORTS_DIR / "automated_feature_engineering_report.json")

        metadata = {
            "status": "success",
            "features_used": feature_columns,
            "train_shape": train_features_final.shape,
            "test_shape": test_features_final.shape,
            "feature_engineering_method": "automated",
            "total_features_generated": len(feature_columns),
            "detected_domains": analysis_results.get("domain_analysis", {}).get("detected_domains", []),
            "target_column": target_col,
            "ai_feature_summary": ai_feature_summary,
            "ai_generated_features": [feat["name"] for feat in ai_feature_summary.get("created_features", [])],
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
