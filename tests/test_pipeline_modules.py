"""
Tests for the refactored pipeline modules.

Covers dataset loading, feature engineering helpers, AI tuning utilities,
model selection guidance, and compatibility re-exports via src.genML.tools.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.genML import tools as legacy_tools
from src.genML.pipeline import (
    engineer_features,
    generate_predictions,
    load_dataset,
    train_model_pipeline,
)
from src.genML.pipeline import config
from src.genML.pipeline.ai_tuning import (
    build_ai_tuning_override_details,
    extract_override_values,
)
from src.genML.pipeline.feature_engineering import apply_ai_generated_features
from src.genML.pipeline.model_advisor import (
    apply_model_selection_guidance,
    detect_problem_type,
)


@pytest.fixture
def pipeline_paths(tmp_path, monkeypatch):
    """Override pipeline directories with temporary locations."""
    outputs_dir = tmp_path / "outputs"
    mapping = {
        "OUTPUTS_DIR": outputs_dir,
        "DATA_DIR": outputs_dir / "data",
        "FEATURES_DIR": outputs_dir / "features",
        "MODELS_DIR": outputs_dir / "models",
        "PREDICTIONS_DIR": outputs_dir / "predictions",
        "REPORTS_DIR": outputs_dir / "reports",
    }

    for attr, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, attr, path)

    # Keep legacy tools module in sync for compatibility assertions.
    for attr, path in mapping.items():
        setattr(legacy_tools, attr, path)

    return mapping


class TestDatasetModule:
    """Tests for dataset loading utilities."""

    def test_load_dataset_success(self, sample_train_df, sample_test_df, temp_dataset_dir, monkeypatch, pipeline_paths):
        sample_train_df.to_csv(temp_dataset_dir / "train.csv", index=False)
        sample_test_df.to_csv(temp_dataset_dir / "test.csv", index=False)
        monkeypatch.chdir(temp_dataset_dir.parent.parent)

        result = json.loads(load_dataset())

        assert result["status"] == "success"
        assert (pipeline_paths["DATA_DIR"] / "train_data.pkl").exists()
        assert (pipeline_paths["REPORTS_DIR"] / "data_exploration_report.json").exists()

    def test_load_dataset_missing_files(self, tmp_path, monkeypatch, pipeline_paths):
        monkeypatch.chdir(tmp_path)

        result = json.loads(load_dataset())

        assert result["status"] == "failed"
        assert "error" in result


class TestProblemTypeDetection:
    """Tests for automatic problem type detection."""

    def test_detect_classification(self, pipeline_paths):
        y = np.array([0, 1, 0, 1])
        np.save(pipeline_paths["FEATURES_DIR"] / "y_train.npy", y)

        assert detect_problem_type() == "classification"

    def test_detect_regression(self, pipeline_paths):
        y = np.linspace(0, 1, 20, dtype=float)
        np.save(pipeline_paths["FEATURES_DIR"] / "y_train.npy", y)

        assert detect_problem_type() == "regression"

    def test_detect_problem_type_default(self, pipeline_paths):
        # No y_train.npy present
        assert detect_problem_type() == "classification"


class TestFeatureEngineeringHelpers:
    """Tests for feature engineering helper utilities."""

    def test_apply_ai_generated_features_ratio(self):
        train_raw = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 4.0]})
        test_raw = pd.DataFrame({"a": [15.0, 30.0], "b": [3.0, 5.0]})
        train_features = pd.DataFrame({"baseline": [1.0, 1.0]})
        test_features = pd.DataFrame({"baseline": [1.0, 1.0]})

        suggestions = {
            "status": "success",
            "engineered_features": [
                {
                    "name": "ratio_ab",
                    "operation": "ratio",
                    "inputs": ["a", "b"],
                    "parameters": {},
                    "expected_impact": "medium",
                    "rationale": "Density ratio",
                }
            ],
        }

        updated_train, updated_test, summary = apply_ai_generated_features(
            train_raw=train_raw,
            test_raw=test_raw,
            train_features=train_features,
            test_features=test_features,
            suggestions=suggestions,
        )

        assert "ratio_ab" in updated_train.columns
        assert summary["status"] == "applied"
        np.testing.assert_allclose(updated_train["ratio_ab"].values, np.array([5.0, 5.0]))


class TestAiTuningUtilities:
    """Tests for AI tuning sanitisation and extraction."""

    def test_build_ai_tuning_override_details_clamps_values(self):
        recommendations = [
            {
                "model": "CatBoost",
                "parameter": "depth",
                "suggested_value": 25,  # should clamp to max 10
                "confidence": "high",
            },
            {
                "model": "XGBoost",
                "parameter": "learning_rate",
                "suggested_value": 1e-6,  # should clamp to min 0.01
                "confidence": "low",
            },
        ]

        details = build_ai_tuning_override_details(recommendations)
        overrides = extract_override_values(details)

        assert details["CatBoost"]["depth"]["suggested_value"] == 10
        assert pytest.approx(details["XGBoost"]["learning_rate"]["suggested_value"], rel=1e-6) == 0.01
        assert overrides["CatBoost"]["depth"] == 10
        assert pytest.approx(overrides["XGBoost"]["learning_rate"], rel=1e-6) == 0.01


class TestModelSelectionGuidance:
    """Tests for ordering and filtering models via guidance."""

    def test_apply_model_selection_guidance_orders_models(self):
        models = {
            "Random Forest": object(),
            "CatBoost": object(),
            "XGBoost": object(),
            "Linear Regression": object(),
        }
        guidance = {
            "status": "success",
            "recommended_models": ["CatBoost", "XGBoost"],
            "excluded_models": ["Linear Regression"],
            "confidence": "high",
        }

        ordered, tuning_list, summary = apply_model_selection_guidance(
            models=models,
            guidance=guidance,
            base_tuning_list=["Random Forest", "CatBoost", "XGBoost"],
        )

        assert list(ordered.keys()) == ["CatBoost", "XGBoost", "Random Forest"]
        assert tuning_list == ["CatBoost", "XGBoost"]
        assert summary["applied"] is True


class TestPipelineStagesFailures:
    """Ensure pipeline entry points handle missing artefacts gracefully."""

    def test_engineer_features_missing_data(self, pipeline_paths):
        # No pickled datasets present → expect graceful failure payload.
        result = json.loads(engineer_features())
        assert result["status"] == "failed"

    def test_train_model_pipeline_missing_features(self, pipeline_paths):
        result = json.loads(train_model_pipeline())
        assert result["status"] == "failed"

    def test_generate_predictions_missing_model(self, pipeline_paths):
        np.save(pipeline_paths["FEATURES_DIR"] / "X_test.npy", np.random.randn(3, 2))
        result = json.loads(generate_predictions())
        assert result["status"] == "failed"


class TestLegacyToolsCompatibility:
    """Minimal assurance that src.genML.tools continues to expose the new API."""

    def test_tools_reexports_pipeline_functions(self):
        exposed = {
            "load_dataset",
            "engineer_features",
            "train_model_pipeline",
            "generate_predictions",
            "detect_problem_type",
        }

        for name in exposed:
            assert hasattr(legacy_tools, name), f"tools.{name} should be available after refactor"
