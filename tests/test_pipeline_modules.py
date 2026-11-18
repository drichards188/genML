"""
Tests for the refactored pipeline modules.

Covers dataset loading, problem type detection and how stages fail without their inputs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.genML.pipeline import (
    engineer_features,
    generate_predictions,
    load_dataset,
    train_model_pipeline,
)
from src.genML.pipeline import config
from src.genML.pipeline.training import detect_problem_type


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

    return mapping


class TestDatasetModule:
    """Tests for dataset loading utilities."""

    def test_load_dataset_success(self, sample_train_df, sample_test_df, temp_dataset_dir, monkeypatch, pipeline_paths):
        sample_train_df.to_csv(temp_dataset_dir / "train.csv", index=False)
        sample_test_df.to_csv(temp_dataset_dir / "test.csv", index=False)
        monkeypatch.chdir(temp_dataset_dir.parent.parent)

        result = json.loads(load_dataset())

        assert result["status"] == "success"
        assert (pipeline_paths["DATA_DIR"] / "train_data.parquet").exists()
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
