"""
Tests for core ML pipeline tools.

Tests the main pipeline functions: load_dataset, engineer_features,
train_model_pipeline, generate_predictions, and detect_problem_type.
"""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch
from src.genML import tools


class TestLoadDataset:
    """Tests for load_dataset function"""

    def test_load_dataset_success(self, sample_train_df, sample_test_df, temp_dataset_dir, monkeypatch):
        """Test successful dataset loading"""
        # Save test datasets
        sample_train_df.to_csv(temp_dataset_dir / "train.csv", index=False)
        sample_test_df.to_csv(temp_dataset_dir / "test.csv", index=False)

        # Change to temp directory
        monkeypatch.chdir(temp_dataset_dir.parent.parent)

        result_json = tools.load_dataset()
        result = json.loads(result_json)

        assert result['status'] == 'success'
        assert 'train_shape' in result
        assert 'test_shape' in result
        assert result['train_shape'][0] == len(sample_train_df)

    def test_load_dataset_missing_files(self, tmp_path, monkeypatch):
        """Test error handling when files are missing"""
        monkeypatch.chdir(tmp_path)

        result_json = tools.load_dataset()
        result = json.loads(result_json)

        assert result['status'] == 'failed'
        assert 'error' in result

    def test_load_dataset_saves_pickles(self, sample_train_df, sample_test_df, temp_dataset_dir, monkeypatch, tmp_path):
        """Test that datasets are saved as pickle files"""
        # Setup output directory
        data_dir = tmp_path / "outputs" / "data"
        data_dir.mkdir(parents=True)

        # Save test datasets
        sample_train_df.to_csv(temp_dataset_dir / "train.csv", index=False)
        sample_test_df.to_csv(temp_dataset_dir / "test.csv", index=False)

        monkeypatch.chdir(temp_dataset_dir.parent.parent)

        # Temporarily change DATA_DIR
        original_dir = tools.DATA_DIR
        tools.DATA_DIR = data_dir

        try:
            result_json = tools.load_dataset()
            result = json.loads(result_json)

            assert result['status'] == 'success'
            assert (data_dir / 'train_data.pkl').exists()
            assert (data_dir / 'test_data.pkl').exists()
        finally:
            tools.DATA_DIR = original_dir

    def test_load_dataset_invalid_csv(self, temp_dataset_dir, monkeypatch):
        """Test error handling for invalid CSV files"""
        # Create invalid CSV
        (temp_dataset_dir / "train.csv").write_text("invalid,csv,content\n")
        (temp_dataset_dir / "test.csv").write_text("invalid\n")

        monkeypatch.chdir(temp_dataset_dir.parent.parent)

        result_json = tools.load_dataset()
        result = json.loads(result_json)

        # Should handle error gracefully
        assert result['status'] == 'failed'


class TestDetectProblemType:
    """Tests for detect_problem_type function"""

    def test_detect_classification_binary(self, tmp_path):
        """Test classification problem detection for binary target"""
        # Create binary target
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0])

        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        np.save(features_dir / "y_train.npy", y)

        # Temporarily change FEATURES_DIR
        original_dir = tools.FEATURES_DIR
        tools.FEATURES_DIR = features_dir

        try:
            problem_type = tools.detect_problem_type()
            assert problem_type == 'classification'
        finally:
            tools.FEATURES_DIR = original_dir

    def test_detect_classification_multiclass(self, tmp_path):
        """Test classification problem detection for multiclass target"""
        # Create multiclass target
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        np.save(features_dir / "y_train.npy", y)

        original_dir = tools.FEATURES_DIR
        tools.FEATURES_DIR = features_dir

        try:
            problem_type = tools.detect_problem_type()
            assert problem_type == 'classification'
        finally:
            tools.FEATURES_DIR = original_dir

    def test_detect_regression(self, tmp_path):
        """Test regression problem detection"""
        # Create continuous target
        np.random.seed(42)
        y = np.random.randn(100) * 10 + 50

        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        np.save(features_dir / "y_train.npy", y)

        original_dir = tools.FEATURES_DIR
        tools.FEATURES_DIR = features_dir

        try:
            problem_type = tools.detect_problem_type()
            assert problem_type == 'regression'
        finally:
            tools.FEATURES_DIR = original_dir

    def test_detect_problem_type_error_handling(self, tmp_path):
        """Test error handling when y_train.npy is missing"""
        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)

        original_dir = tools.FEATURES_DIR
        tools.FEATURES_DIR = features_dir

        try:
            # Should default to classification
            problem_type = tools.detect_problem_type()
            assert problem_type == 'classification'
        finally:
            tools.FEATURES_DIR = original_dir


class TestEngineerFeatures:
    """Tests for engineer_features function"""

    def test_engineer_features_basic(self, sample_train_df, sample_test_df, tmp_path):
        """Test basic feature engineering"""
        # Setup data directory
        data_dir = tmp_path / "outputs" / "data"
        data_dir.mkdir(parents=True)
        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        reports_dir = tmp_path / "outputs" / "reports"
        reports_dir.mkdir(parents=True)

        # Save datasets
        sample_train_df.to_pickle(data_dir / 'train_data.pkl')
        sample_test_df.to_pickle(data_dir / 'test_data.pkl')

        # Temporarily change directories
        original_data_dir = tools.DATA_DIR
        original_features_dir = tools.FEATURES_DIR
        original_reports_dir = tools.REPORTS_DIR
        tools.DATA_DIR = data_dir
        tools.FEATURES_DIR = features_dir
        tools.REPORTS_DIR = reports_dir

        try:
            result_json = tools.engineer_features()
            result = json.loads(result_json)

            assert result['status'] == 'success'
            assert 'features_used' in result
            assert 'train_shape' in result
            assert 'test_shape' in result
            assert (features_dir / 'X_train.npy').exists()
            assert (features_dir / 'X_test.npy').exists()
            assert (features_dir / 'y_train.npy').exists()
        finally:
            tools.DATA_DIR = original_data_dir
            tools.FEATURES_DIR = original_features_dir
            tools.REPORTS_DIR = original_reports_dir


class TestTrainModelPipeline:
    """Tests for train_model_pipeline function"""

    def test_train_model_classification(self, tmp_path):
        """Test model training for classification"""
        # Create sample features and target
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 2, 50)

        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        models_dir = tmp_path / "outputs" / "models"
        models_dir.mkdir(parents=True)
        reports_dir = tmp_path / "outputs" / "reports"
        reports_dir.mkdir(parents=True)

        np.save(features_dir / 'X_train.npy', X_train)
        np.save(features_dir / 'y_train.npy', y_train)

        original_features_dir = tools.FEATURES_DIR
        original_models_dir = tools.MODELS_DIR
        original_reports_dir = tools.REPORTS_DIR
        tools.FEATURES_DIR = features_dir
        tools.MODELS_DIR = models_dir
        tools.REPORTS_DIR = reports_dir

        try:
            result_json = tools.train_model_pipeline()
            result = json.loads(result_json)

            assert result['status'] == 'success'
            assert result['problem_type'] == 'classification'
            assert 'best_model' in result
            assert 'best_score' in result
            assert len(list(models_dir.glob('best_model_*.pkl'))) > 0
        finally:
            tools.FEATURES_DIR = original_features_dir
            tools.MODELS_DIR = original_models_dir
            tools.REPORTS_DIR = original_reports_dir


class TestCatBoostTuning:
    """CatBoost-specific tests covering GPU-aware tuning behaviour."""

    def test_optimize_catboost_uses_gpu_params(self, monkeypatch):
        catboost = pytest.importorskip("catboost")
        import optuna

        captured_params = {}

        class DummyRegressor:
            def __init__(self, **kwargs):
                captured_params.update(kwargs)

        monkeypatch.setattr(tools.cb, "CatBoostRegressor", DummyRegressor)
        monkeypatch.setattr(tools, "is_catboost_gpu_available", lambda: True)
        monkeypatch.setattr(tools, "cross_val_score", lambda *args, **kwargs: np.array([0.5, 0.6, 0.55]))

        trial = optuna.trial.FixedTrial({
            'cb_iterations': 500,
            'cb_learning_rate': 0.05,
            'cb_depth': 6,
            'cb_l2_leaf_reg': 5.0,
            'cb_border_count': 64,
            'cb_bagging_temperature': 1.5,
            'cb_random_strength': 2.0,
            'cb_min_data_in_leaf': 16,
            'cb_subsample': 0.75
        })

        score = tools.optimize_catboost(
            trial,
            X=np.random.randn(10, 3),
            y=np.random.randn(10),
            problem_type='regression',
            cv=3
        )

        assert np.isclose(score, np.mean([0.5, 0.6, 0.55]))
        assert captured_params['task_type'] == 'GPU'
        assert captured_params['devices'] == '0'
        assert captured_params['loss_function'] == 'RMSE'
        assert captured_params['allow_writing_files'] is False

    def test_optimize_catboost_cpu_fallback(self, monkeypatch):
        pytest.importorskip("catboost")
        import optuna

        captured_params = {}

        class DummyRegressor:
            def __init__(self, **kwargs):
                captured_params.update(kwargs)

        monkeypatch.setattr(tools.cb, "CatBoostRegressor", DummyRegressor)
        monkeypatch.setattr(tools, "is_catboost_gpu_available", lambda: False)
        monkeypatch.setattr(tools, "cross_val_score", lambda *args, **kwargs: np.array([0.42, 0.4]))

        trial = optuna.trial.FixedTrial({
            'cb_iterations': 400,
            'cb_learning_rate': 0.1,
            'cb_depth': 5,
            'cb_l2_leaf_reg': 7.0,
            'cb_border_count': 50,
            'cb_bagging_temperature': 0.5,
            'cb_random_strength': 1.0,
            'cb_min_data_in_leaf': 8,
            'cb_subsample': 0.9
        })

        tools.optimize_catboost(
            trial,
            X=np.random.randn(8, 2),
            y=np.random.randn(8),
            problem_type='regression',
            cv=3
        )

        assert captured_params['task_type'] == 'CPU'
        assert 'devices' not in captured_params

    def test_train_model_regression(self, tmp_path):
        """Test model training for regression"""
        # Create sample features and target for regression
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randn(50) * 10 + 50

        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        models_dir = tmp_path / "outputs" / "models"
        models_dir.mkdir(parents=True)
        reports_dir = tmp_path / "outputs" / "reports"
        reports_dir.mkdir(parents=True)

        np.save(features_dir / 'X_train.npy', X_train)
        np.save(features_dir / 'y_train.npy', y_train)

        original_features_dir = tools.FEATURES_DIR
        original_models_dir = tools.MODELS_DIR
        original_reports_dir = tools.REPORTS_DIR
        tools.FEATURES_DIR = features_dir
        tools.MODELS_DIR = models_dir
        tools.REPORTS_DIR = reports_dir

        try:
            result_json = tools.train_model_pipeline()
            result = json.loads(result_json)

            assert result['status'] == 'success'
            assert result['problem_type'] == 'regression'
            assert 'best_model' in result
        finally:
            tools.FEATURES_DIR = original_features_dir
            tools.MODELS_DIR = original_models_dir
            tools.REPORTS_DIR = original_reports_dir


class TestAiAutomationUtilities:
    """Tests for AI-driven automation helpers."""

    def test_apply_ai_generated_features_creates_ratio(self):
        train_raw = pd.DataFrame({
            'speed': [30.0, 60.0, 90.0],
            'lanes': [1.0, 2.0, 3.0]
        })
        test_raw = pd.DataFrame({
            'speed': [45.0, 50.0],
            'lanes': [1.0, 2.0]
        })
        train_feats = pd.DataFrame({'baseline': [1.0, 1.0, 1.0]})
        test_feats = pd.DataFrame({'baseline': [1.0, 1.0]})

        suggestions = {
            'status': 'success',
            'engineered_features': [
                {
                    'name': 'speed_per_lane',
                    'operation': 'ratio',
                    'inputs': ['speed', 'lanes'],
                    'parameters': {},
                    'expected_impact': 'high',
                    'rationale': 'Speed density per lane'
                }
            ]
        }

        updated_train, updated_test, summary = tools.apply_ai_generated_features(
            train_raw,
            test_raw,
            train_feats,
            test_feats,
            suggestions
        )

        assert 'speed_per_lane' in updated_train.columns
        assert 'speed_per_lane' in updated_test.columns
        assert summary['successful'] == 1
        expected_train = np.array([30.0, 30.0, 30.0], dtype=np.float32)
        np.testing.assert_allclose(updated_train['speed_per_lane'].to_numpy(), expected_train, rtol=1e-5)

    def test_build_ai_tuning_override_details_validates_ranges(self):
        recommendations = [
            {
                'model': 'CatBoost',
                'parameter': 'depth',
                'suggested_value': 20,
                'rationale': 'Underfitting detected for high-error segments',
                'confidence': 'high'
            },
            {
                'model': 'XGBoost',
                'parameter': 'learning_rate',
                'suggested_value': 0.00001,
                'rationale': 'Stabilize gradients on outliers',
                'confidence': 'low'
            }
        ]

        details = tools.build_ai_tuning_override_details(recommendations)
        overrides = tools.extract_override_values(details)

        assert details['CatBoost']['depth']['value'] == 10
        assert pytest.approx(details['XGBoost']['learning_rate']['value'], rel=1e-6) == 0.01
        assert overrides['CatBoost']['depth'] == 10
        assert overrides['XGBoost']['learning_rate'] == pytest.approx(0.01, rel=1e-6)


class TestGeneratePredictions:
    """Tests for generate_predictions function"""

    def test_generate_predictions_basic(self, sample_test_df, tmp_path):
        """Test basic prediction generation"""
        # Create sample data
        np.random.seed(42)
        X_test = np.random.randn(5, 5)

        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        models_dir = tmp_path / "outputs" / "models"
        models_dir.mkdir(parents=True)
        data_dir = tmp_path / "outputs" / "data"
        data_dir.mkdir(parents=True)
        predictions_dir = tmp_path / "outputs" / "predictions"
        predictions_dir.mkdir(parents=True)
        reports_dir = tmp_path / "outputs" / "reports"
        reports_dir.mkdir(parents=True)

        np.save(features_dir / 'X_test.npy', X_test)
        sample_test_df.to_pickle(data_dir / 'test_data.pkl')

        # Train and save a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        import joblib
        joblib.dump(model, models_dir / 'best_model_test.pkl')

        original_features_dir = tools.FEATURES_DIR
        original_models_dir = tools.MODELS_DIR
        original_data_dir = tools.DATA_DIR
        original_predictions_dir = tools.PREDICTIONS_DIR
        original_reports_dir = tools.REPORTS_DIR
        tools.FEATURES_DIR = features_dir
        tools.MODELS_DIR = models_dir
        tools.DATA_DIR = data_dir
        tools.PREDICTIONS_DIR = predictions_dir
        tools.REPORTS_DIR = reports_dir

        try:
            # Change to tmp_path for submission file creation
            import os
            original_cwd = os.getcwd()
            os.chdir(tmp_path)

            result_json = tools.generate_predictions()
            result = json.loads(result_json)

            assert result['status'] == 'success'
            assert 'submission_file' in result
            assert 'predictions_count' in result
            assert result['predictions_count'] == 5

            os.chdir(original_cwd)
        finally:
            tools.FEATURES_DIR = original_features_dir
            tools.MODELS_DIR = original_models_dir
            tools.DATA_DIR = original_data_dir
            tools.PREDICTIONS_DIR = original_predictions_dir
            tools.REPORTS_DIR = original_reports_dir

    def test_generate_predictions_no_model(self, tmp_path):
        """Test error handling when no model exists"""
        features_dir = tmp_path / "outputs" / "features"
        features_dir.mkdir(parents=True)
        models_dir = tmp_path / "outputs" / "models"
        models_dir.mkdir(parents=True)

        np.save(features_dir / 'X_test.npy', np.random.randn(5, 5))

        original_features_dir = tools.FEATURES_DIR
        original_models_dir = tools.MODELS_DIR
        tools.FEATURES_DIR = features_dir
        tools.MODELS_DIR = models_dir

        try:
            result_json = tools.generate_predictions()
            result = json.loads(result_json)

            assert result['status'] == 'failed'
            assert 'error' in result
        finally:
            tools.FEATURES_DIR = original_features_dir
            tools.MODELS_DIR = original_models_dir
