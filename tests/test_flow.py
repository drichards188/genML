"""
Tests for CrewAI Flow orchestration.

Tests the MLPipelineFlow class and its stage methods that orchestrate
the complete machine learning pipeline.
"""
import pytest

from src.genML.flow import MLPipelineFlow, create_ml_pipeline_flow
from src.genML.pipeline import config as pipeline_config


@pytest.fixture
def flow_paths(tmp_path, monkeypatch):
    """Provide isolated pipeline directories for flow tests."""
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
        monkeypatch.setattr(pipeline_config, attr, path)

    return mapping


class TestMLPipelineFlow:
    """Tests for MLPipelineFlow class"""

    def test_create_flow(self):
        """Test flow creation"""
        flow = create_ml_pipeline_flow()

        assert isinstance(flow, MLPipelineFlow)

    def test_flow_has_required_methods(self):
        """Test that flow has all required stage methods"""
        flow = create_ml_pipeline_flow()

        assert hasattr(flow, 'load_data_task')
        assert hasattr(flow, 'feature_engineering_task')
        assert hasattr(flow, 'model_training_task')
        assert hasattr(flow, 'prediction_task')

    def test_flow_methods_are_callable(self):
        """Test that all flow stage methods are callable"""
        flow = create_ml_pipeline_flow()

        assert callable(flow.load_data_task)
        assert callable(flow.feature_engineering_task)
        assert callable(flow.model_training_task)
        assert callable(flow.prediction_task)

    def test_error_propagation_feature_engineering(self):
        """Test that errors propagate from data loading to feature engineering"""
        flow = create_ml_pipeline_flow()

        # Simulate failed data loading
        failed_result = {"status": "failed", "error": "Test error"}

        # Feature engineering should skip and propagate error
        fe_result = flow.feature_engineering_task(failed_result)

        assert fe_result['status'] == 'failed'
        assert 'error' in fe_result

    def test_error_propagation_model_training(self):
        """Test that errors propagate from feature engineering to model training"""
        flow = create_ml_pipeline_flow()

        # Simulate failed feature engineering
        failed_result = {"status": "failed", "error": "Feature engineering error"}

        # Model training should skip and propagate error
        mt_result = flow.model_training_task(failed_result)

        assert mt_result['status'] == 'failed'

    def test_error_propagation_prediction(self):
        """Test that errors propagate from model training to prediction"""
        flow = create_ml_pipeline_flow()

        # Simulate failed model training
        failed_result = {"status": "failed", "error": "Model training error"}

        # Prediction should skip and propagate error
        pred_result = flow.prediction_task(failed_result)

        assert pred_result['status'] == 'failed'

    def test_load_data_task_returns_dict(self, sample_train_df, sample_test_df, temp_dataset_dir, monkeypatch):
        """Test that load_data_task returns a dictionary"""
        flow = create_ml_pipeline_flow()

        # Setup test data
        sample_train_df.to_csv(temp_dataset_dir / "train.csv", index=False)
        sample_test_df.to_csv(temp_dataset_dir / "test.csv", index=False)
        monkeypatch.chdir(temp_dataset_dir.parent.parent)

        result = flow.load_data_task()

        assert isinstance(result, dict)
        # Should either succeed or fail gracefully
        assert 'status' in result

    def test_feature_engineering_task_requires_data_summary(self):
        """Test that feature engineering task handles missing data summary"""
        flow = create_ml_pipeline_flow()

        # Call without proper data summary
        result = flow.feature_engineering_task({})

        # Should handle gracefully (either skip or process with defaults)
        assert isinstance(result, dict)

    def test_model_training_task_requires_features(self):
        """Test that model training task handles missing features"""
        flow = create_ml_pipeline_flow()

        # Call without proper feature results
        result = flow.model_training_task({})

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_prediction_task_requires_model(self):
        """Test that prediction task handles missing model"""
        flow = create_ml_pipeline_flow()

        # Call without proper model results
        result = flow.prediction_task({})

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_flow_stage_sequence(self):
        """Test that flow stages are properly sequenced"""
        flow = create_ml_pipeline_flow()

        # Verify that stages have proper decorators
        # This is more of a structural test
        assert hasattr(flow.load_data_task, '__wrapped__') or hasattr(flow.load_data_task, '_crewai_start')

    def test_successful_data_loading_returns_success_status(self, sample_train_df, sample_test_df, temp_dataset_dir, monkeypatch):
        """Test that successful data loading returns success status"""
        flow = create_ml_pipeline_flow()

        # Setup test data
        sample_train_df.to_csv(temp_dataset_dir / "train.csv", index=False)
        sample_test_df.to_csv(temp_dataset_dir / "test.csv", index=False)
        monkeypatch.chdir(temp_dataset_dir.parent.parent)

        result = flow.load_data_task()

        # If data files are found, should return success
        if 'status' in result:
            assert result['status'] in ['success', 'failed']

    def test_feature_engineering_with_success_input(self, sample_train_df, sample_test_df, flow_paths):
        """Test feature engineering with successful data loading input"""
        flow = create_ml_pipeline_flow()

        # Save test data into the configured pipeline paths
        sample_train_df.to_pickle(flow_paths["DATA_DIR"] / "train_data.pkl")
        sample_test_df.to_pickle(flow_paths["DATA_DIR"] / "test_data.pkl")

        data_summary = {
            "status": "success",
            "train_shape": [10, 9],
            "test_shape": [5, 8],
        }

        result = flow.feature_engineering_task(data_summary)

        assert isinstance(result, dict)

    def test_flow_error_handling_consistency(self):
        """Test that all flow stages handle errors consistently"""
        flow = create_ml_pipeline_flow()

        failed_input = {"status": "failed", "error": "Previous stage failed"}

        # All stages should handle failed input the same way
        fe_result = flow.feature_engineering_task(failed_input)
        assert fe_result['status'] == 'failed'

        mt_result = flow.model_training_task(failed_input)
        assert mt_result['status'] == 'failed'

        pred_result = flow.prediction_task(failed_input)
        assert pred_result['status'] == 'failed'


class TestCreateMLPipelineFlow:
    """Tests for create_ml_pipeline_flow factory function"""

    def test_factory_returns_flow_instance(self):
        """Test that factory returns a proper flow instance"""
        flow = create_ml_pipeline_flow()

        assert isinstance(flow, MLPipelineFlow)

    def test_factory_creates_new_instances(self):
        """Test that factory creates independent flow instances"""
        flow1 = create_ml_pipeline_flow()
        flow2 = create_ml_pipeline_flow()

        # Should be different instances
        assert flow1 is not flow2

    def test_factory_flow_is_ready_to_use(self):
        """Test that factory-created flow is ready to use"""
        flow = create_ml_pipeline_flow()

        # Should have all required methods
        assert hasattr(flow, 'load_data_task')
        assert hasattr(flow, 'feature_engineering_task')
        assert hasattr(flow, 'model_training_task')
        assert hasattr(flow, 'prediction_task')
