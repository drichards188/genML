"""
Tests for AutoFeatureEngine.

Tests the main feature engineering orchestrator that coordinates
data analysis, feature processing, and feature selection.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features import AutoFeatureEngine


class TestAutoFeatureEngine:
    """Tests for AutoFeatureEngine class"""

    def test_initialization(self):
        """Test engine initialization with default config"""
        engine = AutoFeatureEngine()

        assert engine is not None
        assert not engine.is_fitted
        assert engine.config == {}

    def test_initialization_with_config(self):
        """Test engine initialization with custom config"""
        config = {
            'max_features': 100,
            'enable_feature_selection': True
        }
        engine = AutoFeatureEngine(config)

        assert engine.config == config
        assert engine.max_features == 100
        assert engine.enable_feature_selection is True

    def test_analyze_data(self, sample_train_df):
        """Test data analysis functionality"""
        engine = AutoFeatureEngine()
        analysis = engine.analyze_data(sample_train_df)

        assert 'column_types' in analysis
        assert 'data_quality' in analysis
        assert isinstance(analysis['column_types'], dict)
        assert len(analysis['column_types']) > 0

    def test_fit_basic(self, sample_train_df):
        """Test basic fit functionality"""
        engine = AutoFeatureEngine({'enable_feature_selection': False})

        engine.fit(sample_train_df, target_col='Survived')

        assert engine.is_fitted
        assert len(engine.processors) > 0

    def test_fit_transform_basic(self, sample_train_df):
        """Test basic fit and transform"""
        config = {
            'max_features': 50,
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)

        result = engine.fit_transform(sample_train_df, target_col='Survived')

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == sample_train_df.shape[0]
        assert result.shape[1] > 0

    def test_transform_after_fit(self, sample_train_df, sample_test_df):
        """Test transform on new data after fitting"""
        engine = AutoFeatureEngine({'enable_feature_selection': False})

        engine.fit(sample_train_df, target_col='Survived')
        result = engine.transform(sample_test_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == sample_test_df.shape[0]

    def test_transform_consistency(self, sample_train_df, sample_test_df):
        """Test that transform produces consistent features"""
        engine = AutoFeatureEngine({'enable_feature_selection': False})

        engine.fit(sample_train_df, target_col='Survived')
        train_result = engine.transform(sample_train_df)
        test_result = engine.transform(sample_test_df)

        # Should have same columns
        assert list(train_result.columns) == list(test_result.columns)

    def test_feature_selection_reduces_features(self, sample_train_df):
        """Test feature selection reduces feature count"""
        config = {
            'max_features': 10,
            'enable_feature_selection': True,
            'feature_selection': {
                'max_features': 10
            }
        }
        engine = AutoFeatureEngine(config)

        result = engine.fit_transform(sample_train_df, target_col='Survived')

        # Should have fewer features than if we generated all possible
        assert result.shape[1] <= 30  # Reasonable upper bound

    def test_fitted_flag(self, sample_train_df):
        """Test that is_fitted flag works correctly"""
        engine = AutoFeatureEngine()

        assert not engine.is_fitted

        engine.fit(sample_train_df, target_col='Survived')

        assert engine.is_fitted

    def test_transform_before_fit_raises_error(self, sample_train_df):
        """Test that transform before fit raises error"""
        engine = AutoFeatureEngine()

        with pytest.raises(ValueError, match="must be fitted"):
            engine.transform(sample_train_df)

    def test_get_feature_importance(self, sample_features, sample_target_classification):
        """Test feature importance calculation"""
        engine = AutoFeatureEngine()

        importance = engine.get_feature_importance(sample_features, sample_target_classification)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # All importance scores should be non-negative
        assert all(score >= 0 for score in importance.values())

    def test_select_features(self, sample_features, sample_target_classification):
        """Test feature selection"""
        engine = AutoFeatureEngine({'max_features': 5, 'enable_feature_selection': True})

        selected = engine.select_features(sample_features, sample_target_classification, max_features=5)

        assert isinstance(selected, list)
        assert len(selected) <= 5
        assert all(feat in sample_features.columns for feat in selected)

    def test_get_feature_report(self, sample_train_df):
        """Test feature engineering report generation"""
        engine = AutoFeatureEngine()
        engine.fit(sample_train_df, target_col='Survived')

        report = engine.get_feature_report()

        assert isinstance(report, dict)
        assert 'analysis_summary' in report
        assert 'processors_used' in report
        assert 'feature_mapping' in report
        assert 'total_features_generated' in report

    def test_save_report(self, sample_train_df, tmp_path):
        """Test saving feature engineering report to file"""
        engine = AutoFeatureEngine()
        engine.fit(sample_train_df, target_col='Survived')

        report_path = tmp_path / "test_report.json"
        engine.save_report(str(report_path))

        assert report_path.exists()

        # Verify it's valid JSON
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        assert isinstance(report, dict)

    def test_target_column_detection(self, sample_train_df):
        """Test automatic target column detection"""
        engine = AutoFeatureEngine()

        # Analyze first to populate target candidates
        engine.analyze_data(sample_train_df)

        # Fit without specifying target
        engine.fit(sample_train_df, target_col=None)

        # Should still fit successfully
        assert engine.is_fitted

    def test_exclude_id_columns(self, sample_train_df):
        """Test that ID columns are excluded from feature engineering"""
        engine = AutoFeatureEngine({'exclude_id_columns': True})

        engine.fit(sample_train_df, target_col='Survived')

        # PassengerId should not be in processors
        assert 'PassengerId' not in engine.processors

    def test_handle_all_missing_column(self):
        """Test handling of columns with all missing values"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 1],
            'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'feature': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})

        # Should handle gracefully
        result = engine.fit_transform(df, target_col='target')

        assert isinstance(result, pd.DataFrame)

    def test_multiple_transform_calls(self, sample_train_df, sample_test_df):
        """Test that multiple transform calls work correctly"""
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        engine.fit(sample_train_df, target_col='Survived')

        result1 = engine.transform(sample_test_df)
        result2 = engine.transform(sample_test_df)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
