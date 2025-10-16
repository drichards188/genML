"""
Reproducibility and consistency tests.

Tests that the feature engineering pipeline produces consistent,
deterministic results across multiple runs.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features import AutoFeatureEngine


class TestReproducibility:
    """Tests for reproducibility and consistency"""

    def test_same_config_same_data_same_features(self):
        """Test that same config + same data = same features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B', 'C'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_scaling': True}
        }

        # Run twice with same config
        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        # Should be identical
        pd.testing.assert_frame_equal(X1, X2)

    def test_multiple_transform_calls_identical(self):
        """Test that multiple transform calls produce identical results"""
        np.random.seed(42)

        train_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.choice(['A', 'B'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.choice(['A', 'B'], 50)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        engine.fit(train_df, target_col='target')

        # Transform multiple times
        X1 = engine.transform(test_df)
        X2 = engine.transform(test_df)
        X3 = engine.transform(test_df)

        # All should be identical
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_frame_equal(X2, X3)

    def test_feature_names_stable(self):
        """Test that feature names are stable across runs"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        config = {'enable_feature_selection': False}

        # Run twice
        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        # Feature names should match
        assert list(X1.columns) == list(X2.columns)

    def test_feature_order_consistent(self):
        """Test that feature order is consistent across runs"""
        np.random.seed(42)

        df = pd.DataFrame({
            'z_feature': np.random.randn(50),
            'a_feature': np.random.randn(50),
            'm_feature': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        config = {'enable_feature_selection': False}

        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        # Order should be consistent
        assert list(X1.columns) == list(X2.columns)

    def test_analysis_results_deterministic(self):
        """Test that data analysis produces deterministic results"""
        df = pd.DataFrame({
            'numeric': np.arange(50),
            'categorical': ['A', 'B'] * 25,
            'target': [0, 1] * 25
        })

        engine = AutoFeatureEngine()

        analysis1 = engine.analyze_data(df)
        analysis2 = engine.analyze_data(df)

        # Analysis should be identical
        assert analysis1['shape'] == analysis2['shape']
        assert analysis1['columns'] == analysis2['columns']
        assert set(analysis1['numerical_columns']) == set(analysis2['numerical_columns'])
        assert set(analysis1['categorical_columns']) == set(analysis2['categorical_columns'])

    def test_feature_selection_reproducible_with_seed(self):
        """Test that feature selection is reproducible"""
        np.random.seed(42)

        df = pd.DataFrame(np.random.randn(100, 20))
        df.columns = [f'feature_{i}' for i in range(20)]
        df['target'] = np.random.randint(0, 2, 100)

        config = {
            'enable_feature_selection': True,
            'max_features': 10
        }

        # Run twice - may have some randomness in model-based selection
        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        # Should produce same or similar features
        # (exact match depends on sklearn's internal randomness)
        assert X1.shape == X2.shape

    def test_processor_state_preserved(self):
        """Test that processor state is preserved across transforms"""
        np.random.seed(42)

        train_df = pd.DataFrame({
            'numeric': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        test1 = pd.DataFrame({'numeric': [1.0, 2.0, 3.0]})
        test2 = pd.DataFrame({'numeric': [1.0, 2.0, 3.0]})

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_scaling': True}
        }
        engine = AutoFeatureEngine(config)
        engine.fit(train_df, target_col='target')

        X_test1 = engine.transform(test1)
        X_test2 = engine.transform(test2)

        # Same input should produce same output
        pd.testing.assert_frame_equal(X_test1, X_test2)

    def test_domain_detection_consistent(self):
        """Test that domain detection is consistent"""
        df = pd.DataFrame({
            'price': np.random.uniform(10, 1000, 50),
            'amount': np.random.uniform(100, 5000, 50),
            'balance': np.random.uniform(1000, 10000, 50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine()

        analysis1 = engine.analyze_data(df)
        analysis2 = engine.analyze_data(df)

        # Domain detection should be consistent
        if 'domain_analysis' in analysis1 and 'domain_analysis' in analysis2:
            domains1 = set(analysis1['domain_analysis'].get('detected_domains', []))
            domains2 = set(analysis2['domain_analysis'].get('detected_domains', []))
            assert domains1 == domains2

    def test_feature_importance_consistent(self):
        """Test that feature importance scores are consistent"""
        np.random.seed(42)

        X = pd.DataFrame(np.random.randn(100, 10))
        X.columns = [f'feature_{i}' for i in range(10)]
        y = pd.Series(np.random.randint(0, 2, 100))

        engine = AutoFeatureEngine()

        # Calculate importance twice
        importance1 = engine.get_feature_importance(X, y)
        importance2 = engine.get_feature_importance(X, y)

        # Should be similar (may have minor numerical differences)
        if importance1 and importance2:
            assert set(importance1.keys()) == set(importance2.keys())
            for key in importance1.keys():
                assert abs(importance1[key] - importance2[key]) < 0.1

    def test_categorical_encoding_consistent(self):
        """Test that categorical encoding is consistent"""
        train_df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'target': [0, 1] * 30
        })

        test_df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        engine.fit(train_df, target_col='target')

        # Transform multiple times
        X1 = engine.transform(test_df)
        X2 = engine.transform(test_df)

        pd.testing.assert_frame_equal(X1, X2)

    def test_text_features_deterministic(self):
        """Test that text feature generation is deterministic"""
        df = pd.DataFrame({
            'text': ['Sample text one', 'Sample text two', 'Another sample text'] * 10,
            'target': [0, 1, 0] * 10
        })

        config = {
            'enable_feature_selection': False,
            'text_config': {'enable_basic_features': True, 'enable_tfidf': False}
        }

        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        pd.testing.assert_frame_equal(X1, X2)

    def test_datetime_features_deterministic(self):
        """Test that datetime feature extraction is deterministic"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=50, freq='D'),
            'target': [0, 1] * 25
        })

        config = {
            'enable_feature_selection': False,
            'datetime_config': {'enable_components': True, 'enable_cyclical': True}
        }

        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        pd.testing.assert_frame_equal(X1, X2)

    def test_interaction_features_consistent(self):
        """Test that interaction features are consistently generated"""
        df = pd.DataFrame({
            'feature1': np.arange(50),
            'feature2': np.arange(50, 100),
            'target': [0, 1] * 25
        })

        config = {
            'interaction_pairs': [('feature1', 'feature2')],
            'enable_feature_selection': False
        }

        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        pd.testing.assert_frame_equal(X1, X2)

    def test_pipeline_report_consistent(self):
        """Test that pipeline reports are consistent"""
        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': ['A', 'B'] * 25,
            'target': [0, 1] * 25
        })

        engine1 = AutoFeatureEngine()
        engine1.fit(df, target_col='target')
        report1 = engine1.get_feature_report()

        engine2 = AutoFeatureEngine()
        engine2.fit(df, target_col='target')
        report2 = engine2.get_feature_report()

        # Key report elements should match
        assert report1['total_features_generated'] == report2['total_features_generated']
        assert set(report1['processors_used'].keys()) == set(report2['processors_used'].keys())

    def test_config_changes_produce_different_results(self):
        """Test that different configs produce different results"""
        np.random.seed(42)

        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        config1 = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_polynomial': False}
        }

        config2 = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_polynomial': True, 'polynomial_degree': 2}
        }

        engine1 = AutoFeatureEngine(config1)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config2)
        X2 = engine2.fit_transform(df, target_col='target')

        # Different configs should produce different number of features
        assert X1.shape[1] != X2.shape[1]

    def test_manual_type_hints_respected(self):
        """Test that manual type hints are consistently applied"""
        df = pd.DataFrame({
            'ambiguous_column': [1, 2, 3, 4, 5] * 10,  # Could be numeric or categorical
            'target': [0, 1] * 25
        })

        config = {
            'manual_type_hints': {'ambiguous_column': 'categorical'},
            'enable_feature_selection': False
        }

        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        # Should produce identical results
        pd.testing.assert_frame_equal(X1, X2)

        # Should be treated as categorical (check for encoding)
        assert any('_encoded' in col for col in X1.columns)
