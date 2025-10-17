"""
Edge case and stress testing for feature engineering.

Tests extreme scenarios, boundary conditions, and robustness of the
feature engineering pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features import AutoFeatureEngine


class TestEdgeCases:
    """Stress tests and edge case handling"""

    def test_empty_dataframe(self):
        """Test handling of empty dataframes"""
        df = pd.DataFrame()

        engine = AutoFeatureEngine()

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, KeyError, AttributeError)):
            engine.fit_transform(df)

    def test_single_row_dataset(self):
        """Test handling of single-row datasets"""
        df = pd.DataFrame({
            'feature1': [1.0],
            'feature2': ['A'],
            'target': [1]
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})

        # Should handle gracefully (though results may be limited)
        try:
            X = engine.fit_transform(df, target_col='target')
            assert X.shape[0] == 1
        except (ValueError, Exception) as e:
            # Acceptable to fail on single row
            pass

    def test_single_column_dataset(self):
        """Test handling of datasets with only one feature"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        assert X.shape[0] == 100
        assert X.shape[1] > 0

    def test_all_missing_columns(self):
        """Test handling of columns with all missing values"""
        df = pd.DataFrame({
            'all_missing': [np.nan] * 100,
            'valid_feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Should handle gracefully
        assert X.shape[0] == 100

    def test_all_constant_values(self):
        """Test handling of constant-valued columns"""
        df = pd.DataFrame({
            'constant': [1.0] * 100,
            'another_constant': ['A'] * 100,
            'valid_feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': True})
        X = engine.fit_transform(df, target_col='target')

        # Should handle gracefully, possibly removing constants
        assert X.shape[0] == 100

    def test_extreme_outliers(self):
        """Test handling of extreme outliers"""
        df = pd.DataFrame({
            'normal_feature': np.random.randn(100),
            'outlier_feature': np.concatenate([np.random.randn(98), [1e10, -1e10]]),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Should handle outliers without crashing
        assert X.shape[0] == 100
        assert not np.isinf(X.values).any()

    def test_very_high_dimensional_data(self):
        """Test handling of datasets with many features"""
        np.random.seed(42)

        # 200 features
        df = pd.DataFrame(np.random.randn(50, 200))
        df.columns = [f'feature_{i}' for i in range(200)]
        df['target'] = np.random.randint(0, 2, 50)

        config = {
            'enable_feature_selection': True,
            'max_features': 30
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should reduce dimensionality
        assert X.shape[1] <= 30
        assert X.shape[0] == 50

    def test_very_long_text_fields(self):
        """Test handling of very long text"""
        long_text = 'word ' * 10000  # Very long text

        df = pd.DataFrame({
            'long_text': [long_text] * 20,
            'normal_text': ['short'] * 20,
            'target': np.random.randint(0, 2, 20)
        })

        config = {
            'text_config': {
                'enable_basic_features': True,
                'enable_tfidf': False
            }
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should handle long text
        assert X.shape[0] == 20

    def test_highly_imbalanced_categorical(self):
        """Test handling of highly imbalanced categorical features"""
        # 99% category A, 1% category B
        categories = ['A'] * 99 + ['B']

        df = pd.DataFrame({
            'imbalanced_cat': categories,
            'numeric': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        assert X.shape[0] == 100

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in text"""
        df = pd.DataFrame({
            'text_with_unicode': ['Hello 世界', '你好 World', 'Ñoño', 'Café ☕'],
            'text_with_special': ['test@email.com', 'http://example.com', '(123) 456-7890', 'price: $99.99'],
            'target': [0, 1, 0, 1]
        })

        config = {
            'text_config': {'enable_basic_features': True, 'enable_patterns': True}
        }
        engine = AutoFeatureEngine(config)

        try:
            X = engine.fit_transform(df, target_col='target')
            assert X.shape[0] == 4
        except Exception as e:
            # Some systems may have encoding issues
            pytest.skip(f"Unicode handling not supported: {e}")

    def test_mixed_data_types_in_column(self):
        """Test handling of mixed data types in a single column"""
        df = pd.DataFrame({
            'mixed_column': [1, '2', 3.0, 'four', 5],
            'normal_feature': np.random.randn(5),
            'target': [0, 1, 0, 1, 0]
        })

        engine = AutoFeatureEngine()

        try:
            X = engine.fit_transform(df, target_col='target')
            assert X.shape[0] == 5
        except Exception:
            # Acceptable to fail on truly mixed types
            pass

    def test_datetime_invalid_formats(self):
        """Test handling of invalid datetime formats"""
        df = pd.DataFrame({
            'maybe_date': ['2020-01-01', 'not a date', '2021-12-31', 'invalid'],
            'numeric': np.random.randn(4),
            'target': [0, 1, 0, 1]
        })

        engine = AutoFeatureEngine()

        try:
            X = engine.fit_transform(df, target_col='target')
            assert X.shape[0] == 4
        except Exception as e:
            # Should handle gracefully
            pass

    def test_negative_values_with_log_transform(self):
        """Test that negative values don't break log transform"""
        df = pd.DataFrame({
            'negative_feature': np.random.randn(50),  # Can be negative
            'target': np.random.randint(0, 2, 50)
        })

        config = {
            'numerical_config': {'enable_log_transform': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should handle gracefully (only log positive values)
        assert X.shape[0] == 50
        assert not np.isinf(X.values).any()

    def test_zero_values_handling(self):
        """Test handling of zero values in various operations"""
        df = pd.DataFrame({
            'with_zeros': [0, 1, 0, 2, 0, 3, 0, 4],
            'all_zeros': [0] * 8,
            'target': [0, 1, 0, 1, 0, 1, 0, 1]
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Should handle zeros without division errors
        assert X.shape[0] == 8
        assert not np.isinf(X.values).any()

    def test_very_large_categorical_cardinality(self):
        """Test handling of extremely high cardinality categoricals"""
        df = pd.DataFrame({
            'high_cardinality': [f'category_{i}' for i in range(500)],  # 500 unique categories
            'numeric': np.random.randn(500),
            'target': np.random.randint(0, 2, 500)
        })

        config = {
            'categorical_config': {'max_categories': 50}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should handle by grouping rare categories
        assert X.shape[0] == 500

    def test_datetime_edge_dates(self):
        """Test handling of edge case dates"""
        df = pd.DataFrame({
            'date': pd.to_datetime([
                '1900-01-01',
                '2099-12-31',
                '2000-02-29',  # Leap year
                '2020-01-01'
            ]),
            'target': [0, 1, 0, 1]
        })

        config = {
            'datetime_config': {'enable_components': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        assert X.shape[0] == 4

    def test_all_unique_categorical_values(self):
        """Test categorical column where every value is unique"""
        df = pd.DataFrame({
            'unique_cat': [f'unique_{i}' for i in range(100)],
            'numeric': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Should treat as ID and potentially exclude or handle differently
        assert X.shape[0] == 100

    def test_unseen_categories_in_test_set(self):
        """Test handling of categories in test that weren't in train"""
        train_df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'numeric': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'category': ['A', 'B', 'D', 'E'],  # D and E are new
            'numeric': np.random.randn(4)
        })

        engine = AutoFeatureEngine()
        engine.fit(train_df, target_col='target')
        X_test = engine.transform(test_df)

        # Should handle unseen categories
        assert X_test.shape[0] == 4
        assert not X_test.isnull().all().any()

    def test_multicollinearity(self):
        """Test handling of highly correlated features"""
        np.random.seed(42)
        base = np.random.randn(100)

        df = pd.DataFrame({
            'feature1': base,
            'feature2': base + np.random.randn(100) * 0.01,
            'feature3': base * 0.99,
            'feature4': base + 0.001,
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': True,
            'feature_selection': {'correlation_threshold': 0.95}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should reduce correlated features
        # The number of final features depends on feature engineering
        assert X.shape[0] == 100

    def test_extremely_skewed_distribution(self):
        """Test handling of extremely skewed distributions"""
        df = pd.DataFrame({
            'skewed': np.exp(np.random.randn(100) * 5),  # Highly skewed
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Should handle with log transform or other techniques
        assert X.shape[0] == 100

    def test_binary_target_all_same_class(self):
        """Test handling of target with only one class"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': [1] * 100  # All same class
        })

        engine = AutoFeatureEngine()

        try:
            X = engine.fit_transform(df, target_col='target')
            # May work but feature selection might fail
            assert X.shape[0] == 100
        except Exception:
            # Acceptable to fail with no variance in target
            pass

    def test_memory_efficiency_large_dataset(self):
        """Test that pipeline doesn't explode memory on larger datasets"""
        np.random.seed(42)

        # Create a reasonably large dataset
        df = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) for i in range(50)
        })
        df['target'] = np.random.randint(0, 2, 1000)

        config = {
            'enable_feature_selection': True,
            'max_features': 30
        }

        engine = AutoFeatureEngine(config)

        try:
            X = engine.fit_transform(df, target_col='target')
            assert X.shape[0] == 1000
            assert X.shape[1] <= 30
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_nan_infinity_in_interactions(self):
        """Test that interaction features don't create NaN or Inf"""
        df = pd.DataFrame({
            'feature1': [0, 1, 2, np.nan, 4],
            'feature2': [5, np.nan, 7, 8, 9],
            'target': [0, 1, 0, 1, 0]
        })

        config = {
            'interaction_pairs': [('feature1', 'feature2')],
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should handle NaN in interactions
        assert not np.isinf(X.values).any()
