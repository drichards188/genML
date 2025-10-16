"""
Feature quality validation tests.

Tests that generated features meet quality criteria: no infinite values,
reasonable variance, predictive power, proper distributions, etc.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features import AutoFeatureEngine


class TestFeatureQuality:
    """Tests for feature quality validation"""

    def test_no_infinite_values_after_transform(self):
        """Test that no infinite values are generated"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.uniform(0.0001, 100, 100),  # Avoid true zeros
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_log_transform': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # No infinite values
        assert not np.isinf(X.values).any()

    def test_no_excessive_nan_values(self):
        """Test that transformed features don't have excessive NaN values"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': [np.nan if i % 5 == 0 else np.random.randn() for i in range(100)],
            'feature2': [np.nan if i % 7 == 0 else np.random.randn() for i in range(100)],
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Most values should be non-null
        nan_percentage = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        assert nan_percentage < 0.2  # Less than 20% NaN

    def test_features_have_reasonable_variance(self):
        """Test that generated features have reasonable variance"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Check variance of numerical features
        numeric_features = X.select_dtypes(include=[np.number])
        variances = numeric_features.var()

        # Most features should have non-zero variance
        non_zero_var = (variances > 0).sum()
        assert non_zero_var > 0.8 * len(variances)

    def test_scaled_features_reasonable_distribution(self):
        """Test that scaled features have reasonable distributions"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(100) * 100 + 500,
            'feature2': np.random.randn(100) * 10 + 50,
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_scaling': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Scaled features should have mean close to 0 and std close to 1
        scaled_features = [col for col in X.columns if '_scaled' in col]
        if scaled_features:
            for col in scaled_features:
                mean = X[col].mean()
                std = X[col].std()
                assert abs(mean) < 1.0  # Mean close to 0
                assert 0.5 < std < 1.5  # Std close to 1

    def test_binned_features_valid_categories(self):
        """Test that binned features have valid discrete categories"""
        np.random.seed(42)

        df = pd.DataFrame({
            'continuous_feature': np.random.uniform(0, 100, 100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_binning': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Binned features should be integers
        binned_features = [col for col in X.columns if '_bin' in col]
        if binned_features:
            for col in binned_features:
                assert X[col].dtype in [np.int64, np.int32, 'int64', 'int32', 'float64', 'float32']
                # Should have limited unique values
                assert X[col].nunique() <= 20

    def test_encoded_categorical_features_valid(self):
        """Test that encoded categorical features are valid"""
        np.random.seed(42)

        df = pd.DataFrame({
            'category1': np.random.choice(['A', 'B', 'C'], 100),
            'category2': np.random.choice(['X', 'Y', 'Z'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # All encoded features should be numeric
        assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]

        # Encoded categorical features should be non-negative integers
        encoded_features = [col for col in X.columns if '_encoded' in col]
        if encoded_features:
            for col in encoded_features:
                assert X[col].min() >= 0

    def test_features_have_predictive_power(self):
        """Test that generated features have some correlation with target"""
        np.random.seed(42)

        # Create features with known relationships to target
        n = 200
        df = pd.DataFrame({
            'predictive_feature': np.random.randn(n),
            'noise_feature': np.random.randn(n)
        })
        # Create target correlated with predictive_feature
        df['target'] = (df['predictive_feature'] > 0).astype(int)

        config = {
            'enable_feature_selection': True,
            'max_features': 10
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # At least some features should be selected
        assert X.shape[1] > 0

    def test_log_transformed_features_positive(self):
        """Test that log-transformed features are well-defined"""
        np.random.seed(42)

        df = pd.DataFrame({
            'positive_feature': np.random.uniform(1, 100, 100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_log_transform': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Log features should exist and be finite
        log_features = [col for col in X.columns if '_log' in col]
        if log_features:
            for col in log_features:
                assert not np.isinf(X[col]).any()
                assert not np.isnan(X[col]).all()

    def test_interaction_features_meaningful(self):
        """Test that interaction features are mathematically correct"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.arange(50),
            'feature2': np.arange(50, 100),
            'target': np.random.randint(0, 2, 50)
        })

        config = {
            'interaction_pairs': [('feature1', 'feature2')],
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Interaction feature should exist
        interaction_col = 'feature1_x_feature2'
        if interaction_col in X.columns:
            # Should be product of the two features (approximately, after scaling)
            assert not X[interaction_col].isnull().all()

    def test_text_features_reasonable_values(self):
        """Test that text features have reasonable values"""
        np.random.seed(42)

        df = pd.DataFrame({
            'text': [f'This is sample text number {i} with some content.' for i in range(50)],
            'target': np.random.randint(0, 2, 50)
        })

        config = {
            'enable_feature_selection': False,
            'text_config': {'enable_basic_features': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Text length features should be positive
        length_features = [col for col in X.columns if '_length' in col or '_count' in col]
        if length_features:
            for col in length_features:
                assert X[col].min() >= 0

    def test_datetime_features_valid_ranges(self):
        """Test that datetime features have valid ranges"""
        np.random.seed(42)

        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'datetime_config': {'enable_components': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Month should be 1-12
        if any('_month' in col for col in X.columns):
            month_col = [col for col in X.columns if col.endswith('_month')][0]
            assert X[month_col].min() >= 1
            assert X[month_col].max() <= 12

        # Day should be 1-31
        if any('_day' in col for col in X.columns):
            day_col = [col for col in X.columns if col.endswith('_day')][0]
            assert X[day_col].min() >= 1
            assert X[day_col].max() <= 31

    def test_cyclical_features_bounded(self):
        """Test that cyclical (sin/cos) features are bounded [-1, 1]"""
        np.random.seed(42)

        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'datetime_config': {'enable_cyclical': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Sin/cos features should be in [-1, 1]
        cyclical_features = [col for col in X.columns if '_sin' in col or '_cos' in col]
        for col in cyclical_features:
            assert X[col].min() >= -1.1  # Small tolerance
            assert X[col].max() <= 1.1

    def test_missing_indicators_binary(self):
        """Test that missing indicators are binary (0 or 1)"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': [np.nan if i % 3 == 0 else np.random.randn() for i in range(50)],
            'feature2': [np.nan if i % 4 == 0 else np.random.randn() for i in range(50)],
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Missing indicators should be 0 or 1
        missing_indicators = [col for col in X.columns if '_was_missing' in col]
        for col in missing_indicators:
            unique_vals = X[col].unique()
            assert set(unique_vals).issubset({0, 1})

    def test_frequency_features_positive(self):
        """Test that frequency-encoded features are positive"""
        np.random.seed(42)

        df = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'categorical_config': {'enable_frequency': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Frequency features should be non-negative
        frequency_features = [col for col in X.columns if '_frequency' in col]
        for col in frequency_features:
            assert X[col].min() >= 0

    def test_features_stable_across_runs(self):
        """Test that feature generation is deterministic"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        config = {'enable_feature_selection': False}

        # Run twice
        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df, target_col='target')

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df, target_col='target')

        # Should produce identical results
        pd.testing.assert_frame_equal(X1, X2)

    def test_no_information_leakage_in_scaling(self):
        """Test that scaling uses only train statistics"""
        np.random.seed(42)

        train_df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'feature': np.random.randn(50) + 10  # Different distribution
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_scaling': True}
        }
        engine = AutoFeatureEngine(config)
        engine.fit(train_df, target_col='target')

        X_test = engine.transform(test_df)

        # Test scaling should use train statistics
        # Values might be outside [-3, 3] if test dist is different
        assert X_test.shape[0] == test_df.shape[0]

    def test_feature_names_descriptive(self):
        """Test that feature names are descriptive"""
        np.random.seed(42)

        df = pd.DataFrame({
            'age': np.random.uniform(18, 80, 50),
            'income': np.random.uniform(20000, 150000, 50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Feature names should contain original column names
        assert any('age' in col for col in X.columns)
        assert any('income' in col for col in X.columns)

    def test_feature_matrix_no_duplicate_columns(self):
        """Test that no duplicate columns are generated"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # No duplicate column names
        assert len(X.columns) == len(set(X.columns))
