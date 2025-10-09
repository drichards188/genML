"""
Tests for feature processor modules.

Tests the individual feature processors (Numerical, Categorical, Text, DateTime)
that transform raw data into ML-ready features.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features.feature_processors import (
    NumericalProcessor,
    CategoricalProcessor,
    TextProcessor,
    DateTimeProcessor
)


class TestNumericalProcessor:
    """Tests for NumericalProcessor"""

    def test_basic_numerical_processing(self, numerical_series):
        """Test basic numerical feature generation"""
        processor = NumericalProcessor({
            'enable_scaling': True,
            'enable_binning': True,
            'n_bins': 3
        })
        processor.fit(numerical_series)
        result = processor.transform(numerical_series)

        # Check that features were created
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(numerical_series)
        assert result.shape[1] > 0  # Should have generated features

    def test_missing_value_handling(self):
        """Test that missing values are handled properly"""
        data = pd.Series([1, 2, np.nan, 4, 5], name='test')
        processor = NumericalProcessor({'enable_scaling': True})
        processor.fit(data)
        result = processor.transform(data)

        # No completely null columns after processing
        assert not result.isnull().all().any()

    def test_scaling_enabled(self, numerical_series):
        """Test that scaled features are created when enabled"""
        processor = NumericalProcessor({'enable_scaling': True})
        processor.fit(numerical_series)
        result = processor.transform(numerical_series)

        # Should have at least the original feature
        assert result.shape[1] >= 1

    def test_scaling_disabled(self, numerical_series):
        """Test behavior when scaling is disabled"""
        processor = NumericalProcessor({'enable_scaling': False})
        processor.fit(numerical_series)
        result = processor.transform(numerical_series)

        # Should still return a DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_get_feature_names(self, numerical_series):
        """Test that feature names are returned correctly"""
        processor = NumericalProcessor({})
        processor.fit(numerical_series)
        names = processor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

    def test_fit_transform_consistency(self, numerical_series):
        """Test that fit and transform produce consistent results"""
        processor = NumericalProcessor({'enable_scaling': True})

        # Fit and transform separately
        processor.fit(numerical_series)
        result1 = processor.transform(numerical_series)

        # Should be able to transform multiple times with same result structure
        result2 = processor.transform(numerical_series)

        assert result1.shape == result2.shape
        assert list(result1.columns) == list(result2.columns)


class TestCategoricalProcessor:
    """Tests for CategoricalProcessor"""

    def test_label_encoding(self, categorical_series):
        """Test label encoding of categorical variables"""
        processor = CategoricalProcessor({
            'encoding_method': 'label',
            'enable_frequency': False
        })
        processor.fit(categorical_series)
        result = processor.transform(categorical_series)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(categorical_series)
        assert result.shape[1] >= 1

    def test_one_hot_encoding(self, categorical_series):
        """Test one-hot encoding"""
        processor = CategoricalProcessor({
            'encoding_method': 'onehot',
            'max_categories': 10
        })
        processor.fit(categorical_series)
        result = processor.transform(categorical_series)

        # One-hot should create multiple columns for multiple categories
        assert result.shape[1] >= 1

    def test_frequency_encoding(self, categorical_series):
        """Test frequency encoding feature"""
        processor = CategoricalProcessor({
            'enable_frequency': True
        })
        processor.fit(categorical_series)
        result = processor.transform(categorical_series)

        # Should have at least one feature
        assert result.shape[1] >= 1

    def test_unseen_categories(self, categorical_series):
        """Test handling of unseen categories during transform"""
        processor = CategoricalProcessor({'encoding_method': 'label'})
        processor.fit(categorical_series)

        # New data with unseen category
        new_data = pd.Series(['A', 'B', 'D'], name='test')  # 'D' is unseen
        result = processor.transform(new_data)

        # Should not crash and should handle unseen category
        assert result.shape[0] == len(new_data)
        assert not result.isnull().all().any()

    def test_missing_values_handling(self):
        """Test handling of missing categorical values"""
        data = pd.Series(['A', 'B', np.nan, 'A', 'C'], name='test')
        processor = CategoricalProcessor({'encoding_method': 'label'})
        processor.fit(data)
        result = processor.transform(data)

        # Should handle missing values without crashing
        assert result.shape[0] == len(data)

    def test_get_feature_names(self, categorical_series):
        """Test feature name retrieval"""
        processor = CategoricalProcessor({})
        processor.fit(categorical_series)
        names = processor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0


class TestTextProcessor:
    """Tests for TextProcessor"""

    def test_basic_text_features(self, text_series):
        """Test basic text feature generation"""
        processor = TextProcessor({
            'enable_basic_features': True,
            'enable_tfidf': False
        })
        processor.fit(text_series)
        result = processor.transform(text_series)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(text_series)
        assert result.shape[1] > 0

    def test_text_length_features(self, text_series):
        """Test that text length features are reasonable"""
        processor = TextProcessor({'enable_basic_features': True})
        processor.fit(text_series)
        result = processor.transform(text_series)

        # Should have some features
        assert result.shape[1] > 0

        # All numeric features should be non-negative
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert result[col].min() >= 0

    def test_missing_text_handling(self):
        """Test handling of missing text values"""
        data = pd.Series(['text1', None, 'text3', np.nan, 'text5'], name='test')
        processor = TextProcessor({'enable_basic_features': True})
        processor.fit(data)
        result = processor.transform(data)

        # Should handle missing values without crashing
        assert result.shape[0] == len(data)
        assert not result.isnull().all().any()

    def test_empty_string_handling(self):
        """Test handling of empty strings"""
        data = pd.Series(['text1', '', 'text3', '   ', 'text5'], name='test')
        processor = TextProcessor({'enable_basic_features': True})
        processor.fit(data)
        result = processor.transform(data)

        assert result.shape[0] == len(data)

    def test_get_feature_names(self, text_series):
        """Test feature name retrieval"""
        processor = TextProcessor({'enable_basic_features': True})
        processor.fit(text_series)
        names = processor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0


class TestDateTimeProcessor:
    """Tests for DateTimeProcessor"""

    def test_datetime_feature_extraction(self, datetime_series):
        """Test extraction of datetime features"""
        processor = DateTimeProcessor({
            'extract_year': True,
            'extract_month': True,
            'extract_day': True
        })
        processor.fit(datetime_series)
        result = processor.transform(datetime_series)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(datetime_series)
        assert result.shape[1] > 0

    def test_datetime_components(self, datetime_series):
        """Test that datetime components are extracted correctly"""
        processor = DateTimeProcessor({
            'extract_year': True,
            'extract_month': True,
            'extract_day': True,
            'extract_dayofweek': True
        })
        processor.fit(datetime_series)
        result = processor.transform(datetime_series)

        # Should have multiple features
        assert result.shape[1] > 0

        # All values should be numeric
        assert result.select_dtypes(include=[np.number]).shape[1] == result.shape[1]

    def test_string_datetime_parsing(self):
        """Test parsing of datetime from strings"""
        data = pd.Series(['2020-01-01', '2020-02-01', '2020-03-01'], name='test')
        processor = DateTimeProcessor({'extract_month': True})
        processor.fit(data)
        result = processor.transform(data)

        assert result.shape[0] == len(data)

    def test_get_feature_names(self, datetime_series):
        """Test feature name retrieval"""
        processor = DateTimeProcessor({'extract_year': True, 'extract_month': True})
        processor.fit(datetime_series)
        names = processor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
