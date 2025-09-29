"""
Automatic Data Type Detection and Analysis

This module provides intelligent data type detection that goes beyond pandas' basic
dtype inference. It analyzes column patterns, cardinality, and statistical properties
to classify columns into meaningful categories for feature engineering.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTypeAnalyzer:
    """
    Analyzes datasets to automatically detect column types and characteristics.

    This class provides intelligent classification of columns into categories
    that are meaningful for feature engineering, going beyond basic pandas dtypes.
    """

    def __init__(self, cardinality_threshold: float = 0.05, text_uniqueness_threshold: float = 0.8):
        """
        Initialize the data type analyzer.

        Args:
            cardinality_threshold: Ratio threshold for categorical vs high-cardinality
            text_uniqueness_threshold: Uniqueness threshold for text detection
        """
        self.cardinality_threshold = cardinality_threshold
        self.text_uniqueness_threshold = text_uniqueness_threshold

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of dataset structure and column types.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing analysis results and column classifications
        """
        logger.info(f"Analyzing dataset with shape {df.shape}")

        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'column_types': {},
            'categorical_columns': [],
            'numerical_columns': [],
            'text_columns': [],
            'datetime_columns': [],
            'id_columns': [],
            'target_candidates': [],
            'recommendations': {}
        }

        # Analyze each column
        for col in df.columns:
            col_analysis = self._analyze_column(df[col], col)
            analysis['column_types'][col] = col_analysis

            # Categorize columns by type
            col_type = col_analysis['detected_type']
            if col_type == 'categorical':
                analysis['categorical_columns'].append(col)
            elif col_type == 'numerical':
                analysis['numerical_columns'].append(col)
            elif col_type == 'text':
                analysis['text_columns'].append(col)
            elif col_type == 'datetime':
                analysis['datetime_columns'].append(col)
            elif col_type == 'id':
                analysis['id_columns'].append(col)

        # Detect potential target columns
        analysis['target_candidates'] = self._identify_target_candidates(df, analysis)

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        logger.info(f"Analysis complete: {len(analysis['numerical_columns'])} numerical, "
                   f"{len(analysis['categorical_columns'])} categorical, "
                   f"{len(analysis['text_columns'])} text, "
                   f"{len(analysis['datetime_columns'])} datetime columns")

        return analysis

    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Analyze a single column to determine its type and characteristics.

        Args:
            series: Pandas series to analyze
            col_name: Name of the column

        Returns:
            Dictionary with column analysis results
        """
        analysis = {
            'column_name': col_name,
            'pandas_dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series),
            'unique_count': series.nunique(),
            'unique_percentage': series.nunique() / len(series),
            'detected_type': 'unknown',
            'confidence': 0.0,
            'patterns': [],
            'recommendations': []
        }

        # Remove null values for analysis
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            analysis['detected_type'] = 'empty'
            analysis['confidence'] = 1.0
            return analysis

        # Check for ID patterns first
        if self._is_id_column(non_null_series, col_name):
            analysis['detected_type'] = 'id'
            analysis['confidence'] = 0.9
            analysis['patterns'].append('sequential_id' if self._is_sequential(non_null_series) else 'unique_id')
            return analysis

        # Check for datetime patterns
        datetime_result = self._check_datetime(non_null_series)
        if datetime_result['is_datetime']:
            analysis['detected_type'] = 'datetime'
            analysis['confidence'] = datetime_result['confidence']
            analysis['patterns'] = datetime_result['patterns']
            return analysis

        # Check for text patterns
        text_result = self._check_text_patterns(non_null_series, col_name)
        if text_result['is_text']:
            analysis['detected_type'] = 'text'
            analysis['confidence'] = text_result['confidence']
            analysis['patterns'] = text_result['patterns']
            return analysis

        # Check for categorical vs numerical
        if pd.api.types.is_numeric_dtype(series):
            if self._is_categorical_numeric(non_null_series):
                analysis['detected_type'] = 'categorical'
                analysis['confidence'] = 0.8
                analysis['patterns'].append('low_cardinality_numeric')
            else:
                analysis['detected_type'] = 'numerical'
                analysis['confidence'] = 0.9
                analysis['patterns'].append('continuous_numeric')
        else:
            # Non-numeric data
            if analysis['unique_percentage'] < self.cardinality_threshold:
                analysis['detected_type'] = 'categorical'
                analysis['confidence'] = 0.8
                analysis['patterns'].append('low_cardinality_categorical')
            else:
                analysis['detected_type'] = 'text'
                analysis['confidence'] = 0.7
                analysis['patterns'].append('high_cardinality_categorical')

        return analysis

    def _is_id_column(self, series: pd.Series, col_name: str) -> bool:
        """Check if column is likely an ID column."""
        # Check column name patterns
        id_name_patterns = ['id', 'key', 'index', 'identifier', 'pk', 'uuid']
        if any(pattern in col_name.lower() for pattern in id_name_patterns):
            return True

        # Check if values are unique and sequential or unique identifiers
        if series.nunique() == len(series):  # All unique values
            if self._is_sequential(series) or self._has_id_patterns(series):
                return True

        return False

    def _is_sequential(self, series: pd.Series) -> bool:
        """Check if series contains sequential values."""
        if pd.api.types.is_numeric_dtype(series):
            sorted_series = series.sort_values()
            diff = sorted_series.diff().dropna()
            return (diff == 1).all() or (diff.std() < 0.1)
        return False

    def _has_id_patterns(self, series: pd.Series) -> bool:
        """Check for common ID patterns in string data."""
        if series.dtype == 'object':
            sample = series.head(10).astype(str)
            # Check for patterns like UUID, alphanumeric IDs, etc.
            patterns = [
                r'^[A-Za-z0-9]{8,}$',  # Alphanumeric IDs
                r'^[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}$',  # UUID
                r'^[A-Z]{2,3}\d{4,}$',  # Code patterns like AB1234
            ]
            return any(sample.str.match(pattern).any() for pattern in patterns)
        return False

    def _check_datetime(self, series: pd.Series) -> Dict[str, Any]:
        """Check if series contains datetime data."""
        result = {'is_datetime': False, 'confidence': 0.0, 'patterns': []}

        # Try to convert to datetime
        try:
            sample_size = min(100, len(series))
            sample = series.head(sample_size)

            converted = pd.to_datetime(sample, errors='coerce')
            success_rate = converted.notna().sum() / len(sample)

            if success_rate > 0.8:
                result['is_datetime'] = True
                result['confidence'] = success_rate

                # Identify datetime patterns
                if series.dtype == 'object':
                    sample_str = sample.astype(str)
                    if sample_str.str.match(r'\d{4}-\d{2}-\d{2}').any():
                        result['patterns'].append('iso_date')
                    if sample_str.str.match(r'\d{2}/\d{2}/\d{4}').any():
                        result['patterns'].append('us_date')
                    if sample_str.str.contains(r'\d{2}:\d{2}').any():
                        result['patterns'].append('time_component')
                else:
                    result['patterns'].append('numeric_timestamp')

        except Exception:
            pass

        return result

    def _check_text_patterns(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Check if series contains text data."""
        result = {'is_text': False, 'confidence': 0.0, 'patterns': []}

        if series.dtype != 'object':
            return result

        # Check uniqueness
        uniqueness = series.nunique() / len(series)

        # Sample for text analysis
        sample = series.head(50).astype(str)

        # Check for text patterns
        has_spaces = sample.str.contains(' ').any()
        has_multiple_words = sample.str.split().str.len().mean() > 1
        avg_length = sample.str.len().mean()

        # Text indicators
        text_indicators = 0

        if uniqueness > self.text_uniqueness_threshold:
            text_indicators += 1
            result['patterns'].append('high_uniqueness')

        if has_spaces and has_multiple_words:
            text_indicators += 2
            result['patterns'].append('multi_word')

        if avg_length > 20:
            text_indicators += 1
            result['patterns'].append('long_text')

        # Check for name patterns
        if 'name' in col_name.lower():
            text_indicators += 1
            result['patterns'].append('name_column')

        # Determine if it's text
        if text_indicators >= 2:
            result['is_text'] = True
            result['confidence'] = min(0.9, text_indicators * 0.3)

        return result

    def _is_categorical_numeric(self, series: pd.Series) -> bool:
        """Check if numeric series should be treated as categorical."""
        unique_count = series.nunique()
        total_count = len(series)

        # Low cardinality suggests categorical
        if unique_count <= 10:
            return True

        # Check cardinality ratio
        if unique_count / total_count < self.cardinality_threshold:
            return True

        # Check for integer values that might be codes
        if series.dtype in ['int64', 'int32'] and unique_count < 50:
            return True

        return False

    def _identify_target_candidates(self, df: pd.DataFrame, analysis: Dict) -> List[str]:
        """Identify potential target columns."""
        candidates = []

        # Common target column names
        target_names = ['target', 'label', 'y', 'survived', 'class', 'category', 'outcome']

        for col in df.columns:
            col_lower = col.lower()

            # Check name patterns
            if any(name in col_lower for name in target_names):
                candidates.append(col)
                continue

            # Check if it's a binary/categorical variable suitable as target
            col_type = analysis['column_types'][col]['detected_type']
            if col_type == 'categorical':
                unique_count = df[col].nunique()
                if 2 <= unique_count <= 10:  # Good range for classification targets
                    candidates.append(col)

        return candidates

    def _generate_recommendations(self, analysis: Dict) -> Dict[str, List[str]]:
        """Generate feature engineering recommendations based on analysis."""
        recommendations = {
            'preprocessing': [],
            'feature_engineering': [],
            'feature_selection': [],
            'warnings': []
        }

        # Preprocessing recommendations
        for col, col_analysis in analysis['column_types'].items():
            if col_analysis['null_percentage'] > 0.1:
                recommendations['preprocessing'].append(
                    f"Handle missing values in '{col}' ({col_analysis['null_percentage']:.1%} missing)"
                )

        # Feature engineering recommendations
        if len(analysis['text_columns']) > 0:
            recommendations['feature_engineering'].append(
                "Consider text feature extraction (TF-IDF, word counts) for text columns"
            )

        if len(analysis['datetime_columns']) > 0:
            recommendations['feature_engineering'].append(
                "Extract datetime components (year, month, day, hour) from datetime columns"
            )

        if len(analysis['numerical_columns']) > 3:
            recommendations['feature_engineering'].append(
                "Consider polynomial features or interactions between numerical columns"
            )

        # Feature selection recommendations
        total_features = len(analysis['columns'])
        if total_features > 50:
            recommendations['feature_selection'].append(
                f"Dataset has {total_features} features - consider feature selection techniques"
            )

        # Warnings
        if len(analysis['id_columns']) > 0:
            recommendations['warnings'].append(
                f"ID columns detected: {analysis['id_columns']} - consider excluding from modeling"
            )

        return recommendations