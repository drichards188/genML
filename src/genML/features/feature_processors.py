"""
Modular Feature Processors for Different Data Types

This module provides specialized feature processors for different data types.
Each processor implements domain-specific feature engineering techniques that
are commonly effective for that data type. Uses GPU-accelerated preprocessing when available.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from abc import ABC, abstractmethod
import logging
import sys

# Import GPU-aware StandardScaler
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu_utils import get_standard_scaler

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for feature processors.

    All feature processors should inherit from this class and implement
    the required methods for consistent interface.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize processor with configuration.

        Args:
            config: Configuration dictionary for processor-specific settings
        """
        self.config = config or {}
        self.is_fitted = False
        self.feature_names = []

    @abstractmethod
    def fit(self, data: pd.Series) -> 'BaseProcessor':
        """
        Fit the processor on training data.

        Args:
            data: Training data series

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform data using fitted processor.

        Args:
            data: Data series to transform

        Returns:
            DataFrame with engineered features
        """
        pass

    def fit_transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Fit processor and transform data in one step.

        Args:
            data: Data series to fit and transform

        Returns:
            DataFrame with engineered features
        """
        return self.fit(data).transform(data)

    def get_feature_names(self) -> List[str]:
        """Get names of generated features."""
        return self.feature_names


class NumericalProcessor(BaseProcessor):
    """
    Feature processor for numerical data.

    Handles scaling, binning, polynomial features, and statistical transformations
    for numerical columns.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        StandardScalerClass = get_standard_scaler()
        self.scaler = StandardScalerClass()
        self.bin_edges = {}
        self.base_name = ""

        # Configuration
        self.enable_scaling = config.get('enable_scaling', True)
        self.enable_binning = config.get('enable_binning', True)
        self.enable_polynomial = config.get('enable_polynomial', False)
        self.enable_log_transform = config.get('enable_log_transform', True)
        self.n_bins = config.get('n_bins', 5)
        self.polynomial_degree = config.get('polynomial_degree', 2)

    def fit(self, data: pd.Series) -> 'NumericalProcessor':
        """
        Fit numerical processor on training data.

        Args:
            data: Numerical series to fit on

        Returns:
            self for method chaining
        """
        self.base_name = data.name or 'feature'
        clean_data = data.dropna()

        if len(clean_data) == 0:
            logger.warning(f"No valid data for numerical feature {self.base_name}")
            self.is_fitted = True
            return self

        # Fit scaler
        if self.enable_scaling:
            self.scaler.fit(clean_data.values.reshape(-1, 1))

        # Calculate bin edges for binning
        if self.enable_binning:
            try:
                _, self.bin_edges[self.base_name] = pd.cut(clean_data, bins=self.n_bins, retbins=True)
            except Exception as e:
                logger.warning(f"Could not create bins for {self.base_name}: {e}")
                self.enable_binning = False

        # Generate feature names
        self._generate_feature_names()

        self.is_fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform numerical data into engineered features.

        Args:
            data: Numerical series to transform

        Returns:
            DataFrame with engineered numerical features
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")

        result = pd.DataFrame()
        base_name = self.base_name

        # Handle missing values
        data_filled = data.fillna(data.median())

        # Original feature (scaled)
        if self.enable_scaling:
            scaled_data = self.scaler.transform(data_filled.values.reshape(-1, 1)).flatten()
            result[f"{base_name}_scaled"] = scaled_data
        else:
            result[f"{base_name}_original"] = data_filled

        # Binned version
        if self.enable_binning and base_name in self.bin_edges:
            try:
                binned = pd.cut(data_filled, bins=self.bin_edges[base_name], labels=False, include_lowest=True)
                result[f"{base_name}_bin"] = binned.fillna(0).astype(int)
            except Exception as e:
                logger.warning(f"Could not apply binning to {base_name}: {e}")

        # Log transformation (for positive values)
        if self.enable_log_transform and (data_filled > 0).all():
            result[f"{base_name}_log"] = np.log1p(data_filled)

        # Polynomial features
        if self.enable_polynomial:
            result[f"{base_name}_squared"] = data_filled ** 2
            if self.polynomial_degree >= 3:
                result[f"{base_name}_cubed"] = data_filled ** 3

        # Statistical indicators
        median_val = data.median()
        result[f"{base_name}_above_median"] = (data_filled > median_val).astype(int)

        # Missing value indicator
        if data.isnull().any():
            result[f"{base_name}_was_missing"] = data.isnull().astype(int)

        return result

    def _generate_feature_names(self):
        """Generate feature names based on enabled transformations."""
        base_name = self.base_name
        names = []

        if self.enable_scaling:
            names.append(f"{base_name}_scaled")
        else:
            names.append(f"{base_name}_original")

        if self.enable_binning:
            names.append(f"{base_name}_bin")

        if self.enable_log_transform:
            names.append(f"{base_name}_log")

        if self.enable_polynomial:
            names.append(f"{base_name}_squared")
            if self.polynomial_degree >= 3:
                names.append(f"{base_name}_cubed")

        names.append(f"{base_name}_above_median")
        names.append(f"{base_name}_was_missing")

        self.feature_names = names


class CategoricalProcessor(BaseProcessor):
    """
    Feature processor for categorical data.

    Handles encoding, frequency features, and interaction features for
    categorical columns.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.label_encoder = LabelEncoder()
        self.value_counts = {}
        self.base_name = ""

        # Configuration
        self.encoding_method = config.get('encoding_method', 'label')  # 'label' or 'onehot'
        self.enable_frequency = config.get('enable_frequency', True)
        self.enable_rarity = config.get('enable_rarity', True)
        self.min_frequency = config.get('min_frequency', 2)
        self.max_categories = config.get('max_categories', 50)

    def fit(self, data: pd.Series) -> 'CategoricalProcessor':
        """
        Fit categorical processor on training data.

        Args:
            data: Categorical series to fit on

        Returns:
            self for method chaining
        """
        self.base_name = data.name or 'feature'
        clean_data = data.dropna().astype(str)

        if len(clean_data) == 0:
            logger.warning(f"No valid data for categorical feature {self.base_name}")
            self.is_fitted = True
            return self

        # Calculate value counts
        self.value_counts = clean_data.value_counts().to_dict()

        # Handle high cardinality by grouping rare categories
        if len(self.value_counts) > self.max_categories:
            logger.info(f"High cardinality detected for {self.base_name}: {len(self.value_counts)} categories")
            self._handle_high_cardinality(clean_data)

        # Fit label encoder
        self.label_encoder.fit(clean_data)

        # Generate feature names
        self._generate_feature_names()

        self.is_fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform categorical data into engineered features.

        Args:
            data: Categorical series to transform

        Returns:
            DataFrame with engineered categorical features
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")

        result = pd.DataFrame()
        base_name = self.base_name

        # Handle missing values and convert to string
        data_filled = data.fillna('MISSING').astype(str)

        # Handle unseen categories
        data_cleaned = data_filled.apply(lambda x: x if x in self.label_encoder.classes_ else 'UNKNOWN')

        # Label encoding
        try:
            # Add UNKNOWN category if needed
            if 'UNKNOWN' in data_cleaned.values and 'UNKNOWN' not in self.label_encoder.classes_:
                # Create new encoder with UNKNOWN category
                all_categories = list(self.label_encoder.classes_) + ['UNKNOWN']
                new_encoder = LabelEncoder()
                new_encoder.fit(all_categories)
                encoded = new_encoder.transform(data_cleaned)
            else:
                encoded = self.label_encoder.transform(data_cleaned)

            result[f"{base_name}_encoded"] = encoded
        except Exception as e:
            logger.warning(f"Could not encode {base_name}: {e}")
            result[f"{base_name}_encoded"] = 0

        # Frequency encoding
        if self.enable_frequency:
            frequencies = data_filled.map(self.value_counts).fillna(0)
            result[f"{base_name}_frequency"] = frequencies

        # Rarity indicator
        if self.enable_rarity:
            is_rare = data_filled.map(lambda x: self.value_counts.get(x, 0) < self.min_frequency)
            result[f"{base_name}_is_rare"] = is_rare.astype(int)

        # Missing indicator
        if data.isnull().any():
            result[f"{base_name}_was_missing"] = data.isnull().astype(int)

        return result

    def _handle_high_cardinality(self, data: pd.Series):
        """Handle high cardinality categorical features."""
        # Keep only top categories, group others as 'OTHER'
        top_categories = data.value_counts().head(self.max_categories - 1).index.tolist()

        # Update value counts to include 'OTHER' category
        other_count = data.value_counts().tail(len(data.value_counts()) - len(top_categories)).sum()
        if other_count > 0:
            top_categories.append('OTHER')
            new_value_counts = data.value_counts().head(self.max_categories - 1).to_dict()
            new_value_counts['OTHER'] = other_count
            self.value_counts = new_value_counts

    def _generate_feature_names(self):
        """Generate feature names based on enabled transformations."""
        base_name = self.base_name
        names = [f"{base_name}_encoded"]

        if self.enable_frequency:
            names.append(f"{base_name}_frequency")

        if self.enable_rarity:
            names.append(f"{base_name}_is_rare")

        names.append(f"{base_name}_was_missing")

        self.feature_names = names


class TextProcessor(BaseProcessor):
    """
    Feature processor for text data.

    Handles text length features, word counts, TF-IDF, and pattern extraction
    for text columns.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.tfidf = None
        self.base_name = ""

        # Configuration
        self.enable_basic_features = config.get('enable_basic_features', True)
        self.enable_tfidf = config.get('enable_tfidf', False)  # Expensive, disabled by default
        self.enable_patterns = config.get('enable_patterns', True)
        self.max_tfidf_features = config.get('max_tfidf_features', 100)
        self.tfidf_ngram_range = config.get('tfidf_ngram_range', (1, 2))

    def fit(self, data: pd.Series) -> 'TextProcessor':
        """
        Fit text processor on training data.

        Args:
            data: Text series to fit on

        Returns:
            self for method chaining
        """
        self.base_name = data.name or 'feature'
        clean_data = data.dropna().astype(str)

        if len(clean_data) == 0:
            logger.warning(f"No valid data for text feature {self.base_name}")
            self.is_fitted = True
            return self

        # Fit TF-IDF if enabled
        if self.enable_tfidf:
            self.tfidf = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                ngram_range=self.tfidf_ngram_range,
                stop_words='english'
            )
            try:
                self.tfidf.fit(clean_data)
            except Exception as e:
                logger.warning(f"Could not fit TF-IDF for {self.base_name}: {e}")
                self.enable_tfidf = False

        # Generate feature names
        self._generate_feature_names()

        self.is_fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform text data into engineered features.

        Args:
            data: Text series to transform

        Returns:
            DataFrame with engineered text features
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")

        result = pd.DataFrame()
        base_name = self.base_name

        # Handle missing values
        data_filled = data.fillna('').astype(str)

        # Basic text features
        if self.enable_basic_features:
            result[f"{base_name}_length"] = data_filled.str.len()
            result[f"{base_name}_word_count"] = data_filled.str.split().str.len()
            result[f"{base_name}_char_count"] = data_filled.str.len()
            result[f"{base_name}_uppercase_ratio"] = data_filled.str.count(r'[A-Z]') / data_filled.str.len().replace(0, 1)
            result[f"{base_name}_digit_count"] = data_filled.str.count(r'\d')
            result[f"{base_name}_special_char_count"] = data_filled.str.count(r'[^a-zA-Z0-9\s]')

        # Pattern-based features
        if self.enable_patterns:
            result[f"{base_name}_has_email"] = data_filled.str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True).astype(int)
            result[f"{base_name}_has_url"] = data_filled.str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', regex=True).astype(int)
            result[f"{base_name}_has_phone"] = data_filled.str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', regex=True).astype(int)

            # Name patterns (titles, etc.)
            if 'name' in base_name.lower():
                result[f"{base_name}_has_title"] = data_filled.str.contains(r'\b(Mr|Mrs|Miss|Dr|Prof|Sr|Jr)\.?\b', regex=True).astype(int)

        # TF-IDF features
        if self.enable_tfidf and self.tfidf is not None:
            try:
                tfidf_features = self.tfidf.transform(data_filled).toarray()
                feature_names = [f"{base_name}_tfidf_{i}" for i in range(tfidf_features.shape[1])]
                tfidf_df = pd.DataFrame(tfidf_features, columns=feature_names, index=data.index)
                result = pd.concat([result, tfidf_df], axis=1)
            except Exception as e:
                logger.warning(f"Could not apply TF-IDF to {base_name}: {e}")

        # Missing indicator
        if data.isnull().any():
            result[f"{base_name}_was_missing"] = data.isnull().astype(int)

        return result

    def _generate_feature_names(self):
        """Generate feature names based on enabled transformations."""
        base_name = self.base_name
        names = []

        if self.enable_basic_features:
            names.extend([
                f"{base_name}_length",
                f"{base_name}_word_count",
                f"{base_name}_char_count",
                f"{base_name}_uppercase_ratio",
                f"{base_name}_digit_count",
                f"{base_name}_special_char_count"
            ])

        if self.enable_patterns:
            names.extend([
                f"{base_name}_has_email",
                f"{base_name}_has_url",
                f"{base_name}_has_phone"
            ])

            if 'name' in base_name.lower():
                names.append(f"{base_name}_has_title")

        if self.enable_tfidf and self.tfidf is not None:
            names.extend([f"{base_name}_tfidf_{i}" for i in range(self.max_tfidf_features)])

        names.append(f"{base_name}_was_missing")

        self.feature_names = names


class DateTimeProcessor(BaseProcessor):
    """
    Feature processor for datetime data.

    Extracts temporal components, creates cyclical features, and generates
    time-based indicators from datetime columns.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.base_name = ""

        # Configuration
        self.enable_components = config.get('enable_components', True)
        self.enable_cyclical = config.get('enable_cyclical', True)
        self.enable_derived = config.get('enable_derived', True)

    def fit(self, data: pd.Series) -> 'DateTimeProcessor':
        """
        Fit datetime processor on training data.

        Args:
            data: Datetime series to fit on

        Returns:
            self for method chaining
        """
        self.base_name = data.name or 'feature'

        # Generate feature names
        self._generate_feature_names()

        self.is_fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform datetime data into engineered features.

        Args:
            data: Datetime series to transform

        Returns:
            DataFrame with engineered datetime features
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")

        result = pd.DataFrame()
        base_name = self.base_name

        # Convert to datetime if not already
        try:
            datetime_data = pd.to_datetime(data, errors='coerce')
        except Exception as e:
            logger.warning(f"Could not convert {base_name} to datetime: {e}")
            return result

        # Basic components
        if self.enable_components:
            result[f"{base_name}_year"] = datetime_data.dt.year
            result[f"{base_name}_month"] = datetime_data.dt.month
            result[f"{base_name}_day"] = datetime_data.dt.day
            result[f"{base_name}_hour"] = datetime_data.dt.hour
            result[f"{base_name}_minute"] = datetime_data.dt.minute
            result[f"{base_name}_dayofweek"] = datetime_data.dt.dayofweek
            result[f"{base_name}_dayofyear"] = datetime_data.dt.dayofyear
            result[f"{base_name}_quarter"] = datetime_data.dt.quarter

        # Cyclical features (sine/cosine encoding)
        if self.enable_cyclical:
            # Month (12 months)
            result[f"{base_name}_month_sin"] = np.sin(2 * np.pi * datetime_data.dt.month / 12)
            result[f"{base_name}_month_cos"] = np.cos(2 * np.pi * datetime_data.dt.month / 12)

            # Day of week (7 days)
            result[f"{base_name}_dow_sin"] = np.sin(2 * np.pi * datetime_data.dt.dayofweek / 7)
            result[f"{base_name}_dow_cos"] = np.cos(2 * np.pi * datetime_data.dt.dayofweek / 7)

            # Hour (24 hours)
            result[f"{base_name}_hour_sin"] = np.sin(2 * np.pi * datetime_data.dt.hour / 24)
            result[f"{base_name}_hour_cos"] = np.cos(2 * np.pi * datetime_data.dt.hour / 24)

        # Derived features
        if self.enable_derived:
            result[f"{base_name}_is_weekend"] = (datetime_data.dt.dayofweek >= 5).astype(int)
            result[f"{base_name}_is_month_start"] = datetime_data.dt.is_month_start.astype(int)
            result[f"{base_name}_is_month_end"] = datetime_data.dt.is_month_end.astype(int)
            result[f"{base_name}_is_quarter_start"] = datetime_data.dt.is_quarter_start.astype(int)
            result[f"{base_name}_is_year_start"] = datetime_data.dt.is_year_start.astype(int)

            # Age in days from minimum date
            min_date = datetime_data.min()
            result[f"{base_name}_days_since_min"] = (datetime_data - min_date).dt.days

        # Missing indicator
        if data.isnull().any():
            result[f"{base_name}_was_missing"] = data.isnull().astype(int)

        return result

    def _generate_feature_names(self):
        """Generate feature names based on enabled transformations."""
        base_name = self.base_name
        names = []

        if self.enable_components:
            names.extend([
                f"{base_name}_year",
                f"{base_name}_month",
                f"{base_name}_day",
                f"{base_name}_hour",
                f"{base_name}_minute",
                f"{base_name}_dayofweek",
                f"{base_name}_dayofyear",
                f"{base_name}_quarter"
            ])

        if self.enable_cyclical:
            names.extend([
                f"{base_name}_month_sin",
                f"{base_name}_month_cos",
                f"{base_name}_dow_sin",
                f"{base_name}_dow_cos",
                f"{base_name}_hour_sin",
                f"{base_name}_hour_cos"
            ])

        if self.enable_derived:
            names.extend([
                f"{base_name}_is_weekend",
                f"{base_name}_is_month_start",
                f"{base_name}_is_month_end",
                f"{base_name}_is_quarter_start",
                f"{base_name}_is_year_start",
                f"{base_name}_days_since_min"
            ])

        names.append(f"{base_name}_was_missing")

        self.feature_names = names