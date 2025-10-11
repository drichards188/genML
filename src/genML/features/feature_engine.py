"""
Automated Feature Engineering Engine

This module provides the main orchestration layer for automated feature engineering.
It combines data type detection, modular processors, and intelligent feature selection
to create a comprehensive feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json

from .data_analyzer import DataTypeAnalyzer
from .feature_processors import (
    NumericalProcessor,
    CategoricalProcessor,
    TextProcessor,
    DateTimeProcessor
)
from .feature_selector import AdvancedFeatureSelector
from .domain_researcher import DomainResearcher

logger = logging.getLogger(__name__)


class AutoFeatureEngine:
    """
    Automated feature engineering engine.

    This class orchestrates the entire feature engineering process:
    1. Analyzes data types and patterns
    2. Applies appropriate processors for each column type
    3. Performs feature selection and optimization
    4. Provides comprehensive reporting and transparency
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the automated feature engineering engine.

        Args:
            config: Configuration dictionary for engine settings
        """
        self.config = config or {}
        self.analyzer = DataTypeAnalyzer()
        self.feature_selector = AdvancedFeatureSelector(config.get('feature_selection', {}))
        self.domain_researcher = DomainResearcher(config.get('domain_research', {}))
        self.processors = {}
        self.feature_map = {}
        self.analysis_results = {}
        self.domain_analysis = {}
        self.research_results = {}
        self.selected_features = []
        self.feature_selection_report = {}
        self.is_fitted = False

        # Configuration settings
        self.max_features = self.config.get('max_features', 200)
        self.enable_feature_selection = self.config.get('enable_feature_selection', True)
        self.feature_importance_threshold = self.config.get('feature_importance_threshold', 0.01)
        self.exclude_id_columns = self.config.get('exclude_id_columns', True)
        self.manual_type_hints = self.config.get('manual_type_hints', {})
        raw_interaction_pairs = self.config.get('interaction_pairs', [])
        self.interaction_pairs = [tuple(pair) for pair in raw_interaction_pairs if isinstance(pair, (list, tuple)) and len(pair) == 2]
        self.interaction_feature_names = []
        self.interaction_fill_values = {}

        # Processor configurations
        self.processor_configs = {
            'numerical': self.config.get('numerical_config', {}),
            'categorical': self.config.get('categorical_config', {}),
            'text': self.config.get('text_config', {}),
            'datetime': self.config.get('datetime_config', {})
        }

    def analyze_data(self, df: pd.DataFrame, web_search_func=None) -> Dict[str, Any]:
        """
        Analyze dataset to understand structure and plan feature engineering.

        Args:
            df: DataFrame to analyze
            web_search_func: Optional function for web search research

        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Analyzing dataset with shape {df.shape}")

        # Basic data type analysis
        self.analysis_results = self.analyzer.analyze_dataset(df, overrides=self.manual_type_hints)

        # Domain analysis for intelligent feature engineering
        self.domain_analysis = self.domain_researcher.analyze_domain(df, self.analysis_results)

        # Research domain-specific strategies if web search is available
        if web_search_func:
            self.research_results = self.domain_researcher.research_feature_strategies(
                self.domain_analysis, web_search_func
            )

        # Combine all analysis results
        combined_results = {
            **self.analysis_results,
            'domain_analysis': self.domain_analysis,
            'research_results': self.research_results
        }

        return combined_results

    def fit(self, train_df: pd.DataFrame, target_col: Optional[str] = None) -> 'AutoFeatureEngine':
        """
        Fit feature engineering pipeline on training data.

        Args:
            train_df: Training DataFrame
            target_col: Name of target column (if None, will try to detect)

        Returns:
            self for method chaining
        """
        logger.info(f"Fitting feature engineering pipeline on data with shape {train_df.shape}")

        # Analyze the dataset first
        if not self.analysis_results:
            self.analyze_data(train_df)

        # Identify target column
        if target_col is None:
            target_col = self._identify_target_column(train_df)

        # Prepare features for processing (exclude target and ID columns)
        feature_columns = self._select_feature_columns(train_df, target_col)

        feature_data = train_df[feature_columns].copy()
        self._initialize_interactions(feature_data)

        # Fit processors for each column type
        self._fit_processors(feature_data)

        # Generate feature mapping
        self._generate_feature_mapping()

        # Perform feature selection if enabled and target provided
        if self.enable_feature_selection and target_col:
            self._perform_feature_selection(train_df, target_col)

        self.is_fitted = True
        logger.info(f"Feature engineering pipeline fitted successfully")
        return self

    def _generate_features_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate engineered feature DataFrame for provided raw data."""
        engineered_features = []

        for col, processor in self.processors.items():
            if col not in df.columns:
                continue
            try:
                features_df = processor.transform(df[col])
                engineered_features.append(features_df)
                logger.debug(f"Processed {col}: generated {features_df.shape[1]} features")
            except Exception as e:
                logger.warning(f"Failed to process column {col}: {e}")

        if self.interaction_pairs:
            interactions_df = self._create_interaction_features(df)
            if not interactions_df.empty:
                engineered_features.append(interactions_df)
                logger.debug(f"Generated {interactions_df.shape[1]} interaction features")

        if engineered_features:
            result = pd.concat(engineered_features, axis=1)
            try:
                # Ensure all feature columns are numeric for downstream compatibility
                numeric_result = result.apply(pd.to_numeric, errors='coerce')
                # Replace any conversion failures with 0 to keep consistent matrix shape
                numeric_result = numeric_result.fillna(0).astype(float)
                return numeric_result
            except Exception as exc:
                logger.warning(f"Could not coerce engineered features to numeric types: {exc}")
                return result

        return pd.DataFrame(index=df.index)

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create configured interaction features."""
        if not self.interaction_pairs:
            return pd.DataFrame(index=df.index)

        interaction_data = {}

        for (left, right), feature_name in zip(self.interaction_pairs, self.interaction_feature_names):
            if left not in df.columns or right not in df.columns:
                continue

            left_values = pd.to_numeric(df[left], errors='coerce')
            right_values = pd.to_numeric(df[right], errors='coerce')

            left_fill = self.interaction_fill_values.get(left)
            right_fill = self.interaction_fill_values.get(right)

            if left_fill is None:
                left_fill = float(left_values.median()) if left_values.notna().any() else 0.0
                if pd.isna(left_fill):
                    left_fill = 0.0
                self.interaction_fill_values[left] = left_fill

            if right_fill is None:
                right_fill = float(right_values.median()) if right_values.notna().any() else 0.0
                if pd.isna(right_fill):
                    right_fill = 0.0
                self.interaction_fill_values[right] = right_fill

            interaction_series = left_values.fillna(left_fill) * right_values.fillna(right_fill)
            interaction_data[feature_name] = interaction_series

        if interaction_data:
            return pd.DataFrame(interaction_data, index=df.index)

        return pd.DataFrame(index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted feature engineering pipeline.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with engineered features (optionally feature-selected)
        """
        if not self.is_fitted:
            raise ValueError("Feature engine must be fitted before transform")

        logger.info(f"Transforming data with shape {df.shape}")

        result_df = self._generate_features_for_dataframe(df)

        # Apply feature selection if available
        if self.selected_features:
            # Only keep selected features that exist in the result
            available_selected = [f for f in self.selected_features if f in result_df.columns]
            if available_selected:
                result_df = result_df[available_selected]
                logger.info(f"Applied feature selection: {len(available_selected)} features selected")

        logger.info(f"Final feature matrix shape: {result_df.shape}")
        return result_df

    def fit_transform(self, train_df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit pipeline and transform data in one step.

        Args:
            train_df: Training DataFrame
            target_col: Name of target column

        Returns:
            DataFrame with engineered features
        """
        return self.fit(train_df, target_col).transform(train_df)

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance using a simple model (GPU-accelerated if available).

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            from sklearn.preprocessing import LabelEncoder
            # Use GPU-aware Random Forest imports
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from gpu_utils import get_random_forest_classifier, get_random_forest_regressor

            # Determine if classification or regression
            if y.nunique() <= 10 and y.dtype in ['int64', 'object']:
                # Classification
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                else:
                    y_encoded = y
                RandomForestClassifierClass = get_random_forest_classifier()
                model = RandomForestClassifierClass(n_estimators=50, random_state=42)
            else:
                # Regression
                y_encoded = y
                RandomForestRegressorClass = get_random_forest_regressor()
                model = RandomForestRegressorClass(n_estimators=50, random_state=42)

            # Handle missing values in features
            X_clean = X.fillna(X.median())

            # Fit model and get importance
            model.fit(X_clean, y_encoded)
            importance_scores = dict(zip(X.columns, model.feature_importances_))

            return importance_scores

        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            return {}

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       max_features: Optional[int] = None) -> List[str]:
        """
        Select top features based on importance scores.

        Args:
            X: Feature matrix
            y: Target vector
            max_features: Maximum number of features to select

        Returns:
            List of selected feature names
        """
        if not self.enable_feature_selection:
            return list(X.columns)

        max_features = max_features or self.max_features

        # Calculate feature importance
        importance_scores = self.get_feature_importance(X, y)

        if not importance_scores:
            # Fallback: select features with low correlation and good variance
            return self._fallback_feature_selection(X, max_features)

        # Sort by importance and select top features
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter by threshold and max count
        selected_features = []
        for feature, score in sorted_features:
            if score >= self.feature_importance_threshold and len(selected_features) < max_features:
                selected_features.append(feature)

        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        return selected_features

    def get_feature_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report about feature engineering process.

        Returns:
            Dictionary containing detailed feature engineering report
        """
        report = {
            'analysis_summary': self.analysis_results,
            'processors_used': {},
            'feature_mapping': self.feature_map,
            'total_features_generated': sum(len(features) for features in self.feature_map.values()),
            'processing_config': self.config
        }

        # Add processor details
        for col, processor in self.processors.items():
            processor_type = type(processor).__name__
            report['processors_used'][col] = {
                'processor_type': processor_type,
                'feature_count': len(processor.get_feature_names()),
                'features': processor.get_feature_names()
            }

        return report

    def save_report(self, filepath: str):
        """
        Save feature engineering report to file.

        Args:
            filepath: Path to save the report
        """
        report = self.get_feature_report()

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Save as JSON
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=convert_numpy)

        logger.info(f"Feature engineering report saved to {filepath}")

    def _identify_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the target column from the dataset."""
        if self.analysis_results and self.analysis_results.get('target_candidates'):
            # Use the first candidate
            return self.analysis_results['target_candidates'][0]

        # Fallback: common target column names
        target_candidates = ['target', 'label', 'y', 'survived', 'class', 'outcome']
        for col in df.columns:
            if col.lower() in target_candidates:
                return col

        return None

    def _select_feature_columns(self, df: pd.DataFrame, target_col: Optional[str]) -> List[str]:
        """Select columns to use for feature engineering."""
        columns = list(df.columns)

        # Remove target column
        if target_col and target_col in columns:
            columns.remove(target_col)

        # Remove ID columns if configured
        if self.exclude_id_columns and self.analysis_results:
            id_columns = self.analysis_results.get('id_columns', [])
            columns = [col for col in columns if col not in id_columns]

        return columns

    def _initialize_interactions(self, df: pd.DataFrame) -> None:
        """Prepare interaction feature configuration based on available columns."""
        if not self.interaction_pairs:
            self.interaction_feature_names = []
            self.interaction_fill_values = {}
            return

        available_columns = set(df.columns)
        filtered_pairs = []
        feature_names = []
        fill_values = {}

        for left, right in self.interaction_pairs:
            if left in available_columns and right in available_columns:
                filtered_pairs.append((left, right))
                feature_names.append(f"{left}_x_{right}")
                for col in (left, right):
                    if col not in fill_values:
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        median_value = float(numeric_series.median()) if numeric_series.notna().any() else 0.0
                        if pd.isna(median_value):
                            median_value = 0.0
                        fill_values[col] = median_value

        self.interaction_pairs = filtered_pairs
        self.interaction_feature_names = feature_names
        self.interaction_fill_values = fill_values

    def _fit_processors(self, df: pd.DataFrame):
        """Fit appropriate processors for each column."""
        if not self.analysis_results:
            raise ValueError("Data analysis must be performed before fitting processors")

        column_types = self.analysis_results.get('column_types', {})

        for col in df.columns:
            if col not in column_types:
                logger.warning(f"No type analysis found for column {col}")
                continue

            col_type = column_types[col]['detected_type']

            try:
                if col_type == 'numerical':
                    processor = NumericalProcessor(self.processor_configs['numerical'])
                elif col_type == 'categorical':
                    processor = CategoricalProcessor(self.processor_configs['categorical'])
                elif col_type == 'text':
                    processor = TextProcessor(self.processor_configs['text'])
                elif col_type == 'datetime':
                    processor = DateTimeProcessor(self.processor_configs['datetime'])
                else:
                    # Default to categorical for unknown types
                    processor = CategoricalProcessor(self.processor_configs['categorical'])

                processor.fit(df[col])
                self.processors[col] = processor
                logger.debug(f"Fitted {type(processor).__name__} for column {col}")

            except Exception as e:
                logger.warning(f"Failed to fit processor for column {col}: {e}")

    def _generate_feature_mapping(self):
        """Generate mapping of original columns to engineered features."""
        for col, processor in self.processors.items():
            self.feature_map[col] = processor.get_feature_names()
        if self.interaction_feature_names:
            self.feature_map['__interactions__'] = self.interaction_feature_names

    def _perform_feature_selection(self, train_df: pd.DataFrame, target_col: str):
        """Perform feature selection using the advanced feature selector."""
        try:
            # Generate features first
            feature_columns = list(self.processors.keys())
            feature_subset = train_df[feature_columns].copy()
            X = self._generate_features_for_dataframe(feature_subset)

            if X.empty:
                logger.warning("No features generated for feature selection")
                return

            y = train_df[target_col]

            # Align indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            # Perform feature selection
            selected_features, selection_report = self.feature_selector.select_features(X, y)

            self.selected_features = selected_features
            self.feature_selection_report = selection_report

            logger.info(f"Feature selection completed: {len(selected_features)} features selected from {X.shape[1]}")

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            # Fallback: use all generated features
            self.selected_features = []

    def _fallback_feature_selection(self, X: pd.DataFrame, max_features: int) -> List[str]:
        """Fallback feature selection when importance calculation fails."""
        # Remove features with very low variance
        numeric_features = X.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            variances = numeric_features.var()
            low_variance_features = variances[variances < 0.01].index.tolist()
        else:
            low_variance_features = []

        # Remove highly correlated features
        highly_correlated = []
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            highly_correlated = [column for column in upper_triangle.columns
                               if any(upper_triangle[column] > 0.95)]

        # Select features
        excluded = set(low_variance_features + highly_correlated)
        selected = [col for col in X.columns if col not in excluded]

        # If still too many, select randomly
        if len(selected) > max_features:
            import random
            random.seed(42)
            selected = random.sample(selected, max_features)

        return selected
