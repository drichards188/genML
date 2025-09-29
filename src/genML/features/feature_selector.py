"""
Advanced Feature Selection Module

This module provides sophisticated feature selection techniques that go beyond
simple importance scoring. It includes correlation analysis, statistical tests,
and iterative selection methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    chi2, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureSelector:
    """
    Advanced feature selection with multiple strategies.

    This class provides comprehensive feature selection using:
    - Statistical tests (ANOVA, Chi-square, Mutual Information)
    - Model-based importance (Random Forest, Linear models)
    - Correlation analysis and redundancy removal
    - Iterative selection (RFE, RFECV)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the advanced feature selector.

        Args:
            config: Configuration dictionary for selection parameters
        """
        self.config = config or {}
        self.selected_features = []
        self.feature_scores = {}
        self.selection_methods = []

        # Configuration
        self.max_features = self.config.get('max_features', 100)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.95)
        self.variance_threshold = self.config.get('variance_threshold', 0.01)
        self.enable_statistical_tests = self.config.get('enable_statistical_tests', True)
        self.enable_model_based = self.config.get('enable_model_based', True)
        self.enable_iterative = self.config.get('enable_iterative', False)  # Expensive
        self.selection_strategy = self.config.get('selection_strategy', 'ensemble')  # 'ensemble', 'best', 'union'

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       problem_type: str = 'auto') -> Tuple[List[str], Dict[str, Any]]:
        """
        Perform comprehensive feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            problem_type: 'classification', 'regression', or 'auto'

        Returns:
            Tuple of (selected_features, selection_report)
        """
        logger.info(f"Starting feature selection on {X.shape[1]} features")

        # Determine problem type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y)

        # Prepare data
        X_clean, y_clean = self._prepare_data(X, y)

        # Initialize selection report
        report = {
            'original_features': X.shape[1],
            'problem_type': problem_type,
            'methods_used': [],
            'feature_scores': {},
            'selection_steps': []
        }

        # Step 1: Remove low variance features
        high_variance_features = self._remove_low_variance(X_clean)
        X_filtered = X_clean[high_variance_features]
        report['selection_steps'].append({
            'step': 'variance_filter',
            'features_before': len(X_clean.columns),
            'features_after': len(X_filtered.columns),
            'removed_features': len(X_clean.columns) - len(X_filtered.columns)
        })

        # Step 2: Remove highly correlated features
        uncorrelated_features = self._remove_correlated_features(X_filtered)
        X_filtered = X_filtered[uncorrelated_features]
        report['selection_steps'].append({
            'step': 'correlation_filter',
            'features_before': len(high_variance_features),
            'features_after': len(X_filtered.columns),
            'removed_features': len(high_variance_features) - len(X_filtered.columns)
        })

        # Step 3: Statistical feature selection
        selected_features_list = []

        if self.enable_statistical_tests:
            statistical_features = self._statistical_selection(X_filtered, y_clean, problem_type)
            selected_features_list.append(('statistical', statistical_features))
            report['methods_used'].append('statistical_tests')

        # Step 4: Model-based feature selection
        if self.enable_model_based:
            model_features = self._model_based_selection(X_filtered, y_clean, problem_type)
            selected_features_list.append(('model_based', model_features))
            report['methods_used'].append('model_based')

        # Step 5: Iterative feature selection (expensive)
        if self.enable_iterative and len(X_filtered.columns) <= 100:  # Only for smaller feature sets
            iterative_features = self._iterative_selection(X_filtered, y_clean, problem_type)
            selected_features_list.append(('iterative', iterative_features))
            report['methods_used'].append('iterative')

        # Combine selections based on strategy
        final_features = self._combine_selections(selected_features_list, X_filtered.columns)

        # Ensure we don't exceed max_features
        if len(final_features) > self.max_features:
            # Use feature scores to prioritize if available
            if self.feature_scores:
                scored_features = [(f, self.feature_scores.get(f, 0)) for f in final_features]
                scored_features.sort(key=lambda x: x[1], reverse=True)
                final_features = [f for f, _ in scored_features[:self.max_features]]
            else:
                final_features = final_features[:self.max_features]

        self.selected_features = final_features

        # Complete report
        report['final_features'] = len(final_features)
        report['selected_features'] = final_features
        report['feature_scores'] = self.feature_scores

        logger.info(f"Feature selection complete: {len(final_features)} features selected")

        return final_features, report

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Detect if problem is classification or regression."""
        unique_values = y.nunique()
        if unique_values <= 20 and y.dtype in ['int64', 'object', 'bool']:
            return 'classification'
        else:
            return 'regression'

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for feature selection."""
        # Handle missing values
        X_clean = X.copy()

        # Fill numeric columns with median
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())

        # Fill categorical columns with mode
        categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X_clean[col] = X_clean[col].fillna(X_clean[col].mode().iloc[0] if len(X_clean[col].mode()) > 0 else 'unknown')

        # Ensure all data is numeric
        for col in categorical_cols:
            try:
                le = LabelEncoder()
                X_clean[col] = le.fit_transform(X_clean[col].astype(str))
            except Exception as e:
                logger.warning(f"Could not encode column {col}: {e}")
                X_clean = X_clean.drop(columns=[col])

        # Clean target
        y_clean = y.dropna()
        X_clean = X_clean.loc[y_clean.index]

        return X_clean, y_clean

    def _remove_low_variance(self, X: pd.DataFrame) -> List[str]:
        """Remove features with low variance."""
        try:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(X)
            selected_indices = selector.get_support(indices=True)
            return X.columns[selected_indices].tolist()
        except Exception as e:
            logger.warning(f"Could not apply variance threshold: {e}")
            return list(X.columns)

    def _remove_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """Remove highly correlated features."""
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()

            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features to drop
            to_drop = [column for column in upper_triangle.columns
                      if any(upper_triangle[column] > self.correlation_threshold)]

            return [col for col in X.columns if col not in to_drop]

        except Exception as e:
            logger.warning(f"Could not remove correlated features: {e}")
            return list(X.columns)

    def _statistical_selection(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> List[str]:
        """Perform statistical feature selection."""
        try:
            if problem_type == 'classification':
                # For classification: use f_classif or mutual_info_classif
                if all(X[col].min() >= 0 for col in X.columns):  # Non-negative features for chi2
                    score_func = chi2
                else:
                    score_func = f_classif

                selector = SelectKBest(score_func=score_func, k=min(self.max_features, X.shape[1]))
            else:
                # For regression: use f_regression or mutual_info_regression
                selector = SelectKBest(score_func=f_regression, k=min(self.max_features, X.shape[1]))

            selector.fit(X, y)

            # Get feature scores
            feature_scores = dict(zip(X.columns, selector.scores_))
            self.feature_scores.update(feature_scores)

            # Get selected features
            selected_indices = selector.get_support(indices=True)
            return X.columns[selected_indices].tolist()

        except Exception as e:
            logger.warning(f"Statistical selection failed: {e}")
            return list(X.columns)

    def _model_based_selection(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> List[str]:
        """Perform model-based feature selection."""
        try:
            if problem_type == 'classification':
                if y.nunique() == 2:
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)

            # Fit model
            model.fit(X, y)

            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance_scores = dict(zip(X.columns, np.abs(model.coef_.flatten())))
            else:
                return list(X.columns)

            self.feature_scores.update(importance_scores)

            # Select top features
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            n_features = min(self.max_features, len(sorted_features))

            return [feature for feature, _ in sorted_features[:n_features]]

        except Exception as e:
            logger.warning(f"Model-based selection failed: {e}")
            return list(X.columns)

    def _iterative_selection(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> List[str]:
        """Perform iterative feature selection (RFE/RFECV)."""
        try:
            if problem_type == 'classification':
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                estimator = LinearRegression()

            # Use RFECV for automatic feature number selection
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=3,
                scoring='accuracy' if problem_type == 'classification' else 'r2',
                min_features_to_select=min(5, X.shape[1] // 4)
            )

            selector.fit(X, y)

            # Get selected features
            selected_indices = selector.get_support(indices=True)
            return X.columns[selected_indices].tolist()

        except Exception as e:
            logger.warning(f"Iterative selection failed: {e}")
            return list(X.columns)

    def _combine_selections(self, selected_features_list: List[Tuple[str, List[str]]],
                           all_features: List[str]) -> List[str]:
        """Combine feature selections from different methods."""
        if not selected_features_list:
            return list(all_features)

        if self.selection_strategy == 'ensemble':
            # Ensemble: features selected by majority of methods
            feature_votes = {}
            for method, features in selected_features_list:
                for feature in features:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1

            # Require at least half the methods to select a feature
            min_votes = max(1, len(selected_features_list) // 2)
            return [feature for feature, votes in feature_votes.items() if votes >= min_votes]

        elif self.selection_strategy == 'best':
            # Use the method with the most features (assumes it's most confident)
            best_method = max(selected_features_list, key=lambda x: len(x[1]))
            return best_method[1]

        elif self.selection_strategy == 'union':
            # Union: features selected by any method
            all_selected = set()
            for method, features in selected_features_list:
                all_selected.update(features)
            return list(all_selected)

        else:
            # Fallback to first method
            return selected_features_list[0][1] if selected_features_list else list(all_features)

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by importance score."""
        if not self.feature_scores:
            return []

        return sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)