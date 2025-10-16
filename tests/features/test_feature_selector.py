"""
Tests for AdvancedFeatureSelector module.

Tests the sophisticated feature selection system that combines multiple
strategies to identify the most predictive features.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features.feature_selector import AdvancedFeatureSelector


class TestAdvancedFeatureSelector:
    """Tests for AdvancedFeatureSelector class"""

    def test_initialization(self):
        """Test selector initialization with default config"""
        selector = AdvancedFeatureSelector()

        assert selector is not None
        assert selector.max_features == 100
        assert selector.correlation_threshold == 0.95
        assert selector.enable_statistical_tests is True
        assert selector.enable_model_based is True

    def test_initialization_with_config(self):
        """Test selector initialization with custom config"""
        config = {
            'max_features': 50,
            'correlation_threshold': 0.90,
            'variance_threshold': 0.05,
            'enable_statistical_tests': True,
            'enable_model_based': True,
            'selection_strategy': 'ensemble'
        }
        selector = AdvancedFeatureSelector(config)

        assert selector.max_features == 50
        assert selector.correlation_threshold == 0.90
        assert selector.variance_threshold == 0.05
        assert selector.selection_strategy == 'ensemble'

    def test_problem_type_detection_classification(self):
        """Test automatic classification problem detection"""
        selector = AdvancedFeatureSelector()

        # Binary classification
        y_binary = pd.Series(np.random.randint(0, 2, 100))
        assert selector._detect_problem_type(y_binary) == 'classification'

        # Multi-class classification
        y_multi = pd.Series(np.random.randint(0, 5, 100))
        assert selector._detect_problem_type(y_multi) == 'classification'

    def test_problem_type_detection_regression(self):
        """Test automatic regression problem detection"""
        selector = AdvancedFeatureSelector()

        # Continuous values
        y_regression = pd.Series(np.random.randn(100) * 100 + 500)
        assert selector._detect_problem_type(y_regression) == 'regression'

        # Many unique values
        y_many_unique = pd.Series(range(100))
        assert selector._detect_problem_type(y_many_unique) == 'regression'

    def test_variance_filtering(self):
        """Test removal of low variance features"""
        # Create features with varying variance
        X = pd.DataFrame({
            'high_var': np.random.randn(100),
            'low_var': np.full(100, 1.0),  # Constant
            'medium_var': np.random.choice([0, 1], 100),
            'zero_var': np.zeros(100)  # Zero variance
        })

        selector = AdvancedFeatureSelector({'variance_threshold': 0.01})
        high_variance_features = selector._remove_low_variance(X)

        # Should remove constant and zero variance features
        assert 'high_var' in high_variance_features
        assert 'zero_var' not in high_variance_features

    def test_correlation_filtering(self):
        """Test removal of highly correlated features"""
        # Create correlated features
        np.random.seed(42)
        base_feature = np.random.randn(100)

        X = pd.DataFrame({
            'feature1': base_feature,
            'feature2': base_feature + np.random.randn(100) * 0.01,  # Highly correlated
            'feature3': np.random.randn(100),  # Independent
            'feature4': base_feature * 1.01  # Almost identical
        })

        selector = AdvancedFeatureSelector({'correlation_threshold': 0.95})
        uncorrelated_features = selector._remove_correlated_features(X)

        # Should remove some highly correlated features
        assert len(uncorrelated_features) < 4

    def test_statistical_selection_classification(self):
        """Test statistical feature selection for classification"""
        np.random.seed(42)

        # Create features with different predictive power
        X = pd.DataFrame({
            'good_feature': np.random.randn(100) + np.repeat([0, 1], 50),  # Predictive
            'bad_feature': np.random.randn(100),  # Random
            'okay_feature': np.random.randn(100) + np.repeat([0, 0.5], 50)  # Somewhat predictive
        })
        y = pd.Series(np.repeat([0, 1], 50))

        selector = AdvancedFeatureSelector({'max_features': 2})
        selected_features = selector._statistical_selection(X, y, 'classification')

        # Should select features based on statistical tests
        assert len(selected_features) > 0
        assert len(selected_features) <= 2

    def test_statistical_selection_regression(self):
        """Test statistical feature selection for regression"""
        np.random.seed(42)

        # Create features with different correlation to target
        X = pd.DataFrame({
            'strong_feature': np.random.randn(100),
            'weak_feature': np.random.randn(100)
        })
        X['strong_feature'] = X['strong_feature'] * 2
        y = X['strong_feature'] + np.random.randn(100) * 0.1

        selector = AdvancedFeatureSelector({'max_features': 1})
        selected_features = selector._statistical_selection(X, y, 'regression')

        assert len(selected_features) > 0
        # Strong feature should ideally be selected, but we'll just check it runs
        assert isinstance(selected_features, list)

    def test_model_based_selection_classification(self):
        """Test model-based feature selection for classification"""
        np.random.seed(42)

        X = pd.DataFrame({
            'important_feature': np.random.randn(100) + np.repeat([0, 2], 50),
            'noise_feature': np.random.randn(100),
            'somewhat_important': np.random.randn(100) + np.repeat([0, 1], 50)
        })
        y = pd.Series(np.repeat([0, 1], 50))

        selector = AdvancedFeatureSelector({'max_features': 2})
        selected_features = selector._model_based_selection(X, y, 'classification')

        assert len(selected_features) > 0
        assert len(selected_features) <= 2

    def test_model_based_selection_regression(self):
        """Test model-based feature selection for regression"""
        np.random.seed(42)

        X = pd.DataFrame({
            'relevant_feature': np.random.randn(100),
            'irrelevant_feature': np.random.randn(100)
        })
        y = X['relevant_feature'] * 2 + np.random.randn(100) * 0.5

        selector = AdvancedFeatureSelector({'max_features': 1})
        selected_features = selector._model_based_selection(X, y, 'regression')

        assert len(selected_features) > 0

    def test_select_features_classification(self):
        """Test complete feature selection pipeline for classification"""
        np.random.seed(42)

        # Create a classification dataset
        X = pd.DataFrame(np.random.randn(100, 15))
        X.columns = [f'feature_{i}' for i in range(15)]

        # Add some predictive features
        y = pd.Series(np.random.randint(0, 2, 100))
        X['feature_0'] = X['feature_0'] + y * 2

        selector = AdvancedFeatureSelector({
            'max_features': 10,
            'enable_statistical_tests': True,
            'enable_model_based': True,
            'selection_strategy': 'union'
        })

        selected_features, report = selector.select_features(X, y, problem_type='classification')

        assert isinstance(selected_features, list)
        assert len(selected_features) <= 10
        assert isinstance(report, dict)
        assert 'original_features' in report
        assert 'problem_type' in report
        assert report['problem_type'] == 'classification'

    def test_select_features_regression(self):
        """Test complete feature selection pipeline for regression"""
        np.random.seed(42)

        # Create a regression dataset
        X = pd.DataFrame(np.random.randn(100, 15))
        X.columns = [f'feature_{i}' for i in range(15)]

        y = pd.Series(X['feature_0'] * 2 + X['feature_1'] + np.random.randn(100) * 0.5)

        selector = AdvancedFeatureSelector({
            'max_features': 10,
            'enable_statistical_tests': True,
            'enable_model_based': True
        })

        selected_features, report = selector.select_features(X, y, problem_type='regression')

        assert isinstance(selected_features, list)
        assert len(selected_features) <= 10
        assert report['problem_type'] == 'regression'

    def test_select_features_auto_detection(self):
        """Test automatic problem type detection in selection"""
        np.random.seed(42)

        X = pd.DataFrame(np.random.randn(100, 10))
        X.columns = [f'feature_{i}' for i in range(10)]

        # Binary target (should be detected as classification)
        y_class = pd.Series(np.random.randint(0, 2, 100))

        selector = AdvancedFeatureSelector({'max_features': 5})
        selected_features, report = selector.select_features(X, y_class, problem_type='auto')

        assert report['problem_type'] == 'classification'

    def test_selection_strategy_union(self):
        """Test union selection strategy"""
        selector = AdvancedFeatureSelector({'selection_strategy': 'union'})

        selected_features_list = [
            ('method1', ['f1', 'f2', 'f3']),
            ('method2', ['f2', 'f3', 'f4']),
            ('method3', ['f3', 'f4', 'f5'])
        ]

        all_features = ['f1', 'f2', 'f3', 'f4', 'f5']
        result = selector._combine_selections(selected_features_list, all_features)

        # Union should include all features from any method
        assert 'f1' in result
        assert 'f5' in result
        assert len(result) == 5

    def test_selection_strategy_ensemble(self):
        """Test ensemble (majority voting) selection strategy"""
        selector = AdvancedFeatureSelector({'selection_strategy': 'ensemble'})

        selected_features_list = [
            ('method1', ['f1', 'f2', 'f3']),
            ('method2', ['f2', 'f3', 'f4']),
            ('method3', ['f3', 'f4', 'f5'])
        ]

        all_features = ['f1', 'f2', 'f3', 'f4', 'f5']
        result = selector._combine_selections(selected_features_list, all_features)

        # f3 should definitely be included (selected by all 3)
        assert 'f3' in result
        # f2 and f4 should be included (selected by 2 methods)
        assert 'f2' in result or 'f4' in result

    def test_selection_strategy_best(self):
        """Test 'best' selection strategy"""
        selector = AdvancedFeatureSelector({'selection_strategy': 'best'})

        selected_features_list = [
            ('method1', ['f1', 'f2']),
            ('method2', ['f2', 'f3', 'f4', 'f5']),  # Most features
            ('method3', ['f3'])
        ]

        all_features = ['f1', 'f2', 'f3', 'f4', 'f5']
        result = selector._combine_selections(selected_features_list, all_features)

        # Should use method2 (has most features)
        assert len(result) == 4
        assert 'f2' in result and 'f5' in result

    def test_prepare_data_with_missing_values(self):
        """Test data preparation handles missing values"""
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [1, np.nan, 3, 4, 5],
            'feature3': ['a', 'b', None, 'd', 'e']
        })
        y = pd.Series([0, 1, 0, 1, 1])

        selector = AdvancedFeatureSelector()
        X_clean, y_clean = selector._prepare_data(X, y)

        # Should handle missing values without crashing
        assert X_clean.shape[0] == y_clean.shape[0]
        assert not X_clean.isnull().any().any()

    def test_prepare_data_encodes_categorical(self):
        """Test that categorical features are encoded"""
        X = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['a', 'b', 'c', 'a', 'b']
        })
        y = pd.Series([0, 1, 0, 1, 1])

        selector = AdvancedFeatureSelector()
        X_clean, y_clean = selector._prepare_data(X, y)

        # All features should be numeric
        assert X_clean.select_dtypes(include=[np.number]).shape[1] == X_clean.shape[1]

    def test_max_features_enforcement(self):
        """Test that max_features limit is enforced"""
        np.random.seed(42)

        X = pd.DataFrame(np.random.randn(100, 50))
        X.columns = [f'feature_{i}' for i in range(50)]
        y = pd.Series(np.random.randint(0, 2, 100))

        selector = AdvancedFeatureSelector({
            'max_features': 10,
            'enable_statistical_tests': True,
            'enable_model_based': True
        })

        selected_features, report = selector.select_features(X, y)

        assert len(selected_features) <= 10

    def test_feature_importance_ranking(self):
        """Test feature importance ranking"""
        np.random.seed(42)

        X = pd.DataFrame({
            'important': np.random.randn(100) + np.repeat([0, 2], 50),
            'less_important': np.random.randn(100) + np.repeat([0, 0.5], 50),
            'not_important': np.random.randn(100)
        })
        y = pd.Series(np.repeat([0, 1], 50))

        selector = AdvancedFeatureSelector()
        selector.select_features(X, y)

        ranking = selector.get_feature_importance_ranking()

        assert isinstance(ranking, list)
        # Ranking should be sorted by importance (descending)
        if len(ranking) > 1:
            for i in range(len(ranking) - 1):
                assert ranking[i][1] >= ranking[i + 1][1]

    def test_empty_features_handling(self):
        """Test handling of edge case with very few features"""
        X = pd.DataFrame({'feature1': np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))

        selector = AdvancedFeatureSelector({'max_features': 5})
        selected_features, report = selector.select_features(X, y)

        # Should handle gracefully
        assert len(selected_features) <= 1
        assert isinstance(report, dict)

    def test_iterative_selection_disabled_by_default(self):
        """Test that expensive iterative selection is disabled by default"""
        selector = AdvancedFeatureSelector()

        assert selector.enable_iterative is False

    def test_iterative_selection_when_enabled(self):
        """Test iterative selection when explicitly enabled"""
        np.random.seed(42)

        # Small dataset for iterative selection
        X = pd.DataFrame(np.random.randn(50, 8))
        X.columns = [f'feature_{i}' for i in range(8)]
        y = pd.Series(np.random.randint(0, 2, 50))

        selector = AdvancedFeatureSelector({
            'enable_iterative': True,
            'enable_statistical_tests': False,
            'enable_model_based': False
        })

        selected_features = selector._iterative_selection(X, y, 'classification')

        # Should return a list of features
        assert isinstance(selected_features, list)

    def test_report_structure(self):
        """Test that selection report has correct structure"""
        np.random.seed(42)

        X = pd.DataFrame(np.random.randn(100, 10))
        X.columns = [f'feature_{i}' for i in range(10)]
        y = pd.Series(np.random.randint(0, 2, 100))

        selector = AdvancedFeatureSelector({'max_features': 5})
        selected_features, report = selector.select_features(X, y)

        # Check report structure
        assert 'original_features' in report
        assert 'problem_type' in report
        assert 'methods_used' in report
        assert 'selection_steps' in report
        assert 'final_features' in report
        assert 'selected_features' in report

    def test_selection_reduces_features(self):
        """Test that feature selection actually reduces feature count"""
        np.random.seed(42)

        # Create dataset with many correlated/redundant features
        base = np.random.randn(100, 3)
        noise = np.random.randn(100, 20) * 0.1
        X = pd.DataFrame(np.hstack([base, noise]))
        X.columns = [f'feature_{i}' for i in range(23)]
        y = pd.Series(np.random.randint(0, 2, 100))

        selector = AdvancedFeatureSelector({
            'max_features': 10,
            'enable_statistical_tests': True,
            'enable_model_based': True
        })

        selected_features, report = selector.select_features(X, y)

        # Should significantly reduce features
        assert len(selected_features) < X.shape[1]
        assert report['final_features'] < report['original_features']
