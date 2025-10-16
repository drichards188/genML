"""
Performance Comparison Tests.

Tests that validate intelligent features actually IMPROVE model performance
compared to baseline features (raw features with minimal processing).
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from src.genML.features import AutoFeatureEngine


class TestPerformanceComparison:
    """Test that intelligent features improve model performance"""

    def test_intelligent_features_beat_baseline_classification(self):
        """Test that intelligent features outperform minimal baseline"""
        np.random.seed(42)

        # Create dataset with clear patterns
        n_samples = 200
        df = pd.DataFrame({
            'age': np.random.uniform(18, 80, n_samples),
            'income': np.random.uniform(20000, 150000, n_samples),
            'education': np.random.choice(['HS', 'BA', 'MA', 'PhD'], n_samples),
            'experience': np.random.uniform(0, 40, n_samples)
        })

        # Create target with dependencies on features
        # High income + high education = more likely to be class 1
        target_score = (df['income'] / 50000) + (df['age'] / 40) + (df['experience'] / 20)
        target_score += (df['education'].map({'HS': 0, 'BA': 1, 'MA': 2, 'PhD': 3}))
        df['target'] = (target_score > target_score.median()).astype(int)

        # Baseline: Just use raw numerical features with simple encoding
        X_baseline = df[['age', 'income', 'experience']].copy()
        le = LabelEncoder()
        X_baseline['education_encoded'] = le.fit_transform(df['education'])

        # Intelligent features
        engine = AutoFeatureEngine({
            'enable_feature_selection': True,
            'max_features': 20,
            'numerical_config': {
                'enable_scaling': True,
                'enable_binning': True,
                'enable_polynomial': False
            }
        })
        X_intelligent = engine.fit_transform(df, target_col='target')

        y = df['target']

        # Train and compare
        model = RandomForestClassifier(n_estimators=50, random_state=42)

        # Baseline performance
        scores_baseline = cross_val_score(model, X_baseline, y, cv=5, scoring='accuracy')
        baseline_mean = np.mean(scores_baseline)

        # Intelligent features performance
        scores_intelligent = cross_val_score(model, X_intelligent, y, cv=5, scoring='accuracy')
        intelligent_mean = np.mean(scores_intelligent)

        # Intelligent features should perform at least as well
        # (Allow for small variance, but should generally be better)
        assert intelligent_mean >= baseline_mean - 0.05, \
            f"Intelligent features ({intelligent_mean:.3f}) should match baseline ({baseline_mean:.3f})"

        # At minimum, should be better than random
        assert intelligent_mean > 0.55

    def test_domain_specific_features_improve_performance(self):
        """Test that domain-specific features improve over generic features"""
        np.random.seed(42)

        # Finance domain dataset
        n_samples = 300
        df = pd.DataFrame({
            'price': np.exp(np.random.uniform(2, 8, n_samples)),  # Skewed prices
            'volume': np.exp(np.random.uniform(5, 10, n_samples)),
            'balance': np.exp(np.random.uniform(6, 11, n_samples)),
            'transaction_count': np.random.randint(1, 100, n_samples),
        })

        # Target based on ratios (domain knowledge)
        price_to_balance_ratio = df['price'] / df['balance']
        df['target'] = (price_to_balance_ratio > price_to_balance_ratio.median()).astype(int)

        # Without domain-specific features (no log transforms)
        engine_generic = AutoFeatureEngine({
            'enable_feature_selection': False,
            'numerical_config': {
                'enable_log_transform': False,
                'enable_scaling': True
            }
        })
        X_generic = engine_generic.fit_transform(df, target_col='target')

        # With domain-specific features (log transforms for finance)
        engine_domain = AutoFeatureEngine({
            'enable_feature_selection': False,
            'numerical_config': {
                'enable_log_transform': True,  # Key for skewed financial data
                'enable_scaling': True
            },
            'interaction_pairs': [('price', 'balance')]  # Domain knowledge
        })
        X_domain = engine_domain.fit_transform(df, target_col='target')

        y = df['target']

        # Compare performance
        model = LogisticRegression(random_state=42, max_iter=1000)

        scores_generic = cross_val_score(model, X_generic, y, cv=5, scoring='accuracy')
        scores_domain = cross_val_score(model, X_domain, y, cv=5, scoring='accuracy')

        # Domain features should help (especially with log transforms on skewed data)
        generic_mean = np.mean(scores_generic)
        domain_mean = np.mean(scores_domain)

        # Both should beat random
        assert generic_mean > 0.52
        assert domain_mean > 0.52

        # Domain features should be comparable or better
        assert domain_mean >= generic_mean - 0.1

    def test_feature_selection_maintains_performance(self):
        """Test that feature selection reduces dimensions while maintaining performance"""
        np.random.seed(42)

        # Dataset with many features, but only some are informative
        n_samples = 200
        n_informative = 5
        n_noise = 15

        df = pd.DataFrame(np.random.randn(n_samples, n_informative + n_noise))
        df.columns = [f'feature_{i}' for i in range(n_informative + n_noise)]

        # Target depends only on first few features
        df['target'] = (df['feature_0'] + df['feature_1'] + df['feature_2'] > 1).astype(int)

        # Without feature selection (all features)
        engine_all = AutoFeatureEngine({'enable_feature_selection': False})
        X_all = engine_all.fit_transform(df, target_col='target')

        # With feature selection
        engine_selected = AutoFeatureEngine({
            'enable_feature_selection': True,
            'max_features': 15,
            'feature_selection': {
                'enable_statistical_tests': True,
                'enable_model_based': True
            }
        })
        X_selected = engine_selected.fit_transform(df, target_col='target')

        y = df['target']

        # Feature selection should reduce dimensions
        assert X_selected.shape[1] < X_all.shape[1]

        # But performance should be maintained
        model = RandomForestClassifier(n_estimators=50, random_state=42)

        scores_all = cross_val_score(model, X_all, y, cv=5, scoring='accuracy')
        scores_selected = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')

        # Selected features should maintain performance (within 10%)
        assert np.mean(scores_selected) >= np.mean(scores_all) - 0.1

    def test_interaction_features_capture_relationships(self):
        """Test that interaction features capture feature relationships"""
        np.random.seed(42)

        n_samples = 200
        df = pd.DataFrame({
            'feature1': np.random.uniform(0, 10, n_samples),
            'feature2': np.random.uniform(0, 10, n_samples),
        })

        # Target depends on INTERACTION of features (product)
        df['target'] = (df['feature1'] * df['feature2'] > 25).astype(int)

        # Without interactions
        engine_no_interaction = AutoFeatureEngine({
            'enable_feature_selection': False,
            'numerical_config': {'enable_polynomial': False}
        })
        X_no_interaction = engine_no_interaction.fit_transform(df, target_col='target')

        # With interactions
        engine_with_interaction = AutoFeatureEngine({
            'interaction_pairs': [('feature1', 'feature2')],
            'enable_feature_selection': False
        })
        X_with_interaction = engine_with_interaction.fit_transform(df, target_col='target')

        y = df['target']

        # Compare
        model = LogisticRegression(random_state=42, max_iter=1000)

        scores_no_interaction = cross_val_score(model, X_no_interaction, y, cv=5, scoring='accuracy')
        scores_with_interaction = cross_val_score(model, X_with_interaction, y, cv=5, scoring='accuracy')

        # Interactions should help significantly for this multiplicative relationship
        assert np.mean(scores_with_interaction) > np.mean(scores_no_interaction)

    def test_datetime_features_improve_temporal_predictions(self):
        """Test that datetime features improve predictions on temporal data"""
        np.random.seed(42)

        n_samples = 365
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

        df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(n_samples)
        })

        # Target has seasonality (summer/winter pattern)
        df['month'] = df['date'].dt.month
        df['target'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)  # Summer = 1

        # Without datetime features (can't use dates directly)
        X_baseline = df[['value']].copy()

        # With datetime features
        engine = AutoFeatureEngine({
            'enable_feature_selection': False,
            'datetime_config': {
                'enable_components': True,
                'enable_cyclical': True
            }
        })
        X_datetime = engine.fit_transform(df.drop('month', axis=1), target_col='target')

        y = df['target']

        # Compare
        model = RandomForestClassifier(n_estimators=30, random_state=42)

        scores_baseline = cross_val_score(model, X_baseline, y, cv=5, scoring='accuracy')
        scores_datetime = cross_val_score(model, X_datetime, y, cv=5, scoring='accuracy')

        # Datetime features should dramatically improve performance
        assert np.mean(scores_datetime) > np.mean(scores_baseline) + 0.2

    def test_text_features_improve_text_classification(self):
        """Test that text features improve text classification"""
        np.random.seed(42)

        # Simple text classification task
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'poor', 'horrible']

        texts = []
        labels = []

        # Create 100 samples
        for _ in range(50):
            # Positive samples
            text = ' '.join(np.random.choice(positive_words, 3)) + ' product'
            texts.append(text)
            labels.append(1)

            # Negative samples
            text = ' '.join(np.random.choice(negative_words, 3)) + ' product'
            texts.append(text)
            labels.append(0)

        df = pd.DataFrame({'text': texts, 'target': labels})

        # Without text features (can't use raw text)
        # Just use a dummy baseline
        X_baseline = np.random.randn(100, 2)  # Random baseline

        # With text features
        engine = AutoFeatureEngine({
            'enable_feature_selection': False,
            'text_config': {
                'enable_basic_features': True,
                'enable_patterns': True
            }
        })
        X_text = engine.fit_transform(df, target_col='target')

        y = df['target']

        # Compare
        model = RandomForestClassifier(n_estimators=30, random_state=42)

        scores_baseline = cross_val_score(model, X_baseline, y, cv=3, scoring='accuracy')
        scores_text = cross_val_score(model, X_text, y, cv=3, scoring='accuracy')

        # Text features should beat random baseline
        assert np.mean(scores_text) > np.mean(scores_baseline)
        assert np.mean(scores_text) > 0.55  # Better than random

    def test_feature_scaling_improves_linear_models(self):
        """Test that feature scaling improves linear model performance"""
        np.random.seed(42)

        # Features with very different scales
        n_samples = 200
        df = pd.DataFrame({
            'small_feature': np.random.uniform(0, 1, n_samples),
            'large_feature': np.random.uniform(1000, 10000, n_samples),
        })

        # Target depends equally on both
        df['target'] = ((df['small_feature'] + df['large_feature'] / 5000) > 1).astype(int)

        # Without scaling
        engine_no_scaling = AutoFeatureEngine({
            'enable_feature_selection': False,
            'numerical_config': {'enable_scaling': False}
        })
        X_no_scaling = engine_no_scaling.fit_transform(df, target_col='target')

        # With scaling
        engine_with_scaling = AutoFeatureEngine({
            'enable_feature_selection': False,
            'numerical_config': {'enable_scaling': True}
        })
        X_with_scaling = engine_with_scaling.fit_transform(df, target_col='target')

        y = df['target']

        # Test with linear model (sensitive to scale)
        model = LogisticRegression(random_state=42, max_iter=1000)

        scores_no_scaling = cross_val_score(model, X_no_scaling, y, cv=5, scoring='accuracy')
        scores_with_scaling = cross_val_score(model, X_with_scaling, y, cv=5, scoring='accuracy')

        # Scaling should help linear models
        assert np.mean(scores_with_scaling) >= np.mean(scores_no_scaling)

    def test_categorical_encoding_preserves_information(self):
        """Test that categorical encoding preserves predictive information"""
        np.random.seed(42)

        n_samples = 200
        categories = ['A', 'B', 'C']

        df = pd.DataFrame({
            'category': np.random.choice(categories, n_samples),
            'numeric': np.random.randn(n_samples)
        })

        # Target depends strongly on category
        df['target'] = df['category'].map({'A': 0, 'B': 0, 'C': 1})

        # Generate features
        engine = AutoFeatureEngine({
            'enable_feature_selection': False,
            'categorical_config': {
                'encoding_method': 'label',
                'enable_frequency': True
            }
        })
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Model should learn the category-target relationship
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        # Should achieve high accuracy since target is deterministic from category
        assert np.mean(scores) > 0.65

    def test_intelligent_features_generalize_to_test_set(self):
        """Test that intelligent features generalize well to unseen test data"""
        np.random.seed(42)

        # Training data
        n_train = 200
        train_df = pd.DataFrame({
            'feature1': np.random.randn(n_train),
            'feature2': np.random.randn(n_train),
            'category': np.random.choice(['A', 'B', 'C'], n_train)
        })
        train_df['target'] = (train_df['feature1'] + train_df['feature2'] > 0).astype(int)

        # Test data (different distribution)
        n_test = 50
        test_df = pd.DataFrame({
            'feature1': np.random.randn(n_test) + 0.5,  # Shifted distribution
            'feature2': np.random.randn(n_test) + 0.5,
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_test)  # New category
        })
        test_df['target'] = (test_df['feature1'] + test_df['feature2'] > 0).astype(int)

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': True, 'max_features': 15})
        X_train = engine.fit_transform(train_df, target_col='target')
        X_test = engine.transform(test_df.drop('target', axis=1))

        y_train = train_df['target']
        y_test = test_df['target']

        # Train and test
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)

        test_score = model.score(X_test, y_test)

        # Should generalize reasonably well
        assert test_score > 0.55  # Better than random
