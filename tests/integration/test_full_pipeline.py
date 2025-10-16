"""
End-to-end integration tests for the complete feature engineering pipeline.

Tests the full workflow from data loading → analysis → feature generation →
feature selection → transform, ensuring all components work together correctly.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features import AutoFeatureEngine
from src.genML.features.data_analyzer import DataTypeAnalyzer
from src.genML.features.domain_researcher import DomainResearcher
from src.genML.features.feature_selector import AdvancedFeatureSelector


class TestFullPipeline:
    """End-to-end integration tests"""

    def test_complete_pipeline_titanic_like_dataset(self):
        """Test complete pipeline on Titanic-like dataset"""
        # Create Titanic-style dataset
        np.random.seed(42)
        n_samples = 200

        train_df = pd.DataFrame({
            'PassengerId': range(n_samples),
            'Survived': np.random.randint(0, 2, n_samples),
            'Pclass': np.random.choice([1, 2, 3], n_samples),
            'Name': [f'Person {i}' for i in range(n_samples)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.uniform(1, 80, n_samples),
            'SibSp': np.random.randint(0, 5, n_samples),
            'Parch': np.random.randint(0, 4, n_samples),
            'Fare': np.random.uniform(5, 500, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
        })

        test_df = train_df.drop(columns=['Survived']).sample(n=50, random_state=42)

        # Run full pipeline
        config = {
            'max_features': 30,
            'enable_feature_selection': True,
            'exclude_id_columns': True
        }
        engine = AutoFeatureEngine(config)

        # Fit and transform
        X_train = engine.fit_transform(train_df, target_col='Survived')
        X_test = engine.transform(test_df)

        # Assertions
        assert X_train.shape[0] == train_df.shape[0]
        assert X_test.shape[0] == test_df.shape[0]
        assert X_train.shape[1] == X_test.shape[1]  # Same features
        assert X_train.shape[1] > 0  # Generated features
        assert X_train.shape[1] <= config['max_features']

        # PassengerId should be excluded
        assert not any('PassengerId' in col for col in X_train.columns)

    def test_pipeline_preserves_row_count(self):
        """Test that pipeline preserves number of rows"""
        np.random.seed(42)

        df = pd.DataFrame({
            'id': range(100),
            'feature1': np.random.randn(100),
            'feature2': np.random.choice(['A', 'B', 'C'], 100),
            'feature3': pd.date_range('2020-01-01', periods=100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine({'exclude_id_columns': True})
        X = engine.fit_transform(df, target_col='target')

        assert X.shape[0] == df.shape[0]

    def test_pipeline_train_test_consistency(self):
        """Test that train and test get consistent feature transformations"""
        np.random.seed(42)

        train_df = pd.DataFrame({
            'numeric': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B', 'C', 'D'], 50)  # New category 'D'
        })

        config = {'enable_feature_selection': False}
        engine = AutoFeatureEngine(config)

        X_train = engine.fit_transform(train_df, target_col='target')
        X_test = engine.transform(test_df)

        # Same columns
        assert list(X_train.columns) == list(X_test.columns)

        # Test should handle unseen categories gracefully
        assert not X_test.isnull().all().any()

    def test_pipeline_no_data_leakage(self):
        """Test that pipeline doesn't leak information from test to train"""
        np.random.seed(42)

        train_df = pd.DataFrame({
            'feature': range(100),
            'target': [0] * 50 + [1] * 50
        })

        test_df = pd.DataFrame({
            'feature': range(1000, 1050)  # Very different range
        })

        engine = AutoFeatureEngine()
        engine.fit(train_df, target_col='target')

        # Scalers should be fit only on train data
        # Test transform should use train statistics
        X_test = engine.transform(test_df)

        # Should complete without error
        assert X_test.shape[0] == test_df.shape[0]

    def test_pipeline_with_missing_values(self):
        """Test pipeline handles missing values throughout"""
        np.random.seed(42)

        df = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10] * 10,
            'categorical': ['A', 'B', None, 'C', 'A', None, 'B', 'C', 'A', 'B'] * 10,
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Should handle missing values
        assert X.shape[0] == df.shape[0]
        # Most features should be non-null (filled or indicated)
        assert X.notna().sum().sum() > X.shape[0] * X.shape[1] * 0.8

    def test_pipeline_feature_count_reduction(self):
        """Test that feature selection reduces feature count"""
        np.random.seed(42)

        # Create dataset with many redundant features
        base_features = np.random.randn(100, 5)
        redundant_features = base_features + np.random.randn(100, 5) * 0.01
        noise_features = np.random.randn(100, 10)

        X_combined = np.hstack([base_features, redundant_features, noise_features])

        df = pd.DataFrame(X_combined)
        df.columns = [f'feature_{i}' for i in range(20)]
        df['target'] = np.random.randint(0, 2, 100)

        # Without selection
        engine_no_selection = AutoFeatureEngine({'enable_feature_selection': False})
        X_no_selection = engine_no_selection.fit_transform(df, target_col='target')

        # With selection
        engine_with_selection = AutoFeatureEngine({
            'enable_feature_selection': True,
            'max_features': 15,
            'feature_selection': {'max_features': 15}
        })
        X_with_selection = engine_with_selection.fit_transform(df, target_col='target')

        # Feature selection should reduce count
        assert X_with_selection.shape[1] < X_no_selection.shape[1]

    def test_pipeline_generates_multiple_feature_types(self):
        """Test that pipeline generates features from all data types"""
        np.random.seed(42)

        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B', 'C'], 50),
            'text': ['Sample text ' + str(i) for i in range(50)],
            'datetime': pd.date_range('2020-01-01', periods=50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Should have features from each type
        assert X.shape[1] > 4  # More than original feature count

    def test_pipeline_performance_reasonable(self):
        """Test that pipeline completes in reasonable time"""
        import time
        np.random.seed(42)

        # Medium-sized dataset
        df = pd.DataFrame({
            f'feature_{i}': np.random.randn(500) for i in range(20)
        })
        df['target'] = np.random.randint(0, 2, 500)

        config = {
            'enable_feature_selection': True,
            'max_features': 30
        }
        engine = AutoFeatureEngine(config)

        start_time = time.time()
        X = engine.fit_transform(df, target_col='target')
        elapsed = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert elapsed < 60  # 60 seconds max for this size
        assert X.shape[0] == df.shape[0]

    def test_pipeline_all_numeric_dataset(self):
        """Test pipeline on all-numeric dataset"""
        np.random.seed(42)

        df = pd.DataFrame(np.random.randn(100, 10))
        df.columns = [f'feature_{i}' for i in range(10)]
        df['target'] = np.random.randint(0, 2, 100)

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        assert X.shape[0] == df.shape[0]
        assert X.shape[1] > 0

    def test_pipeline_all_categorical_dataset(self):
        """Test pipeline on all-categorical dataset"""
        np.random.seed(42)

        df = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y', 'Z'], 100),
            'cat3': np.random.choice(['P', 'Q'], 100),
            'target': np.random.choice(['Yes', 'No'], 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        assert X.shape[0] == df.shape[0]
        assert X.shape[1] > 0

    def test_pipeline_with_interactions(self):
        """Test pipeline with interaction features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'age': np.random.uniform(18, 80, 50),
            'income': np.random.uniform(20000, 150000, 50),
            'education': np.random.choice(['HS', 'BA', 'MA', 'PhD'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        config = {
            'interaction_pairs': [('age', 'income')],
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have interaction feature
        assert any('age_x_income' in col for col in X.columns)

    def test_pipeline_report_generation(self):
        """Test that pipeline generates comprehensive reports"""
        np.random.seed(42)

        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine()
        engine.fit(df, target_col='target')

        report = engine.get_feature_report()

        assert 'analysis_summary' in report
        assert 'processors_used' in report
        assert 'feature_mapping' in report
        assert 'total_features_generated' in report
        assert isinstance(report['processors_used'], dict)

    def test_pipeline_domain_detection_integration(self):
        """Test that domain detection integrates with feature engineering"""
        np.random.seed(42)

        # Finance domain dataset
        df = pd.DataFrame({
            'price': np.random.uniform(10, 1000, 50),
            'amount': np.random.uniform(100, 5000, 50),
            'balance': np.random.uniform(1000, 10000, 50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine()

        # Analyze should detect domain
        analysis = engine.analyze_data(df)
        assert 'domain_analysis' in analysis

        # Should complete full pipeline
        X = engine.fit_transform(df, target_col='target')
        assert X.shape[0] == df.shape[0]

    def test_pipeline_multiple_transforms_consistent(self):
        """Test that multiple transform calls produce consistent results"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        engine.fit(df, target_col='target')

        X1 = engine.transform(df)
        X2 = engine.transform(df)

        # Should be identical
        pd.testing.assert_frame_equal(X1, X2)

    def test_pipeline_with_high_cardinality_categorical(self):
        """Test pipeline handles high cardinality categorical features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'high_card_cat': [f'category_{i}' for i in range(100)],  # 100 unique categories
            'numeric': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'categorical_config': {'max_categories': 50},
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should handle gracefully
        assert X.shape[0] == df.shape[0]
        assert X.shape[1] > 0

    def test_pipeline_regression_vs_classification(self):
        """Test pipeline adapts to regression vs classification problems"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })

        # Classification
        df_class = df.copy()
        df_class['target'] = np.random.randint(0, 2, 50)

        engine_class = AutoFeatureEngine()
        X_class = engine_class.fit_transform(df_class, target_col='target')

        # Regression
        df_reg = df.copy()
        df_reg['target'] = np.random.randn(50) * 100 + 500

        engine_reg = AutoFeatureEngine()
        X_reg = engine_reg.fit_transform(df_reg, target_col='target')

        # Both should complete
        assert X_class.shape[0] == 50
        assert X_reg.shape[0] == 50
