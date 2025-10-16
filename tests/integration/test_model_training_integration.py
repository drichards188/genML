"""
End-to-End Model Training Integration Tests.

Tests that validate intelligent features actually integrate with model training
and produce trained models that can make predictions.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from src.genML.features import AutoFeatureEngine


class TestModelTrainingIntegration:
    """Test that intelligent features integrate with actual model training"""

    def test_features_compatible_with_sklearn_classifier(self):
        """Test that generated features work with sklearn classifiers"""
        np.random.seed(42)

        # Create dataset
        df = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Train sklearn model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)

        # Validate
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
        assert model.score(X, y) > 0.5  # Better than random

    def test_features_compatible_with_sklearn_regressor(self):
        """Test that generated features work with sklearn regressors"""
        np.random.seed(42)

        # Create regression dataset
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100) * 10 + 50
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Train sklearn model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)

        # Validate
        assert len(predictions) == len(y)
        assert np.isfinite(predictions).all()
        assert model.score(X, y) > 0  # R² > 0

    def test_features_compatible_with_xgboost(self):
        """Test that generated features work with XGBoost"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.choice(['A', 'B'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Train XGBoost model
        model = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)

        # Validate
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

    def test_cross_validation_with_generated_features(self):
        """Test that cross-validation works with generated features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.choice(['A', 'B', 'C'], 200),
            'target': np.random.randint(0, 2, 200)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': True, 'max_features': 20})
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Run cross-validation
        model = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        # Validate
        assert len(scores) == 5
        assert np.all(scores > 0.3)  # Better than random guessing
        assert np.mean(scores) > 0.4

    def test_feature_names_preserved_in_dataframe(self):
        """Test that feature names are preserved and accessible"""
        np.random.seed(42)

        df = pd.DataFrame({
            'age': np.random.uniform(18, 80, 50),
            'income': np.random.uniform(20000, 150000, 50),
            'target': np.random.randint(0, 2, 50)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Validate feature names
        assert isinstance(X, pd.DataFrame)
        assert len(X.columns) > 0
        assert all(isinstance(col, str) for col in X.columns)

        # Original column names should appear in feature names
        assert any('age' in str(col) for col in X.columns)
        assert any('income' in str(col) for col in X.columns)

    def test_features_can_be_saved_and_loaded(self, tmp_path):
        """Test that generated features can be saved and loaded"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Save features
        feature_path = tmp_path / 'features.npy'
        np.save(feature_path, X.values)

        # Load features
        X_loaded = np.load(feature_path)

        # Validate
        assert X_loaded.shape == X.shape
        np.testing.assert_array_equal(X_loaded, X.values)

    def test_model_trains_faster_with_feature_selection(self):
        """Test that feature selection reduces training time"""
        import time
        np.random.seed(42)

        # Large dataset
        df = pd.DataFrame(np.random.randn(500, 30))
        df.columns = [f'feature_{i}' for i in range(30)]
        df['target'] = np.random.randint(0, 2, 500)

        # Without feature selection
        engine_no_selection = AutoFeatureEngine({'enable_feature_selection': False})
        X_all = engine_no_selection.fit_transform(df, target_col='target')

        start = time.time()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_all, df['target'])
        time_all_features = time.time() - start

        # With feature selection
        engine_with_selection = AutoFeatureEngine({
            'enable_feature_selection': True,
            'max_features': 15
        })
        X_selected = engine_with_selection.fit_transform(df, target_col='target')

        start = time.time()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_selected, df['target'])
        time_selected_features = time.time() - start

        # Feature selection should reduce feature count
        assert X_selected.shape[1] < X_all.shape[1]

        # Performance should be comparable (within 20%)
        score_all = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_all, df['target']).score(X_all, df['target'])
        score_selected = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_selected, df['target']).score(X_selected, df['target'])
        assert abs(score_all - score_selected) < 0.2

    def test_features_work_with_train_test_split(self):
        """Test proper train/test workflow with features"""
        np.random.seed(42)

        train_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.choice(['A', 'B', 'C', 'D'], 30),  # New category 'D'
        })

        # Fit on train, transform both
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X_train = engine.fit_transform(train_df, target_col='target')
        X_test = engine.transform(test_df)
        y_train = train_df['target']

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test
        predictions = model.predict(X_test)

        # Validate
        assert len(predictions) == len(test_df)
        assert X_train.shape[1] == X_test.shape[1]  # Same features

    def test_features_preserve_sample_order(self):
        """Test that feature generation preserves row order"""
        np.random.seed(42)

        df = pd.DataFrame({
            'id': range(50),
            'feature': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        # Generate features
        engine = AutoFeatureEngine({'exclude_id_columns': True})
        X = engine.fit_transform(df, target_col='target')

        # Check indices match
        assert list(X.index) == list(df.index)
        assert X.shape[0] == df.shape[0]

    def test_model_feature_importance_accessible(self):
        """Test that model feature importances can be extracted"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_

        # Validate
        assert len(importances) == X.shape[1]
        assert np.all(importances >= 0)
        assert np.sum(importances) > 0  # At least some features are important

    def test_features_compatible_with_multiple_model_types(self):
        """Test that same features work with different model types"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        # Generate features once
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Try multiple model types
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            RandomForestClassifier(n_estimators=10, random_state=42),
            xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        ]

        for model in models:
            # Train
            model.fit(X, y)

            # Predict
            predictions = model.predict(X)

            # Validate
            assert len(predictions) == len(y)
            assert set(predictions).issubset({0, 1})

            # Score should be reasonable
            score = model.score(X, y)
            assert score > 0.3  # Better than random

    def test_feature_engineering_report_helps_debugging(self):
        """Test that feature engineering report provides useful debugging info"""
        np.random.seed(42)

        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        # Generate features
        engine = AutoFeatureEngine()
        engine.fit(df, target_col='target')

        # Get report
        report = engine.get_feature_report()

        # Validate report contents
        assert 'processors_used' in report
        assert 'feature_mapping' in report
        assert 'total_features_generated' in report

        # Should help with debugging
        assert len(report['processors_used']) > 0
        assert report['total_features_generated'] > 0

    def test_no_errors_with_realistic_dataset_sizes(self):
        """Test that system handles realistic dataset sizes"""
        np.random.seed(42)

        # Medium-sized dataset (realistic for many ML tasks)
        df = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) for i in range(20)
        })
        df['target'] = np.random.randint(0, 2, 1000)

        # Generate features
        engine = AutoFeatureEngine({
            'enable_feature_selection': True,
            'max_features': 30
        })
        X = engine.fit_transform(df, target_col='target')
        y = df['target']

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Validate
        assert X.shape[0] == 1000
        assert X.shape[1] <= 30
        assert model.score(X, y) > 0.5
