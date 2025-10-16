"""
End-to-End Pipeline Tests.

Tests that validate the complete workflow from feature engineering through
model training to prediction, including file I/O and the full tools.py pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from src.genML.features import AutoFeatureEngine


class TestEndToEndPipeline:
    """Test complete ML pipeline workflow"""

    def test_complete_pipeline_feature_engineering_to_prediction(self, tmp_path):
        """Test full workflow: features → save → load → train → predict"""
        np.random.seed(42)

        # Create datasets
        train_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'category': np.random.choice(['A', 'B', 'C'], 30)
        })

        # Step 1: Feature Engineering
        engine = AutoFeatureEngine({'enable_feature_selection': True, 'max_features': 20})
        X_train = engine.fit_transform(train_df, target_col='target')
        X_test = engine.transform(test_df)
        y_train = train_df['target']

        # Step 2: Save features to disk (mimicking tools.py)
        features_dir = tmp_path / 'features'
        features_dir.mkdir()

        np.save(features_dir / 'X_train.npy', X_train.values)
        np.save(features_dir / 'y_train.npy', y_train.values)
        np.save(features_dir / 'X_test.npy', X_test.values)

        # Save feature names
        with open(features_dir / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(X_train.columns))

        # Save the engine
        joblib.dump(engine, features_dir / 'feature_engine.pkl')

        # Step 3: Load features (mimicking model training)
        X_train_loaded = np.load(features_dir / 'X_train.npy')
        y_train_loaded = np.load(features_dir / 'y_train.npy')
        X_test_loaded = np.load(features_dir / 'X_test.npy')

        # Step 4: Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_loaded, y_train_loaded)

        # Step 5: Save model
        models_dir = tmp_path / 'models'
        models_dir.mkdir()
        joblib.dump(model, models_dir / 'best_model.pkl')

        # Step 6: Load model and make predictions
        model_loaded = joblib.load(models_dir / 'best_model.pkl')
        predictions = model_loaded.predict(X_test_loaded)

        # Step 7: Save predictions
        predictions_dir = tmp_path / 'predictions'
        predictions_dir.mkdir()

        submission = pd.DataFrame({
            'id': range(len(predictions)),
            'prediction': predictions
        })
        submission.to_csv(predictions_dir / 'submission.csv', index=False)

        # Validate entire pipeline
        assert features_dir.exists()
        assert models_dir.exists()
        assert predictions_dir.exists()

        assert (features_dir / 'X_train.npy').exists()
        assert (models_dir / 'best_model.pkl').exists()
        assert (predictions_dir / 'submission.csv').exists()

        # Validate predictions
        assert len(predictions) == len(test_df)
        assert set(predictions).issubset({0, 1})

        # Validate submission file
        submission_loaded = pd.read_csv(predictions_dir / 'submission.csv')
        assert len(submission_loaded) == len(test_df)

    def test_feature_engine_can_be_saved_and_reloaded(self, tmp_path):
        """Test that feature engine state can be persisted and restored"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        # Fit engine
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        engine.fit(df, target_col='target')

        # Save engine
        engine_path = tmp_path / 'engine.pkl'
        joblib.dump(engine, engine_path)

        # Create new data for transform
        new_df = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.choice(['A', 'B', 'C'], 20)  # New category
        })

        # Load engine and transform
        engine_loaded = joblib.load(engine_path)
        X_new = engine_loaded.transform(new_df)

        # Validate
        assert X_new.shape[0] == len(new_df)
        assert X_new.shape[1] > 0

    def test_features_work_with_production_workflow(self, tmp_path):
        """Test features work in a realistic production-like workflow"""
        np.random.seed(42)

        # Training phase
        train_df = pd.DataFrame({
            'numeric': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        # Fit and save everything
        engine = AutoFeatureEngine()
        X_train = engine.fit_transform(train_df, target_col='target')
        y_train = train_df['target']

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Save artifacts
        artifacts_dir = tmp_path / 'artifacts'
        artifacts_dir.mkdir()

        joblib.dump(engine, artifacts_dir / 'feature_engine.pkl')
        joblib.dump(model, artifacts_dir / 'model.pkl')

        # Production/inference phase (new process, reload everything)
        engine_prod = joblib.load(artifacts_dir / 'feature_engine.pkl')
        model_prod = joblib.load(artifacts_dir / 'model.pkl')

        # New incoming data
        new_data = pd.DataFrame({
            'numeric': [0.5, -1.2, 2.3],
            'categorical': ['A', 'B', 'D']  # New category 'D'
        })

        # Transform and predict
        X_new = engine_prod.transform(new_data)
        predictions = model_prod.predict(X_new)

        # Validate
        assert len(predictions) == len(new_data)
        assert all(pred in [0, 1] for pred in predictions)

    def test_multiple_datasets_same_engine_config(self, tmp_path):
        """Test that same engine config works on different datasets"""
        np.random.seed(42)

        config = {
            'enable_feature_selection': True,
            'max_features': 15
        }

        # Dataset 1: Titanic-like
        df1 = pd.DataFrame({
            'Age': np.random.uniform(1, 80, 100),
            'Fare': np.random.uniform(5, 500, 100),
            'Pclass': np.random.choice([1, 2, 3], 100),
            'target': np.random.randint(0, 2, 100)
        })

        engine1 = AutoFeatureEngine(config)
        X1 = engine1.fit_transform(df1, target_col='target')

        # Dataset 2: Finance-like
        df2 = pd.DataFrame({
            'price': np.random.uniform(10, 1000, 100),
            'volume': np.random.uniform(100, 10000, 100),
            'balance': np.random.uniform(1000, 50000, 100),
            'target': np.random.randint(0, 2, 100)
        })

        engine2 = AutoFeatureEngine(config)
        X2 = engine2.fit_transform(df2, target_col='target')

        # Both should work
        assert X1.shape[0] == 100
        assert X2.shape[0] == 100
        assert X1.shape[1] <= config['max_features']
        assert X2.shape[1] <= config['max_features']

    def test_error_handling_preserves_data_integrity(self):
        """Test that errors don't corrupt data"""
        np.random.seed(42)

        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Original data should be unchanged
        assert len(df) == 50
        assert 'target' in df.columns
        assert df['feature1'].notna().all()

        # Features should be separate
        assert X.shape[0] == 50
        assert 'target' not in X.columns

    def test_pipeline_handles_large_feature_count(self):
        """Test pipeline with many generated features"""
        np.random.seed(42)

        # Dataset that will generate many features
        df = pd.DataFrame({
            f'num_{i}': np.random.randn(100) for i in range(10)
        })
        df.update({
            f'cat_{i}': np.random.choice(['A', 'B', 'C'], 100) for i in range(5)
        })
        df['target'] = np.random.randint(0, 2, 100)

        # Generate many features
        engine = AutoFeatureEngine({
            'enable_feature_selection': False,
            'numerical_config': {
                'enable_scaling': True,
                'enable_binning': True,
                'enable_log_transform': True
            }
        })
        X = engine.fit_transform(df, target_col='target')

        # Should generate many features
        assert X.shape[1] > 15

        # But model should still train
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, df['target'])

        predictions = model.predict(X)
        assert len(predictions) == len(df)

    def test_feature_report_saved_and_loadable(self, tmp_path):
        """Test that feature reports can be saved and are useful"""
        np.random.seed(42)

        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })

        # Generate features and report
        engine = AutoFeatureEngine()
        engine.fit(df, target_col='target')

        # Save report
        report_path = tmp_path / 'feature_report.json'
        engine.save_report(str(report_path))

        # Report should exist
        assert report_path.exists()

        # Should be loadable JSON
        import json
        with open(report_path) as f:
            report = json.load(f)

        # Should have useful info
        assert 'processors_used' in report
        assert 'total_features_generated' in report
        assert report['total_features_generated'] > 0

    def test_features_compatible_with_ensemble_models(self):
        """Test that features work with ensemble methods"""
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

        # Try ensemble models
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        # Voting ensemble
        clf1 = LogisticRegression(random_state=42, max_iter=1000)
        clf2 = DecisionTreeClassifier(random_state=42)

        voting = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2)], voting='soft')
        voting.fit(X, y)

        predictions = voting.predict(X)
        assert len(predictions) == len(y)

    def test_batch_prediction_workflow(self):
        """Test batch prediction workflow"""
        np.random.seed(42)

        # Training
        train_df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X_train = engine.fit_transform(train_df, target_col='target')
        y_train = train_df['target']

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Batch 1
        batch1 = pd.DataFrame({'feature': np.random.randn(20)})
        X_batch1 = engine.transform(batch1)
        pred1 = model.predict(X_batch1)

        # Batch 2
        batch2 = pd.DataFrame({'feature': np.random.randn(15)})
        X_batch2 = engine.transform(batch2)
        pred2 = model.predict(X_batch2)

        # Validate
        assert len(pred1) == 20
        assert len(pred2) == 15
        assert X_batch1.shape[1] == X_batch2.shape[1]  # Same features

    def test_feature_names_tracked_through_pipeline(self):
        """Test that feature names are preserved through the pipeline"""
        np.random.seed(42)

        df = pd.DataFrame({
            'important_feature': np.random.randn(50),
            'another_feature': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        # Generate features
        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Feature names should reference original columns
        feature_names = list(X.columns)

        assert len(feature_names) > 0
        assert any('important_feature' in name for name in feature_names)
        assert any('another_feature' in name for name in feature_names)

        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, df['target'])

        # Get feature importances with names
        importances = dict(zip(feature_names, model.feature_importances_))

        # Should be able to identify most important original features
        assert len(importances) == len(feature_names)
