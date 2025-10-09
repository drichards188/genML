"""
Generic ML Tools for Machine Learning Pipeline using CrewAI

This module contains the core data science functions that power the ML pipeline.
Each function represents a major stage in the machine learning workflow and is designed
to be modular, reusable, and well-documented. The functions handle data loading,
feature engineering, model training, and prediction generation.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory structure for organizing pipeline outputs
# This structure separates different types of artifacts for better organization and debugging
OUTPUTS_DIR = Path("outputs")                    # Root output directory
DATA_DIR = OUTPUTS_DIR / "data"                  # Processed datasets
FEATURES_DIR = OUTPUTS_DIR / "features"          # Engineered features and scalers
MODELS_DIR = OUTPUTS_DIR / "models"              # Trained models and training artifacts
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"    # Prediction files and detailed results
REPORTS_DIR = OUTPUTS_DIR / "reports"            # Analysis reports and metadata

# Ensure all output directories exist - critical for pipeline execution
# This prevents file writing errors during pipeline execution
for dir_path in [OUTPUTS_DIR, DATA_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def detect_gpu_support() -> dict:
    """
    Detect GPU availability and return appropriate configuration for models.

    Returns:
        dict: Configuration with gpu_available flag and parameters for XGBoost and RandomForest
    """
    gpu_available = False
    xgb_params = {}

    try:
        # Check if CUDA is available for XGBoost
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # nvidia-smi succeeded, GPU likely available
            # Try to use XGBoost with GPU
            try:
                # Test XGBoost GPU support
                test_model = xgb.XGBClassifier(tree_method='gpu_hist', device='cuda', n_estimators=1)
                # Quick fit test with dummy data
                test_model.fit([[1], [2]], [0, 1])
                gpu_available = True
                xgb_params = {'tree_method': 'gpu_hist', 'device': 'cuda'}
                logger.info("GPU detected and available for XGBoost training")
            except Exception as e:
                logger.warning(f"GPU detected but XGBoost GPU support unavailable: {e}")
                logger.info("Falling back to CPU training")
    except Exception as e:
        logger.info("No GPU detected, using CPU for training")

    config = {
        'gpu_available': gpu_available,
        'xgb_params': xgb_params,
        'rf_params': {'n_jobs': -1}  # Random Forest uses all CPU cores (no direct GPU support)
    }

    return config


def load_dataset() -> str:
    """
    Load Titanic train and test datasets and return summary information.

    Returns:
        JSON string with dataset summary including shapes and preview
    """
    try:
        # Check for datasets in organized directory structure first
        # Priority: datasets/current/ -> datasets/active -> root directory
        dataset_paths = [
            'datasets/current/',
            'datasets/active/',
            '.'
        ]

        train_path = None
        test_path = None

        for base_path in dataset_paths:
            potential_train = os.path.join(base_path, 'train.csv')
            potential_test = os.path.join(base_path, 'test.csv')

            if os.path.exists(potential_train) and os.path.exists(potential_test):
                train_path = potential_train
                test_path = potential_test
                print(f"Using dataset from: {base_path}")
                break

        if not train_path or not test_path:
            return json.dumps({
                "error": "train.csv and test.csv files not found. Please place dataset files in datasets/current/ or project root directory.",
                "status": "failed"
            })

        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Detect target column by comparing train vs test columns
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        train_only_cols = train_cols - test_cols

        # Exclude common ID column names
        id_patterns = {'id', 'ID', 'Id', 'index', 'Index'}
        potential_targets = [col for col in train_only_cols if col not in id_patterns]
        target_col = potential_targets[0] if potential_targets else None

        # Calculate target statistics if target found
        target_rate = None
        if target_col and target_col in train_df.columns:
            try:
                # For binary/categorical targets
                if train_df[target_col].dtype in ['object', 'category'] or train_df[target_col].nunique() <= 20:
                    target_rate = train_df[target_col].value_counts(normalize=True).to_dict()
                else:
                    # For continuous targets, provide basic stats
                    target_rate = {
                        'mean': float(train_df[target_col].mean()),
                        'std': float(train_df[target_col].std()),
                        'min': float(train_df[target_col].min()),
                        'max': float(train_df[target_col].max())
                    }
            except:
                target_rate = None

        # Create summary
        summary = {
            "status": "success",
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "train_columns": list(train_df.columns),
            "test_columns": list(test_df.columns),
            "target_column": target_col,
            "missing_values": {
                "train": train_df.isnull().sum().to_dict(),
                "test": test_df.isnull().sum().to_dict()
            },
            "target_stats": target_rate,
            "train_head": train_df.head().to_dict('records'),
            "test_head": test_df.head().to_dict('records')
        }

        # Save datasets to organized folders
        train_df.to_pickle(DATA_DIR / 'train_data.pkl')
        test_df.to_pickle(DATA_DIR / 'test_data.pkl')

        # Save data exploration report
        with open(REPORTS_DIR / 'data_exploration_report.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return json.dumps(summary, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error loading data: {str(e)}",
            "status": "failed"
        })


def engineer_features() -> str:
    """
    Create engineered features from the loaded dataset using automated feature engineering.

    This function now uses the AutoFeatureEngine to automatically detect data types,
    apply appropriate feature processors, and perform intelligent feature selection.
    It can adapt to different datasets while maintaining the same interface.

    Key capabilities:
    - Automatic data type detection (numerical, categorical, text, datetime)
    - Domain-specific feature engineering strategies
    - Intelligent feature selection
    - Configurable processing based on dataset characteristics

    Returns:
        JSON string with feature engineering results and metadata
    """
    try:
        # Load preprocessed datasets from previous pipeline stage
        train_df = pd.read_pickle(DATA_DIR / 'train_data.pkl')
        test_df = pd.read_pickle(DATA_DIR / 'test_data.pkl')

        # Initialize the automated feature engineering engine
        from src.genML.features import AutoFeatureEngine

        # Configuration for feature engineering
        config = {
            'max_features': 50,  # Reasonable limit for this pipeline
            'enable_feature_selection': True,
            'numerical_config': {
                'enable_scaling': True,
                'enable_binning': True,
                'enable_polynomial': False,  # Keep it simple for now
                'n_bins': 4
            },
            'categorical_config': {
                'encoding_method': 'label',
                'enable_frequency': True,
                'max_categories': 20
            },
            'text_config': {
                'enable_basic_features': True,
                'enable_tfidf': False,  # Disabled for performance
                'enable_patterns': True
            },
            'feature_selection': {
                'max_features': 50,
                'enable_statistical_tests': True,
                'enable_model_based': True,
                'selection_strategy': 'ensemble'
            }
        }

        # Create and fit the feature engine
        feature_engine = AutoFeatureEngine(config)

        # Analyze the training data to understand structure and domain
        analysis_results = feature_engine.analyze_data(train_df)

        # Fit the feature engineering pipeline
        # Detect target column by comparing train vs test columns (most reliable method)
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        train_only_cols = train_cols - test_cols

        # Exclude common ID column names
        id_patterns = {'id', 'ID', 'Id', 'index', 'Index'}
        potential_targets = [col for col in train_only_cols if col not in id_patterns]
        target_col = potential_targets[0] if potential_targets else None

        # Fallback to analysis-based detection if comparison fails
        if target_col is None and analysis_results.get('target_candidates'):
            target_col = analysis_results['target_candidates'][0]

        if target_col:
            print(f"Detected target column: {target_col}")
        else:
            print("Warning: Could not detect target column")

        feature_engine.fit(train_df, target_col)

        # Transform both training and test data
        train_features = feature_engine.transform(train_df)
        test_features = feature_engine.transform(test_df)

        # Combine datasets to maintain original approach for downstream compatibility
        # Add a column to track which rows are train vs test
        train_features['_is_train'] = 1
        test_features['_is_train'] = 0

        combined_features = pd.concat([train_features, test_features], ignore_index=True)

        # Extract final feature matrix for compatibility with existing pipeline
        feature_columns = [col for col in combined_features.columns if col != '_is_train']
        train_len = len(train_df)

        train_features_final = combined_features[combined_features['_is_train'] == 1][feature_columns].values
        test_features_final = combined_features[combined_features['_is_train'] == 0][feature_columns].values

        # Get target variable
        if target_col and target_col in train_df.columns:
            train_target = train_df[target_col].values
        else:
            # Fallback: assume last column or binary target
            train_target = train_df.iloc[:, -1].values

        # Save processed features in the expected format
        np.save(FEATURES_DIR / 'X_train.npy', train_features_final)
        np.save(FEATURES_DIR / 'X_test.npy', test_features_final)
        np.save(FEATURES_DIR / 'y_train.npy', train_target)

        # Save a compatibility scaler (features are already scaled by the feature engine)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(train_features_final.shape[1])
        scaler.scale_ = np.ones(train_features_final.shape[1])
        joblib.dump(scaler, FEATURES_DIR / 'scaler.pkl')

        # Save feature names for downstream compatibility
        with open(FEATURES_DIR / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_columns))

        # Save comprehensive automated feature engineering report
        feature_engine.save_report(REPORTS_DIR / 'automated_feature_engineering_report.json')

        # Prepare metadata for return (compatible with existing pipeline interface)
        metadata = {
            "status": "success",
            "features_used": feature_columns,
            "train_shape": train_features_final.shape,
            "test_shape": test_features_final.shape,
            "feature_engineering_method": "automated",
            "total_features_generated": len(feature_columns),
            "detected_domains": analysis_results.get('domain_analysis', {}).get('detected_domains', []),
            "feature_stats": {
                "mean": np.mean(train_features_final, axis=0).tolist(),
                "std": np.std(train_features_final, axis=0).tolist()
            },
            "target_distribution": {
                "positive_class": int(np.sum(train_target)),
                "negative_class": int(len(train_target) - np.sum(train_target)),
                "positive_rate": float(np.mean(train_target))
            }
        }

        # Save feature engineering report for pipeline compatibility
        with open(REPORTS_DIR / 'feature_engineering_report.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return json.dumps(metadata, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Automated feature engineering failed: {str(e)}",
            "status": "failed"
        })


def detect_problem_type() -> str:
    """
    Automatically detect whether this is a regression or classification problem.

    Analyzes the target variable to determine the problem type based on:
    - Data type (continuous vs discrete)
    - Number of unique values
    - Value range and distribution

    Returns:
        Either 'regression' or 'classification'
    """
    try:
        # Load target variable
        y_train = np.load(FEATURES_DIR / 'y_train.npy')

        # Calculate key statistics for analysis
        unique_values = len(np.unique(y_train))
        total_samples = len(y_train)
        unique_ratio = unique_values / total_samples

        # Check data type characteristics
        is_integer_like = np.allclose(y_train, np.round(y_train))
        value_range = np.max(y_train) - np.min(y_train)

        # Decision logic for problem type detection
        # Classification indicators:
        # - Small number of unique values (< 20 or < 5% of samples)
        # - Integer-like values
        # - Small value range

        if unique_values <= 20:
            # Very few unique values - likely classification
            problem_type = 'classification'
            confidence = 'high'
            reason = f'Only {unique_values} unique target values'

        elif unique_ratio < 0.05 and is_integer_like:
            # Few unique values relative to sample size, and integer-like
            problem_type = 'classification'
            confidence = 'medium'
            reason = f'Low unique ratio ({unique_ratio:.3f}) with integer-like values'

        elif unique_ratio > 0.1:
            # Many unique values relative to sample size - likely regression
            problem_type = 'regression'
            confidence = 'high'
            reason = f'High unique ratio ({unique_ratio:.3f}) indicates continuous target'

        elif not is_integer_like:
            # Non-integer values - likely regression
            problem_type = 'regression'
            confidence = 'medium'
            reason = 'Non-integer target values indicate regression'

        else:
            # Edge case - default to classification for safety
            problem_type = 'classification'
            confidence = 'low'
            reason = 'Ambiguous case - defaulting to classification'

        # Log the decision for transparency
        print(f"Problem type detected: {problem_type} (confidence: {confidence})")
        print(f"Reason: {reason}")
        print(f"Target statistics: {unique_values} unique values, {unique_ratio:.3f} ratio, range: {value_range:.3f}")

        return problem_type

    except Exception as e:
        # Fallback to classification if detection fails
        print(f"Error in problem type detection: {e}")
        print("Defaulting to classification")
        return 'classification'


def train_model_pipeline() -> str:
    """
    Train multiple ML models using cross-validation and select the best performer.

    This function implements a comprehensive model selection pipeline that:
    - Automatically detects problem type (regression vs classification)
    - Trains appropriate algorithms based on problem type
    - Uses cross-validation to get robust performance estimates
    - Prevents overfitting through proper validation methodology
    - Saves the best model for prediction generation

    Supports both regression and classification problems automatically.

    Returns:
        JSON string with model performance results and best model info
    """
    try:
        # Load preprocessed features from feature engineering stage
        X_train = np.load(FEATURES_DIR / 'X_train.npy')
        y_train = np.load(FEATURES_DIR / 'y_train.npy')

        # Automatically detect problem type
        problem_type = detect_problem_type()

        # Detect GPU support and get optimal model parameters
        gpu_config = detect_gpu_support()

        # Define model ensemble based on problem type
        # Each model has different strengths and biases
        if problem_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),                                    # Linear, interpretable
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, **gpu_config['rf_params']), # Non-linear, robust
                'XGBoost': xgb.XGBRegressor(random_state=42, **gpu_config['xgb_params'])   # Gradient boosting, high performance
            }
            # Use regular KFold for regression (no need to preserve class distribution)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            # Use regression scoring metric
            scoring_metric = 'neg_mean_squared_error'
            best_score = float('-inf')  # Higher (less negative) is better for MSE
        else:  # classification
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),  # Linear, interpretable
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, **gpu_config['rf_params']), # Non-linear, robust
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', **gpu_config['xgb_params'])        # Gradient boosting, high performance
            }
            # Use StratifiedKFold to preserve class distribution in each fold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Use classification scoring metric
            scoring_metric = 'accuracy'
            best_score = 0  # Higher is better for accuracy

        # Model evaluation pipeline
        # Systematically evaluate each model using cross-validation
        results = {}
        best_model_name = None
        best_model = None

        for name, model in models.items():
            # Cross-validation provides unbiased performance estimate
            # Using appropriate metric based on problem type
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_metric)

            # Store comprehensive results for analysis
            results[name] = {
                'mean_score': float(scores.mean()),      # Average performance across folds
                'std_score': float(scores.std()),        # Performance consistency measure
                'individual_scores': scores.tolist()     # Detailed fold-by-fold results
            }

            # Track the best performing model
            # Selection logic depends on scoring metric
            if problem_type == 'regression':
                # For regression with neg_mean_squared_error, higher (less negative) is better
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model_name = name
                    best_model = model
            else:
                # For classification with accuracy, higher is better
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model_name = name
                    best_model = model

        # Train the selected model on full training dataset
        # This gives the model access to all available training data
        best_model.fit(X_train, y_train)

        # Save the trained model for prediction stage
        # Timestamped filename prevents overwrites and enables model versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'best_model_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
        joblib.dump(best_model, MODELS_DIR / model_filename)

        # Save detailed cross-validation results for analysis
        # Provides transparency into model selection process
        cv_results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Mean_Score': [results[name]['mean_score'] for name in results.keys()],
            'Std_Score': [results[name]['std_score'] for name in results.keys()]
        })
        cv_results_df.to_csv(MODELS_DIR / 'cross_validation_results.csv', index=False)

        # Prepare results with problem type information
        model_results = {
            "status": "success",
            "problem_type": problem_type,
            "scoring_metric": scoring_metric,
            "best_model": best_model_name,
            "best_score": float(best_score),
            "all_results": results,
            "model_filename": model_filename,
            "timestamp": timestamp,
            "model_saved": str(MODELS_DIR / model_filename)
        }

        # Save model training report
        with open(REPORTS_DIR / 'model_training_report.json', 'w') as f:
            json.dump(model_results, f, indent=2)

        return json.dumps(model_results, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error in model training: {str(e)}",
            "status": "failed"
        })


def generate_predictions() -> str:
    """
    Generate predictions using the trained model and create adaptive submission file.

    This final stage of the ML pipeline uses the best model selected during training
    to generate predictions on the test dataset. It automatically detects the expected
    submission format from sample submission files and creates properly formatted outputs.

    Key functions:
    - Load the best performing model from training stage
    - Generate predictions on preprocessed test data
    - Auto-detect submission format from sample submission files
    - Create properly formatted submission file matching expected format
    - Provide prediction confidence metrics and statistics

    Returns:
        JSON string with prediction results and submission file info
    """
    try:
        # Load preprocessed test features
        X_test = np.load(FEATURES_DIR / 'X_test.npy')

        # Load the most recently trained model
        # This ensures we use the latest model from the training stage
        model_files = list(MODELS_DIR.glob('best_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError("No trained model found in models directory")

        # Select the most recent model file (highest timestamp)
        latest_model = max(model_files, key=os.path.getctime)
        best_model = joblib.load(latest_model)

        # Load original test data for ID columns and submission formatting
        test_df = pd.read_pickle(DATA_DIR / 'test_data.pkl')

        # Generate model predictions on test data
        predictions = best_model.predict(X_test)                    # Predictions (class labels or continuous values)

        # Generate prediction probabilities/confidence scores if available
        # Only classification models have predict_proba method
        try:
            if hasattr(best_model, 'predict_proba'):
                prediction_probabilities = best_model.predict_proba(X_test)[:, 1]  # Confidence scores for positive class
            else:
                # For regression models, use the predictions themselves as "confidence"
                prediction_probabilities = predictions
        except Exception as e:
            # Fallback: use predictions as probabilities
            print(f"Warning: Could not generate prediction probabilities: {e}")
            prediction_probabilities = predictions

        # Use adaptive submission formatting based on sample submission files
        # This automatically detects the expected format and creates matching submissions
        from src.genML.submission_formatter import create_adaptive_submission

        submission_result = create_adaptive_submission(
            test_df=test_df,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            project_dir="."
        )

        submission_df = submission_result['submission_df']
        main_file = submission_result['main_file']
        timestamped_file = submission_result['timestamped_file']
        format_info = submission_result['format_info']

        # Save timestamped submission file to predictions directory as well
        # This maintains the existing pipeline structure for organization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predictions_submission_file = f'submission_{timestamp}.csv'
        submission_df.to_csv(PREDICTIONS_DIR / predictions_submission_file, index=False)

        # Save detailed predictions with confidence scores
        # Use detected column names for consistency
        id_col = format_info['id_column']
        detailed_predictions = pd.DataFrame({
            id_col: test_df[id_col] if id_col in test_df.columns else test_df.iloc[:, 0],
            'Prediction': predictions,
            'Confidence': prediction_probabilities  # Probability of positive class
        })
        detailed_predictions.to_csv(PREDICTIONS_DIR / f'detailed_predictions_{timestamp}.csv', index=False)

        # Compile comprehensive prediction results with format information
        submission_results = {
            "status": "success",
            "submission_file": main_file,                    # Main submission file
            "submission_filename": predictions_submission_file,  # Timestamped backup in predictions dir
            "timestamped_file": timestamped_file,            # Timestamped file in root
            "timestamp": timestamp,
            "model_used": str(latest_model),                 # Model file path for traceability
            "predictions_count": len(predictions),           # Total predictions made
            "predicted_positive": int(np.sum(predictions)),  # Count of positive predictions
            "predicted_negative": int(len(predictions) - np.sum(predictions)),  # Count of negative predictions
            "predicted_positive_rate": float(np.mean(predictions)),  # Overall prediction rate
            "confidence_stats": {                            # Model confidence analysis
                "min_confidence": float(prediction_probabilities.min()),
                "max_confidence": float(prediction_probabilities.max()),
                "mean_confidence": float(prediction_probabilities.mean())
            },
            "format_detection": {                            # Information about detected format
                "id_column": format_info['id_column'],
                "target_column": format_info['target_column'],
                "value_type": format_info['value_type'],
                "source_file": format_info['source_file'],
                "total_expected_rows": format_info['total_rows']
            },
            "submission_preview": submission_df.head(10).to_dict('records')  # Sample for verification
        }

        # Save comprehensive prediction report for analysis
        # Documents the prediction process and results for future reference
        with open(REPORTS_DIR / 'prediction_report.json', 'w') as f:
            json.dump(submission_results, f, indent=2)

        return json.dumps(submission_results, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error in prediction: {str(e)}",
            "status": "failed"
        })