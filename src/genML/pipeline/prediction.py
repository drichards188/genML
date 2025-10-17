"""Prediction generation stage."""

from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.genML.pipeline import config


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
        X_test = np.load(config.FEATURES_DIR / 'X_test.npy')

        # Load the most recently trained model
        # This ensures we use the latest model from the training stage
        model_files = list(config.MODELS_DIR.glob('best_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError("No trained model found in models directory")

        # Select the most recent model file (highest timestamp)
        latest_model = max(model_files, key=os.path.getctime)
        best_model = joblib.load(latest_model)

        # Load original test data for ID columns and submission formatting
        test_df = pd.read_pickle(config.DATA_DIR / 'test_data.pkl')

        # Generate model predictions on test data
        predictions = best_model.predict(X_test)                    # Predictions (class labels or continuous values)

        # Apply inverse target transformation if needed
        transform_info = joblib.load(config.FEATURES_DIR / 'target_transform.pkl')
        if transform_info['applied']:
            print(f"Applying inverse {transform_info['type']} transformation to predictions")
            print(f"Predictions range before inverse transform: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")

            # Inverse logit: y = exp(logit_y) / (1 + exp(logit_y))
            # Equivalent to: y = 1 / (1 + exp(-logit_y))  (more numerically stable)
            predictions = 1 / (1 + np.exp(-predictions))

            # Clip to original bounds for safety
            epsilon = transform_info['epsilon']
            predictions = np.clip(predictions, epsilon, 1 - epsilon)

            print(f"Predictions range after inverse transform: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")

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
        submission_df.to_csv(config.PREDICTIONS_DIR / predictions_submission_file, index=False)

        # Save detailed predictions with confidence scores
        # Use detected column names for consistency
        id_col = format_info['id_column']
        detailed_predictions = pd.DataFrame({
            id_col: test_df[id_col] if id_col in test_df.columns else test_df.iloc[:, 0],
            'Prediction': predictions,
            'Confidence': prediction_probabilities  # Probability of positive class
        })
        detailed_predictions.to_csv(config.PREDICTIONS_DIR / f'detailed_predictions_{timestamp}.csv', index=False)

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
        with open(config.REPORTS_DIR / 'prediction_report.json', 'w') as f:
            json.dump(submission_results, f, indent=2)

        return json.dumps(submission_results, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error in prediction: {str(e)}",
            "status": "failed"
        })
