"""Prediction generation stage."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.genML.pipeline import config, context
from src.genML.pipeline.problem import find_sample_submission
from src.genML.pipeline.submission import build_submission, validate_submission

SUBMISSIONS_DIR = Path("outputs/submissions")


def generate_predictions() -> str:
    """
    Predict on the test set and write a submission in the format the metric needs.

    The output type (labels, positive-class probabilities, per-class
    probabilities, top-k labels or regression values) comes from the resolved
    problem's metric. The sample submission is used to validate the result.

    Returns:
        JSON string with prediction results and submission file info
    """
    try:
        X_test = np.load(config.FEATURES_DIR / 'X_test.npy')

        model_files = list(config.MODELS_DIR.glob('best_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError("No trained model found in models directory")
        latest_model = max(model_files, key=os.path.getctime)
        best_model = joblib.load(latest_model)

        test_df = pd.read_pickle(config.DATA_DIR / 'test_data.pkl')
        transform_info = joblib.load(config.FEATURES_DIR / 'target_transform.pkl')

        problem = context.current_problem()
        if problem is None:
            train_df = pd.read_pickle(config.DATA_DIR / 'train_data.pkl')
            problem = context.ensure_problem(train_df, test_df)

        submission_df, stats = build_submission(problem, best_model, X_test, test_df, transform_info)

        sample_path = find_sample_submission(Path(problem.data_dir)) if problem.data_dir else None
        warnings = validate_submission(submission_df, problem, sample_path)
        for warning in warnings:
            print(f"⚠️  Submission check: {warning}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submissions_dir = SUBMISSIONS_DIR.resolve()
        submissions_dir.mkdir(parents=True, exist_ok=True)
        main_file = submissions_dir / "submission.csv"
        timestamped_file = submissions_dir / f"submission_{problem.name}_{timestamp}.csv"
        submission_df.to_csv(main_file, index=False)
        submission_df.to_csv(timestamped_file, index=False)
        submission_df.to_csv(config.PREDICTIONS_DIR / f'submission_{timestamp}.csv', index=False)

        submission_results = {
            "status": "success",
            "problem": problem.name,
            "metric": problem.metric,
            "output": problem.output,
            "submission_file": str(main_file),
            "timestamped_file": str(timestamped_file),
            "timestamp": timestamp,
            "model_used": str(latest_model),
            "predictions_count": int(len(submission_df)),
            "prediction_stats": stats,
            "validation_warnings": warnings,
            "validated_against": str(sample_path) if sample_path else None,
            "submission_preview": submission_df.head(10).to_dict('records'),
        }

        with open(config.REPORTS_DIR / 'prediction_report.json', 'w') as f:
            json.dump(submission_results, f, indent=2, default=str)

        return json.dumps(submission_results, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": f"Error in prediction: {str(e)}",
            "status": "failed"
        })
