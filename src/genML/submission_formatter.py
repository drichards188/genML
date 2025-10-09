"""
Submission Format Detection and Generation

This module handles automatic detection of submission formats from sample files
and generates properly formatted submissions. It supports various competition
formats by analyzing existing sample submission files in the project directory.

Key features:
- Auto-detection of submission format from sample files
- Support for different ID column names and target column names
- Handling of binary classifications, probability outputs, and regression
- Fallback mechanisms when no sample submission is found
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json


class SubmissionFormatter:
    """
    Handles automatic detection and formatting of competition submissions.

    This class analyzes sample submission files to understand the expected format,
    then generates submissions that match that format exactly. It's designed to
    work across different competition platforms and problem types.
    """

    def __init__(self, project_dir: str = "."):
        """
        Initialize the submission formatter.

        Args:
            project_dir: Directory to search for sample submission files
        """
        self.project_dir = Path(project_dir)
        self.format_info = None

    def detect_submission_format(self) -> Dict[str, Any]:
        """
        Automatically detect submission format by analyzing sample files.

        Searches for common sample submission file patterns and analyzes
        their structure to understand the expected format.

        Returns:
            Dictionary containing format specifications:
            - id_column: Name of the ID column
            - target_column: Name of the target/prediction column
            - value_type: 'binary', 'probability', or 'continuous'
            - sample_values: Example values for reference
            - total_rows: Expected number of predictions
        """

        # Look for sample submission files in organized directory structure first
        dataset_paths = [
            Path("datasets/current"),
            Path("datasets/active"),
            Path(".")
        ]

        sample_files = []
        for base_path in dataset_paths:
            if base_path.exists():
                # Check for sample_submission.csv directly
                sample_file = base_path / "sample_submission.csv"
                if sample_file.exists():
                    sample_files.append(sample_file)
                    break

                # Fallback: look for any submission file patterns
                sample_patterns = [
                    "*sample_submission*",
                    "*submission*"
                ]
                for pattern in sample_patterns:
                    found_files = list(base_path.glob(pattern))
                    if found_files:
                        sample_files.extend(found_files)
                        break

                if sample_files:
                    break

        if not sample_files:
            return self._create_fallback_format()

        # Use the first found sample file
        # TODO: Could be enhanced to prefer certain file names or ask user
        sample_file = sample_files[0]

        try:
            # Read sample submission to analyze format
            sample_df = pd.read_csv(sample_file)

            # Analyze the structure
            format_info = self._analyze_sample_format(sample_df, sample_file.name)

            print(f"Detected submission format from {sample_file.name}:")
            print(f"  ID column: {format_info['id_column']}")
            print(f"  Target column: {format_info['target_column']}")
            print(f"  Value type: {format_info['value_type']}")
            print(f"  Expected rows: {format_info['total_rows']}")

            self.format_info = format_info
            return format_info

        except Exception as e:
            print(f"Error reading sample submission {sample_file}: {e}")
            return self._create_fallback_format()

    def _analyze_sample_format(self, sample_df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Analyze a sample submission DataFrame to extract format information.

        Args:
            sample_df: Sample submission DataFrame
            filename: Name of the sample file for context

        Returns:
            Format specification dictionary
        """

        columns = list(sample_df.columns)

        # Identify ID column (usually first column or contains 'id')
        id_column = columns[0]  # Default to first column
        for col in columns:
            if 'id' in col.lower():
                id_column = col
                break

        # Identify target column (usually second column or last non-ID column)
        target_column = [col for col in columns if col != id_column][0]

        # Analyze value type from target column
        target_values = sample_df[target_column].dropna()
        value_type = self._determine_value_type(target_values)

        # Get sample values for reference
        sample_values = target_values.head(5).tolist()

        return {
            'id_column': id_column,
            'target_column': target_column,
            'value_type': value_type,
            'sample_values': sample_values,
            'total_rows': len(sample_df),
            'source_file': filename,
            'all_columns': columns
        }

    def _determine_value_type(self, values: pd.Series) -> str:
        """
        Determine the type of values expected in the target column.

        Args:
            values: Target column values from sample submission

        Returns:
            One of: 'binary', 'probability', 'continuous', 'categorical'
        """

        unique_values = set(values.unique())

        # Check for binary classification (0, 1)
        if unique_values.issubset({0, 1}):
            return 'binary'

        # Check for probability values (0.0 to 1.0)
        if all(0.0 <= val <= 1.0 for val in values if not pd.isna(val)):
            # If all values are exactly 0.5, it's likely a probability placeholder
            if len(unique_values) == 1 and 0.5 in unique_values:
                return 'probability'
            # If we have diverse values between 0 and 1, it's probability
            elif len(unique_values) > 2:
                return 'probability'

        # Check for categorical (strings or small number of discrete values)
        if values.dtype == 'object' or len(unique_values) < 20:
            return 'categorical'

        # Default to continuous for everything else
        return 'continuous'

    def _create_fallback_format(self) -> Dict[str, Any]:
        """
        Create a fallback format when no sample submission is found.

        Uses common conventions and tries to infer from test data.

        Returns:
            Default format specification
        """

        # Try to analyze test data for hints
        test_files = ['test.csv']
        id_column = 'id'  # Generic default

        for test_file in test_files:
            if os.path.exists(test_file):
                try:
                    test_df = pd.read_csv(test_file)
                    # Look for likely ID columns
                    for col in test_df.columns:
                        if 'id' in col.lower():
                            id_column = col
                            break
                    break
                except:
                    pass

        print("No sample submission found. Using fallback format:")
        print(f"  ID column: {id_column}")
        print(f"  Target column: prediction")
        print(f"  Value type: binary")

        return {
            'id_column': id_column,
            'target_column': 'prediction',
            'value_type': 'binary',
            'sample_values': [0, 1],
            'total_rows': None,  # Unknown
            'source_file': 'fallback',
            'all_columns': [id_column, 'prediction']
        }

    def format_predictions(self,
                         test_df: pd.DataFrame,
                         predictions: np.ndarray,
                         prediction_probabilities: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Format predictions according to the detected submission format.

        Args:
            test_df: Original test DataFrame (for ID values)
            predictions: Model predictions (class labels)
            prediction_probabilities: Prediction probabilities (if available)

        Returns:
            Properly formatted submission DataFrame
        """

        if self.format_info is None:
            self.detect_submission_format()

        format_info = self.format_info

        # Get ID values from test data
        id_column = format_info['id_column']
        if id_column in test_df.columns:
            id_values = test_df[id_column]
        else:
            # Fallback: use index or first column
            print(f"Warning: ID column '{id_column}' not found in test data")
            id_values = test_df.iloc[:, 0] if len(test_df.columns) > 0 else range(len(predictions))

        # Format predictions based on detected value type
        target_column = format_info['target_column']
        value_type = format_info['value_type']

        if value_type == 'binary':
            # Use class predictions (0/1)
            target_values = predictions.astype(int)

        elif value_type == 'probability':
            # Use prediction probabilities if available, otherwise convert
            if prediction_probabilities is not None:
                target_values = prediction_probabilities
            else:
                # Convert binary predictions to probabilities (0.0 or 1.0)
                target_values = predictions.astype(float)

        elif value_type == 'categorical':
            # Keep predictions as-is (might need label decoding)
            target_values = predictions

        else:  # continuous
            # Use probabilities if available, otherwise predictions
            target_values = prediction_probabilities if prediction_probabilities is not None else predictions

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            id_column: id_values,
            target_column: target_values
        })

        return submission_df

    def save_submission(self,
                       submission_df: pd.DataFrame,
                       filename: str = "submission.csv",
                       timestamped: bool = True) -> Tuple[str, str]:
        """
        Save the submission file with optional timestamping.

        Args:
            submission_df: Formatted submission DataFrame
            filename: Base filename for submission
            timestamped: Whether to create a timestamped backup

        Returns:
            Tuple of (main_filename, timestamped_filename)
        """

        from datetime import datetime

        # Create submissions directory
        submissions_dir = Path("outputs/submissions")
        submissions_dir.mkdir(parents=True, exist_ok=True)

        # Save main submission file
        main_file = submissions_dir / filename
        submission_df.to_csv(main_file, index=False)
        main_file = str(main_file)

        timestamped_file = None
        if timestamped:
            # Create timestamped version
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = filename.rsplit('.', 1)[0]
            extension = filename.rsplit('.', 1)[1] if '.' in filename else 'csv'
            timestamped_filename = f"{base_name}_{timestamp}.{extension}"
            timestamped_file = submissions_dir / timestamped_filename
            submission_df.to_csv(timestamped_file, index=False)
            timestamped_file = str(timestamped_file)

        return main_file, timestamped_file

    def get_format_info(self) -> Dict[str, Any]:
        """
        Get the current format information.

        Returns:
            Format specification dictionary
        """
        if self.format_info is None:
            self.detect_submission_format()
        return self.format_info


def create_adaptive_submission(test_df: pd.DataFrame,
                             predictions: np.ndarray,
                             prediction_probabilities: Optional[np.ndarray] = None,
                             project_dir: str = ".") -> Dict[str, Any]:
    """
    Convenience function to create adaptive submissions.

    Args:
        test_df: Test DataFrame
        predictions: Model predictions
        prediction_probabilities: Prediction probabilities
        project_dir: Project directory to search for sample submissions

    Returns:
        Dictionary with submission results and metadata
    """

    formatter = SubmissionFormatter(project_dir)
    submission_df = formatter.format_predictions(test_df, predictions, prediction_probabilities)

    main_file, timestamped_file = formatter.save_submission(submission_df)
    format_info = formatter.get_format_info()

    return {
        'submission_df': submission_df,
        'main_file': main_file,
        'timestamped_file': timestamped_file,
        'format_info': format_info,
        'preview': submission_df.head(10).to_dict('records')
    }