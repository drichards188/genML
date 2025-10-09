"""
Tests for submission formatter module.

Tests the adaptive submission formatting that detects competition format
from sample files and creates properly formatted submissions.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.genML.submission_formatter import SubmissionFormatter, create_adaptive_submission


class TestSubmissionFormatter:
    """Tests for SubmissionFormatter class"""

    def test_initialization(self):
        """Test formatter initialization"""
        formatter = SubmissionFormatter(".")

        assert formatter.project_dir == Path(".")
        assert formatter.format_info is None

    def test_detect_binary_format(self, tmp_path):
        """Test detection of binary classification format"""
        # Create sample submission
        sample_df = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 0, 1, 1]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        formatter = SubmissionFormatter(tmp_path)
        format_info = formatter.detect_submission_format()

        assert format_info['id_column'] == 'PassengerId'
        assert format_info['target_column'] == 'Survived'
        assert format_info['value_type'] == 'binary'
        assert format_info['total_rows'] == 5

    def test_detect_probability_format(self, tmp_path):
        """Test detection of probability output format"""
        sample_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'prediction': [0.5, 0.5, 0.5, 0.5, 0.5]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        formatter = SubmissionFormatter(tmp_path)
        format_info = formatter.detect_submission_format()

        assert format_info['value_type'] == 'probability'
        assert format_info['id_column'] == 'id'
        assert format_info['target_column'] == 'prediction'

    def test_detect_continuous_format(self, tmp_path):
        """Test detection of continuous/regression format"""
        sample_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'target': [12.5, 45.3, 78.1, 23.7, 56.9]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        formatter = SubmissionFormatter(tmp_path)
        format_info = formatter.detect_submission_format()

        assert format_info['value_type'] == 'continuous'

    def test_fallback_format_no_sample(self, tmp_path):
        """Test fallback format when no sample submission exists"""
        formatter = SubmissionFormatter(tmp_path)
        format_info = formatter.detect_submission_format()

        assert 'id_column' in format_info
        assert 'target_column' in format_info
        assert format_info['source_file'] == 'fallback'

    def test_format_predictions_binary(self, sample_test_df, tmp_path):
        """Test formatting predictions for binary classification"""
        # Create sample submission
        sample_df = pd.DataFrame({
            'PassengerId': range(100, 105),
            'Survived': [0, 1, 0, 1, 1]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        formatter = SubmissionFormatter(tmp_path)
        formatter.detect_submission_format()

        predictions = np.array([0, 1, 0, 1, 0])
        result = formatter.format_predictions(sample_test_df, predictions)

        assert list(result.columns) == ['PassengerId', 'Survived']
        assert result.shape[0] == len(predictions)
        assert result['Survived'].dtype in [int, np.int64, np.int32]

    def test_format_predictions_probability(self, sample_test_df, tmp_path):
        """Test formatting predictions with probabilities"""
        # Create sample submission expecting probabilities
        sample_df = pd.DataFrame({
            'PassengerId': range(100, 105),
            'Survived': [0.5, 0.5, 0.5, 0.5, 0.5]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        formatter = SubmissionFormatter(tmp_path)
        formatter.detect_submission_format()

        predictions = np.array([0, 1, 0, 1, 0])
        probabilities = np.array([0.2, 0.8, 0.3, 0.9, 0.4])
        result = formatter.format_predictions(sample_test_df, predictions, probabilities)

        assert list(result.columns) == ['PassengerId', 'Survived']
        assert result.shape[0] == len(predictions)
        # Should use probabilities for probability format
        assert result['Survived'].dtype in [float, np.float64]

    def test_format_predictions_missing_id_column(self, tmp_path):
        """Test handling when ID column is missing from test data"""
        # Create sample submission
        sample_df = pd.DataFrame({
            'PassengerId': range(1, 6),
            'Survived': [0, 1, 0, 1, 1]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        formatter = SubmissionFormatter(tmp_path)
        formatter.detect_submission_format()

        # Test data without PassengerId column
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        predictions = np.array([0, 1, 0])

        result = formatter.format_predictions(test_df, predictions)

        # Should handle gracefully by using first column or creating IDs
        assert result.shape[0] == len(predictions)

    def test_save_submission(self, tmp_path):
        """Test saving submission file"""
        formatter = SubmissionFormatter(tmp_path)

        submission_df = pd.DataFrame({
            'id': [1, 2, 3],
            'prediction': [0, 1, 0]
        })

        main_file, timestamped_file = formatter.save_submission(
            submission_df,
            filename=str(tmp_path / "test_submission.csv"),
            timestamped=True
        )

        assert Path(main_file).exists()
        assert Path(timestamped_file).exists()

        # Verify content
        loaded = pd.read_csv(main_file)
        pd.testing.assert_frame_equal(loaded, submission_df)

    def test_save_submission_no_timestamp(self, tmp_path):
        """Test saving without timestamped file"""
        formatter = SubmissionFormatter(tmp_path)

        submission_df = pd.DataFrame({
            'id': [1, 2, 3],
            'prediction': [0, 1, 0]
        })

        main_file, timestamped_file = formatter.save_submission(
            submission_df,
            filename=str(tmp_path / "test_submission.csv"),
            timestamped=False
        )

        assert Path(main_file).exists()
        assert timestamped_file is None

    def test_get_format_info(self, tmp_path):
        """Test getting format information"""
        # Create sample submission
        sample_df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [0, 1, 0]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        formatter = SubmissionFormatter(tmp_path)
        format_info = formatter.get_format_info()

        assert format_info is not None
        assert 'id_column' in format_info
        assert 'target_column' in format_info


class TestCreateAdaptiveSubmission:
    """Tests for create_adaptive_submission convenience function"""

    def test_create_adaptive_submission_basic(self, sample_test_df, tmp_path):
        """Test basic adaptive submission creation"""
        # Create sample submission
        sample_df = pd.DataFrame({
            'PassengerId': range(100, 105),
            'Survived': [0, 1, 0, 1, 1]
        })
        sample_dir = tmp_path / "datasets" / "current"
        sample_dir.mkdir(parents=True)
        sample_file = sample_dir / "sample_submission.csv"
        sample_df.to_csv(sample_file, index=False)

        predictions = np.array([0, 1, 0, 1, 0])
        probabilities = np.array([0.2, 0.8, 0.3, 0.9, 0.4])

        result = create_adaptive_submission(
            sample_test_df,
            predictions,
            probabilities,
            project_dir=str(tmp_path)
        )

        assert 'submission_df' in result
        assert 'main_file' in result
        assert 'timestamped_file' in result
        assert 'format_info' in result
        assert isinstance(result['submission_df'], pd.DataFrame)

    def test_create_adaptive_submission_returns_preview(self, sample_test_df, tmp_path):
        """Test that preview is included in results"""
        predictions = np.array([0, 1, 0, 1, 0])

        result = create_adaptive_submission(
            sample_test_df,
            predictions,
            project_dir=str(tmp_path)
        )

        assert 'preview' in result
        assert isinstance(result['preview'], list)

    def test_determine_value_type_binary(self):
        """Test value type determination for binary"""
        formatter = SubmissionFormatter()
        values = pd.Series([0, 1, 0, 1, 1, 0])

        value_type = formatter._determine_value_type(values)

        assert value_type == 'binary'

    def test_determine_value_type_probability(self):
        """Test value type determination for probabilities"""
        formatter = SubmissionFormatter()
        values = pd.Series([0.2, 0.8, 0.5, 0.3, 0.9])

        value_type = formatter._determine_value_type(values)

        assert value_type == 'probability'

    def test_determine_value_type_categorical(self):
        """Test value type determination for categorical"""
        formatter = SubmissionFormatter()
        values = pd.Series(['A', 'B', 'A', 'C', 'B'])

        value_type = formatter._determine_value_type(values)

        assert value_type == 'categorical'
