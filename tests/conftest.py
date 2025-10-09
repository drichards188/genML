"""
Pytest configuration and shared fixtures.

This module provides reusable test fixtures for the entire test suite,
including sample datasets, temporary directories, and mock data.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_train_df():
    """Create a sample training dataset for testing"""
    return pd.DataFrame({
        'PassengerId': range(1, 11),
        'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 2, 3],
        'Name': ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown',
                 'Charlie Davis', 'Eve White', 'Frank Black', 'Grace Green',
                 'Henry Blue', 'Ivy Red'],
        'Sex': ['male', 'female', 'male', 'female', 'male', 'female',
                'male', 'female', 'male', 'female'],
        'Age': [22, 38, 26, 35, np.nan, 54, 2, 27, np.nan, 14],
        'SibSp': [1, 1, 0, 1, 0, 0, 1, 0, 2, 0],
        'Parch': [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 151.55, 7.75, 11.13, 31.28],
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'C', 'S']
    })


@pytest.fixture
def sample_test_df():
    """Create a sample test dataset"""
    return pd.DataFrame({
        'PassengerId': range(100, 105),
        'Pclass': [3, 2, 3, 3, 2],
        'Name': ['Test1', 'Test2', 'Test3', 'Test4', 'Test5'],
        'Sex': ['male', 'female', 'male', 'female', 'male'],
        'Age': [34, 47, np.nan, 22, 30],
        'SibSp': [0, 1, 0, 0, 1],
        'Parch': [0, 0, 0, 2, 1],
        'Fare': [7.75, 21.0, 7.75, 22.36, 16.1],
        'Embarked': ['Q', 'S', 'Q', 'S', 'S']
    })


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create temporary directory with test datasets"""
    dataset_dir = tmp_path / "datasets" / "current"
    dataset_dir.mkdir(parents=True)
    return dataset_dir


@pytest.fixture
def numerical_series():
    """Sample numerical data for testing"""
    return pd.Series([1.5, 2.3, 3.7, np.nan, 5.1, 6.8, 7.2, 8.9, 9.4, 10.0], name='test_numerical')


@pytest.fixture
def categorical_series():
    """Sample categorical data for testing"""
    return pd.Series(['A', 'B', 'A', 'C', 'B', 'A', np.nan, 'C', 'B', 'A'], name='test_categorical')


@pytest.fixture
def text_series():
    """Sample text data for testing"""
    return pd.Series([
        'This is a test sentence.',
        'Another example text here!',
        'Short text',
        'A much longer piece of text with many words and characters.',
        'Test'
    ], name='test_text')


@pytest.fixture
def datetime_series():
    """Sample datetime data for testing"""
    return pd.Series(pd.date_range('2020-01-01', periods=10, freq='D'), name='test_datetime')


@pytest.fixture
def temp_outputs_dir(tmp_path):
    """Create temporary outputs directory structure"""
    outputs_dir = tmp_path / "outputs"
    (outputs_dir / "data").mkdir(parents=True)
    (outputs_dir / "features").mkdir(parents=True)
    (outputs_dir / "models").mkdir(parents=True)
    (outputs_dir / "predictions").mkdir(parents=True)
    (outputs_dir / "reports").mkdir(parents=True)
    return outputs_dir


@pytest.fixture
def sample_features():
    """Sample feature matrix for testing"""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )


@pytest.fixture
def sample_target_classification():
    """Sample classification target for testing"""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name='target')


@pytest.fixture
def sample_target_regression():
    """Sample regression target for testing"""
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 10 + 50, name='target')
