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


# Domain-specific dataset fixtures

@pytest.fixture
def finance_dataset():
    """Finance domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'transaction_id': range(100),
        'price': np.exp(np.random.uniform(2, 8, 100)),
        'amount': np.exp(np.random.uniform(3, 9, 100)),
        'balance': np.exp(np.random.uniform(5, 10, 100)),
        'interest_rate': np.random.uniform(0.01, 0.1, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'default': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def healthcare_dataset():
    """Healthcare domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'patient_id': range(100),
        'age': np.random.randint(18, 90, 100),
        'weight': np.random.uniform(50, 120, 100),
        'height': np.random.uniform(150, 200, 100),
        'blood_pressure_systolic': np.random.uniform(90, 140, 100),
        'blood_pressure_diastolic': np.random.uniform(60, 90, 100),
        'heart_rate': np.random.randint(60, 100, 100),
        'diagnosis': np.random.choice(['healthy', 'disease_a', 'disease_b'], 100)
    })


@pytest.fixture
def ecommerce_dataset():
    """E-commerce domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'product_id': range(100),
        'product_name': [f'Product {i}' for i in range(100)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC'], 100),
        'price': np.random.uniform(5, 500, 100),
        'rating': np.random.uniform(1, 5, 100),
        'review_count': np.random.randint(0, 1000, 100),
        'sold': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def transportation_dataset():
    """Transportation domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'trip_id': range(100),
        'speed': np.random.uniform(20, 120, 100),
        'distance': np.random.uniform(1, 500, 100),
        'time_minutes': np.random.uniform(10, 600, 100),
        'fuel_consumption': np.random.uniform(5, 15, 100),
        'vehicle_type': np.random.choice(['car', 'truck', 'bus'], 100),
        'on_time': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def real_estate_dataset():
    """Real estate domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'property_id': range(100),
        'area_sqft': np.random.uniform(500, 5000, 100),
        'bedrooms': np.random.randint(1, 6, 100),
        'bathrooms': np.random.randint(1, 4, 100),
        'floor': np.random.randint(1, 20, 100),
        'location': np.random.choice(['Downtown', 'Suburb', 'Rural'], 100),
        'price': np.random.uniform(100000, 1000000, 100)
    })


@pytest.fixture
def text_dataset():
    """Text analysis domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'document_id': range(50),
        'text_content': [f'This is sample document {i} with various text content. ' * 5 for i in range(50)],
        'comment': [f'Comment {i}' for i in range(50)],
        'review': [f'Review text for document {i}' for i in range(50)],
        'sentiment': np.random.choice(['positive', 'negative', 'neutral'], 50)
    })


@pytest.fixture
def time_series_dataset():
    """Time series domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=365, freq='D'),
        'value': np.random.randn(365).cumsum(),
        'metric': np.random.uniform(10, 100, 365),
        'category': np.random.choice(['A', 'B', 'C'], 365),
        'target': np.random.randint(0, 2, 365)
    })


@pytest.fixture
def mixed_domain_dataset():
    """Mixed domain dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        # Finance features
        'price': np.random.uniform(10, 1000, 100),
        'amount': np.random.uniform(100, 5000, 100),
        # Healthcare features
        'age': np.random.uniform(18, 90, 100),
        'weight': np.random.uniform(50, 120, 100),
        # E-commerce features
        'product_category': np.random.choice(['A', 'B', 'C'], 100),
        'rating': np.random.uniform(1, 5, 100),
        # Time series features
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        # Target
        'target': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def high_dimensional_dataset():
    """High dimensional dataset for feature selection testing"""
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 50))
    df.columns = [f'feature_{i}' for i in range(50)]
    df['target'] = np.random.randint(0, 2, 100)
    return df


@pytest.fixture
def imbalanced_dataset():
    """Highly imbalanced dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.concatenate([np.zeros(95), np.ones(5)])  # 95% class 0, 5% class 1
    })


@pytest.fixture
def missing_values_dataset():
    """Dataset with substantial missing values for testing"""
    np.random.seed(42)
    df = pd.DataFrame({
        'mostly_missing': [np.nan if i % 2 == 0 else np.random.randn() for i in range(100)],
        'some_missing': [np.nan if i % 5 == 0 else np.random.randn() for i in range(100)],
        'rarely_missing': [np.nan if i % 20 == 0 else np.random.randn() for i in range(100)],
        'categorical_missing': [np.nan if i % 3 == 0 else np.random.choice(['A', 'B', 'C']) for i in range(100)],
        'target': np.random.randint(0, 2, 100)
    })
    return df


@pytest.fixture
def correlated_features_dataset():
    """Dataset with highly correlated features for testing"""
    np.random.seed(42)
    base_feature = np.random.randn(100)
    return pd.DataFrame({
        'feature1': base_feature,
        'feature2': base_feature + np.random.randn(100) * 0.01,  # Highly correlated
        'feature3': base_feature * 0.99,  # Highly correlated
        'feature4': base_feature + 0.001,  # Almost identical
        'feature5': np.random.randn(100),  # Independent
        'target': np.random.randint(0, 2, 100)
    })
