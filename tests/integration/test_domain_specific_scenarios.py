"""
Domain-specific integration tests.

Tests that the feature engineering pipeline generates appropriate domain-specific
features for different problem domains (finance, healthcare, e-commerce, etc.).
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features import AutoFeatureEngine


class TestDomainSpecificScenarios:
    """Test domain-specific feature engineering"""

    def test_finance_domain_creates_log_transforms(self):
        """Test that finance datasets get log-transformed price features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'price': np.exp(np.random.uniform(2, 8, 100)),  # Skewed prices
            'amount': np.exp(np.random.uniform(3, 9, 100)),
            'balance': np.exp(np.random.uniform(5, 10, 100)),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_log_transform': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have log-transformed features
        log_features = [col for col in X.columns if '_log' in col]
        assert len(log_features) > 0

    def test_finance_domain_creates_ratios(self):
        """Test that finance datasets get ratio features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'revenue': np.random.uniform(10000, 100000, 50),
            'cost': np.random.uniform(5000, 80000, 50),
            'profit': np.random.uniform(1000, 50000, 50),
            'target': np.random.randint(0, 2, 50)
        })

        config = {
            'interaction_pairs': [('revenue', 'cost')],
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have interaction features (ratios)
        interaction_features = [col for col in X.columns if '_x_' in col]
        assert len(interaction_features) > 0

    def test_healthcare_domain_with_age_height_weight(self):
        """Test healthcare datasets with age, height, weight features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'age': np.random.uniform(18, 90, 100),
            'weight': np.random.uniform(50, 120, 100),  # kg
            'height': np.random.uniform(150, 200, 100),  # cm
            'blood_pressure': np.random.uniform(90, 140, 100),
            'heart_rate': np.random.uniform(60, 100, 100),
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Should generate features for each health metric
        assert X.shape[1] > 5  # More than original features
        assert X.shape[0] == 100

    def test_healthcare_domain_age_binning(self):
        """Test that healthcare datasets bin age into groups"""
        np.random.seed(42)

        df = pd.DataFrame({
            'age': np.random.uniform(1, 100, 100),
            'diagnosis': np.random.choice(['healthy', 'sick'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_binning': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have binned age feature
        binned_features = [col for col in X.columns if '_bin' in col and 'age' in col]
        assert len(binned_features) > 0

    def test_ecommerce_domain_categorical_encoding(self):
        """Test e-commerce datasets with categorical features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
            'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD'], 100),
            'rating': np.random.uniform(1, 5, 100),
            'review_count': np.random.randint(0, 1000, 100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'categorical_config': {'enable_frequency': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have frequency-encoded categorical features
        frequency_features = [col for col in X.columns if '_frequency' in col]
        assert len(frequency_features) > 0

    def test_ecommerce_domain_rating_aggregations(self):
        """Test e-commerce datasets create rating-based features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'product_rating': np.random.uniform(1, 5, 100),
            'seller_rating': np.random.uniform(1, 5, 100),
            'price': np.random.uniform(10, 500, 100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'numerical_config': {'enable_binning': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should generate features for ratings
        assert X.shape[1] > 3

    def test_transportation_domain_speed_distance_time(self):
        """Test transportation datasets with speed/distance/time features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'speed': np.random.uniform(20, 120, 100),  # km/h
            'distance': np.random.uniform(1, 500, 100),  # km
            'time': np.random.uniform(10, 600, 100),  # minutes
            'fuel_consumption': np.random.uniform(5, 15, 100),  # L/100km
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'interaction_pairs': [('speed', 'time'), ('distance', 'fuel_consumption')],
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have interaction features for efficiency metrics
        interaction_features = [col for col in X.columns if '_x_' in col]
        assert len(interaction_features) > 0

    def test_real_estate_domain_area_features(self):
        """Test real estate datasets with area and room features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'area_sqft': np.random.uniform(500, 5000, 100),
            'bedrooms': np.random.randint(1, 6, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'floor': np.random.randint(1, 30, 100),
            'location': np.random.choice(['Downtown', 'Suburb', 'Rural'], 100),
            'target': np.random.uniform(100000, 1000000, 100)  # Price (regression)
        })

        config = {
            'interaction_pairs': [('area_sqft', 'bedrooms')],
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should create area per room ratios
        assert any('area_sqft' in col and '_x_' in col for col in X.columns)

    def test_text_analysis_domain_text_features(self):
        """Test text analysis datasets generate text features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'text_content': [
                f'This is a sample text document number {i} with various content.' * 5
                for i in range(50)
            ],
            'comment': [f'Comment {i} with some text.' for i in range(50)],
            'target': np.random.randint(0, 2, 50)
        })

        config = {
            'enable_feature_selection': False,
            'text_config': {'enable_basic_features': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have text length and word count features
        text_features = [col for col in X.columns if any(
            keyword in col for keyword in ['_length', '_word_count', '_char_count']
        )]
        assert len(text_features) > 0

    def test_time_series_domain_temporal_features(self):
        """Test time series datasets extract temporal features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=365, freq='D'),
            'value': np.random.randn(365),
            'metric': np.random.uniform(10, 100, 365),
            'target': np.random.randint(0, 2, 365)
        })

        config = {
            'enable_feature_selection': False,
            'datetime_config': {
                'enable_components': True,
                'enable_cyclical': True
            }
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should extract temporal components
        temporal_features = [col for col in X.columns if any(
            keyword in col for keyword in ['_month', '_day', '_year', '_dayofweek', '_sin', '_cos']
        )]
        assert len(temporal_features) > 0

    def test_time_series_domain_cyclical_encoding(self):
        """Test time series datasets create cyclical encodings"""
        np.random.seed(42)

        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'enable_feature_selection': False,
            'datetime_config': {'enable_cyclical': True}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should have sine/cosine features for cyclical patterns
        cyclical_features = [col for col in X.columns if '_sin' in col or '_cos' in col]
        assert len(cyclical_features) > 0

    def test_mixed_domain_dataset(self):
        """Test dataset with features from multiple domains"""
        np.random.seed(42)

        df = pd.DataFrame({
            # Finance
            'price': np.random.uniform(10, 1000, 100),
            # Healthcare
            'age': np.random.uniform(18, 90, 100),
            # E-commerce
            'product_category': np.random.choice(['A', 'B', 'C'], 100),
            # Time series
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            # Target
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine({'enable_feature_selection': False})
        X = engine.fit_transform(df, target_col='target')

        # Should handle mixed domains and generate features for all types
        assert X.shape[0] == 100
        assert X.shape[1] > 4  # More than original features

    def test_domain_specific_recommendations(self):
        """Test that domain analysis provides relevant recommendations"""
        np.random.seed(42)

        # Finance dataset
        df_finance = pd.DataFrame({
            'price': np.random.uniform(10, 1000, 50),
            'cost': np.random.uniform(5, 500, 50),
            'profit': np.random.uniform(1, 500, 50),
            'target': np.random.randint(0, 2, 50)
        })

        engine = AutoFeatureEngine()
        analysis = engine.analyze_data(df_finance)

        # Should detect finance domain
        assert 'domain_analysis' in analysis
        domain_analysis = analysis['domain_analysis']

        if domain_analysis and domain_analysis.get('detected_domains'):
            # Should provide recommendations
            assert 'recommendations' in domain_analysis
            assert len(domain_analysis['recommendations']) > 0

    def test_high_cardinality_ecommerce_categories(self):
        """Test e-commerce with many product categories"""
        np.random.seed(42)

        df = pd.DataFrame({
            'product_id': [f'PROD_{i:04d}' for i in range(200)],  # Many unique products
            'category': np.random.choice([f'Cat{i}' for i in range(50)], 200),  # 50 categories
            'subcategory': np.random.choice([f'SubCat{i}' for i in range(100)], 200),  # 100 subcategories
            'price': np.random.uniform(5, 500, 200),
            'target': np.random.randint(0, 2, 200)
        })

        config = {
            'exclude_id_columns': True,
            'categorical_config': {'max_categories': 50}
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should handle high cardinality gracefully
        assert X.shape[0] == 200
        assert X.shape[1] > 0

    def test_healthcare_with_missing_vitals(self):
        """Test healthcare dataset with missing vital signs"""
        np.random.seed(42)

        df = pd.DataFrame({
            'age': np.random.uniform(18, 90, 100),
            'weight': [np.nan if i % 10 == 0 else np.random.uniform(50, 120) for i in range(100)],
            'height': [np.nan if i % 15 == 0 else np.random.uniform(150, 200) for i in range(100)],
            'blood_pressure': [np.nan if i % 20 == 0 else np.random.uniform(90, 140) for i in range(100)],
            'target': np.random.randint(0, 2, 100)
        })

        engine = AutoFeatureEngine()
        X = engine.fit_transform(df, target_col='target')

        # Should handle missing values and create missing indicators
        missing_indicators = [col for col in X.columns if '_was_missing' in col]
        assert len(missing_indicators) > 0
        assert X.shape[0] == 100

    def test_transportation_efficiency_metrics(self):
        """Test that transportation creates efficiency-related features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'distance_km': np.random.uniform(10, 500, 100),
            'time_minutes': np.random.uniform(30, 600, 100),
            'fuel_used_liters': np.random.uniform(2, 50, 100),
            'vehicle_type': np.random.choice(['car', 'truck', 'bus'], 100),
            'target': np.random.randint(0, 2, 100)
        })

        config = {
            'interaction_pairs': [('distance_km', 'fuel_used_liters')],
            'enable_feature_selection': False
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should create efficiency-related interaction features
        efficiency_features = [col for col in X.columns if 'distance' in col and '_x_' in col]
        assert len(efficiency_features) > 0

    def test_real_estate_location_encoding(self):
        """Test real estate with location categorical features"""
        np.random.seed(42)

        df = pd.DataFrame({
            'area_sqft': np.random.uniform(500, 5000, 100),
            'location': np.random.choice([
                'Downtown', 'Midtown', 'Uptown', 'Suburb_North',
                'Suburb_South', 'Suburb_East', 'Suburb_West', 'Rural'
            ], 100),
            'neighborhood': np.random.choice([f'Neighborhood_{i}' for i in range(20)], 100),
            'target': np.random.uniform(100000, 1000000, 100)
        })

        config = {
            'enable_feature_selection': False,
            'categorical_config': {
                'enable_frequency': True,
                'enable_rarity': True
            }
        }
        engine = AutoFeatureEngine(config)
        X = engine.fit_transform(df, target_col='target')

        # Should encode location features
        location_features = [col for col in X.columns if 'location' in col or 'neighborhood' in col]
        assert len(location_features) > 0
