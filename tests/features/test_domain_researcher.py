"""
Tests for DomainResearcher module.

Tests the intelligent domain detection and domain-specific feature strategy
recommendation system.
"""
import pytest
import pandas as pd
import numpy as np
from src.genML.features.domain_researcher import DomainResearcher
from src.genML.features.data_analyzer import DataTypeAnalyzer


class TestDomainResearcher:
    """Tests for DomainResearcher class"""

    def test_initialization(self):
        """Test researcher initialization with default config"""
        researcher = DomainResearcher()

        assert researcher is not None
        assert researcher.domain_patterns is not None
        assert len(researcher.domain_patterns) > 0
        assert 'finance' in researcher.domain_patterns
        assert 'healthcare' in researcher.domain_patterns

    def test_initialization_with_config(self):
        """Test researcher initialization with custom config"""
        config = {
            'enable_web_research': False,
            'max_search_results': 3
        }
        researcher = DomainResearcher(config)

        assert researcher.config == config
        assert researcher.enable_web_research is False
        assert researcher.max_search_results == 3

    def test_finance_domain_detection(self):
        """Test detection of finance domain"""
        df = pd.DataFrame({
            'transaction_id': range(100),
            'price': np.random.uniform(10, 1000, 100),
            'amount': np.random.uniform(100, 5000, 100),
            'balance': np.random.uniform(1000, 10000, 100),
            'interest_rate': np.random.uniform(0.01, 0.1, 100),
            'credit_score': np.random.randint(300, 850, 100)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'detected_domains' in domain_analysis
        assert 'finance' in domain_analysis['detected_domains']

    def test_healthcare_domain_detection(self):
        """Test detection of healthcare domain"""
        df = pd.DataFrame({
            'patient_id': range(50),
            'age': np.random.randint(18, 90, 50),
            'weight': np.random.uniform(50, 120, 50),
            'height': np.random.uniform(150, 200, 50),
            'blood_pressure': [f"{np.random.randint(90, 140)}/{np.random.randint(60, 90)}" for _ in range(50)],
            'heart_rate': np.random.randint(60, 100, 50),
            'diagnosis': np.random.choice(['healthy', 'disease_a', 'disease_b'], 50)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'healthcare' in domain_analysis['detected_domains']

    def test_ecommerce_domain_detection(self):
        """Test detection of e-commerce domain"""
        df = pd.DataFrame({
            'product_id': range(100),
            'product_name': [f'Product {i}' for i in range(100)],
            'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
            'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC'], 100),
            'rating': np.random.uniform(1, 5, 100),
            'review_count': np.random.randint(0, 1000, 100),
            'purchase_count': np.random.randint(0, 500, 100)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'ecommerce' in domain_analysis['detected_domains']

    def test_transportation_domain_detection(self):
        """Test detection of transportation domain"""
        df = pd.DataFrame({
            'trip_id': range(50),
            'speed': np.random.uniform(20, 120, 50),
            'distance': np.random.uniform(1, 500, 50),
            'time': np.random.uniform(10, 600, 50),
            'fuel_consumption': np.random.uniform(5, 15, 50),
            'vehicle_type': np.random.choice(['car', 'truck', 'bus'], 50)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'transportation' in domain_analysis['detected_domains']

    def test_real_estate_domain_detection(self):
        """Test detection of real estate domain"""
        df = pd.DataFrame({
            'property_id': range(30),
            'area_sqft': np.random.uniform(500, 3000, 30),
            'bedrooms': np.random.randint(1, 6, 30),
            'bathrooms': np.random.randint(1, 4, 30),
            'floor': np.random.randint(1, 20, 30),
            'location': np.random.choice(['Downtown', 'Suburb', 'Rural'], 30),
            'room_count': np.random.randint(2, 10, 30)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'real_estate' in domain_analysis['detected_domains']

    def test_text_analysis_domain_detection(self):
        """Test detection of text analysis domain"""
        df = pd.DataFrame({
            'document_id': range(20),
            'text_content': [f'This is a sample document with various text content. Document number {i}.' * 10 for i in range(20)],
            'comment': ['This is a comment.' for _ in range(20)],
            'review': ['Product review text here.' for _ in range(20)],
            'description': ['Detailed description of the item.' for _ in range(20)]
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'text_analysis' in domain_analysis['detected_domains']

    def test_time_series_domain_detection(self):
        """Test detection of time series domain"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100),
            'year': [2020] * 100,
            'month': np.random.randint(1, 13, 100)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'time_series' in domain_analysis['detected_domains']

    def test_multiple_domain_detection(self):
        """Test detection of multiple domains in mixed dataset"""
        df = pd.DataFrame({
            'id': range(50),
            'price': np.random.uniform(10, 1000, 50),  # Finance
            'age': np.random.randint(18, 90, 50),  # Healthcare
            'product_name': [f'Product {i}' for i in range(50)],  # E-commerce
            'timestamp': pd.date_range('2020-01-01', periods=50, freq='D')  # Time series
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        # Should detect multiple domains
        assert len(domain_analysis['detected_domains']) >= 2

    def test_no_domain_detected(self):
        """Test handling when no specific domain is detected"""
        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        # Should complete without error
        assert 'detected_domains' in domain_analysis
        assert isinstance(domain_analysis['detected_domains'], list)

    def test_column_insights_generation(self):
        """Test generation of column insights"""
        df = pd.DataFrame({
            'user_id': range(50),
            'length_cm': np.random.uniform(100, 200, 50),
            'category_type': np.random.choice(['A', 'B', 'C'], 50),
            'created_date': pd.date_range('2020-01-01', periods=50)
        })

        researcher = DomainResearcher()
        column_insights = researcher._analyze_column_names(df.columns.tolist())

        assert 'domain_keywords' in column_insights
        assert 'pattern_matches' in column_insights
        assert 'semantic_groups' in column_insights

        # Check semantic grouping
        if column_insights['semantic_groups']:
            assert 'identifiers' in column_insights['semantic_groups'] or \
                   'measurements' in column_insights['semantic_groups'] or \
                   'categories' in column_insights['semantic_groups'] or \
                   'temporal' in column_insights['semantic_groups']

    def test_feature_strategies_for_finance(self):
        """Test that finance domain gets appropriate feature strategies"""
        df = pd.DataFrame({
            'price': np.random.uniform(10, 1000, 50),
            'cost': np.random.uniform(5, 500, 50),
            'balance': np.random.uniform(1000, 10000, 50)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        if 'finance' in domain_analysis['detected_domains']:
            strategies = domain_analysis['feature_strategies'].get('finance', {})
            assert 'recommended_techniques' in strategies
            assert 'log_transforms' in strategies['recommended_techniques'] or \
                   'ratios' in strategies['recommended_techniques']

    def test_feature_strategies_for_healthcare(self):
        """Test that healthcare domain gets appropriate feature strategies"""
        df = pd.DataFrame({
            'age': np.random.randint(18, 90, 50),
            'weight': np.random.uniform(50, 120, 50),
            'height': np.random.uniform(150, 200, 50)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        if 'healthcare' in domain_analysis['detected_domains']:
            strategies = domain_analysis['feature_strategies'].get('healthcare', {})
            assert 'recommended_techniques' in strategies
            assert 'specific_recommendations' in strategies

    def test_research_queries_generation(self):
        """Test generation of research queries for domains"""
        df = pd.DataFrame({
            'price': np.random.uniform(10, 1000, 50),
            'amount': np.random.uniform(100, 5000, 50)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'research_queries' in domain_analysis
        assert isinstance(domain_analysis['research_queries'], list)

    def test_recommendations_generation(self):
        """Test generation of domain recommendations"""
        df = pd.DataFrame({
            'id': range(50),
            'price': np.random.uniform(10, 1000, 50),
            'date': pd.date_range('2020-01-01', periods=50)
        })

        analyzer = DataTypeAnalyzer()
        analysis_results = analyzer.analyze_dataset(df)

        researcher = DomainResearcher()
        domain_analysis = researcher.analyze_domain(df, analysis_results)

        assert 'recommendations' in domain_analysis
        assert isinstance(domain_analysis['recommendations'], list)
        assert len(domain_analysis['recommendations']) > 0

    def test_research_without_web_search(self):
        """Test research functionality without web search"""
        config = {'enable_web_research': False}
        researcher = DomainResearcher(config)

        domain_analysis = {
            'detected_domains': ['finance', 'healthcare']
        }

        # Should not attempt web research
        research_results = researcher.research_feature_strategies(domain_analysis, web_search_func=None)

        assert research_results['research_performed'] is False

    def test_semantic_grouping_identifiers(self):
        """Test semantic grouping identifies ID columns"""
        columns = ['user_id', 'transaction_key', 'record_index', 'name', 'value']

        researcher = DomainResearcher()
        insights = researcher._analyze_column_names(columns)

        assert 'semantic_groups' in insights
        if 'identifiers' in insights['semantic_groups']:
            identifiers = insights['semantic_groups']['identifiers']
            assert 'user_id' in identifiers or 'transaction_key' in identifiers

    def test_semantic_grouping_measurements(self):
        """Test semantic grouping identifies measurement columns"""
        columns = ['height', 'width', 'length', 'weight', 'area', 'name']

        researcher = DomainResearcher()
        insights = researcher._analyze_column_names(columns)

        assert 'semantic_groups' in insights
        if 'measurements' in insights['semantic_groups']:
            measurements = insights['semantic_groups']['measurements']
            assert len(measurements) >= 1

    def test_semantic_grouping_categories(self):
        """Test semantic grouping identifies categorical columns"""
        columns = ['type', 'category', 'class', 'group', 'kind', 'value']

        researcher = DomainResearcher()
        insights = researcher._analyze_column_names(columns)

        assert 'semantic_groups' in insights
        if 'categories' in insights['semantic_groups']:
            categories = insights['semantic_groups']['categories']
            assert len(categories) >= 1

    def test_semantic_grouping_temporal(self):
        """Test semantic grouping identifies temporal columns"""
        columns = ['date', 'time', 'timestamp', 'year', 'month', 'value']

        researcher = DomainResearcher()
        insights = researcher._analyze_column_names(columns)

        assert 'semantic_groups' in insights
        if 'temporal' in insights['semantic_groups']:
            temporal = insights['semantic_groups']['temporal']
            assert len(temporal) >= 1

    def test_cache_functionality(self):
        """Test that research results are cached"""
        config = {'cache_results': True}
        researcher = DomainResearcher(config)

        # Add something to cache
        researcher.research_cache['test_domain'] = {'strategy': 'test'}

        # Should retrieve from cache
        assert 'test_domain' in researcher.research_cache
        assert researcher.research_cache['test_domain']['strategy'] == 'test'
