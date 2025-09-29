"""
Domain Knowledge Research Agent

This module provides an intelligent agent that can research domain-specific
feature engineering strategies by analyzing dataset characteristics and
searching for relevant information online.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import re
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DomainResearcher:
    """
    Intelligent agent for researching domain-specific feature engineering strategies.

    This class analyzes dataset characteristics and column names to infer the
    problem domain, then researches appropriate feature engineering techniques
    for that domain.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the domain researcher.

        Args:
            config: Configuration dictionary for research parameters
        """
        self.config = config or {}
        self.domain_patterns = self._load_domain_patterns()
        self.research_cache = {}

        # Configuration
        self.enable_web_research = self.config.get('enable_web_research', True)
        self.max_search_results = self.config.get('max_search_results', 5)
        self.cache_results = self.config.get('cache_results', True)

    def analyze_domain(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """
        Analyze dataset to determine problem domain and suggest feature strategies.

        Args:
            df: DataFrame to analyze
            analysis_results: Results from DataTypeAnalyzer

        Returns:
            Dictionary containing domain analysis and feature recommendations
        """
        logger.info("Analyzing dataset domain for feature engineering strategies")

        domain_analysis = {
            'detected_domains': [],
            'confidence_scores': {},
            'feature_strategies': {},
            'column_insights': {},
            'research_queries': [],
            'recommendations': []
        }

        # Analyze column names for domain clues
        column_insights = self._analyze_column_names(df.columns.tolist())
        domain_analysis['column_insights'] = column_insights

        # Detect potential domains
        domains = self._detect_domains(df, column_insights, analysis_results)
        domain_analysis['detected_domains'] = domains

        # Generate feature strategies for detected domains
        for domain in domains:
            strategies = self._get_domain_strategies(domain, df, analysis_results)
            domain_analysis['feature_strategies'][domain] = strategies

        # Generate research queries for web search
        research_queries = self._generate_research_queries(domains, column_insights)
        domain_analysis['research_queries'] = research_queries

        # Generate recommendations
        recommendations = self._generate_domain_recommendations(domain_analysis)
        domain_analysis['recommendations'] = recommendations

        return domain_analysis

    def research_feature_strategies(self, domain_analysis: Dict, web_search_func=None) -> Dict[str, Any]:
        """
        Research domain-specific feature engineering strategies.

        Args:
            domain_analysis: Results from analyze_domain
            web_search_func: Optional function for web search (e.g., WebSearch tool)

        Returns:
            Dictionary containing researched strategies and insights
        """
        if not self.enable_web_research or not web_search_func:
            logger.info("Web research disabled or no search function provided")
            return {'research_performed': False, 'strategies': {}}

        research_results = {
            'research_performed': True,
            'strategies': {},
            'sources': [],
            'insights': []
        }

        # Research each detected domain
        for domain in domain_analysis.get('detected_domains', []):
            if domain in self.research_cache and self.cache_results:
                logger.info(f"Using cached research for domain: {domain}")
                research_results['strategies'][domain] = self.research_cache[domain]
                continue

            try:
                # Generate search query
                query = f"feature engineering {domain} machine learning best practices"
                logger.info(f"Researching domain: {domain} with query: {query}")

                # Perform web search
                search_result = web_search_func(
                    query=query,
                    prompt=f"Extract feature engineering techniques and best practices for {domain} datasets. Focus on practical techniques that improve model performance."
                )

                # Parse and store results
                strategies = self._parse_research_results(search_result, domain)
                research_results['strategies'][domain] = strategies

                # Cache results
                if self.cache_results:
                    self.research_cache[domain] = strategies

            except Exception as e:
                logger.warning(f"Failed to research domain {domain}: {e}")
                research_results['strategies'][domain] = {'error': str(e)}

        return research_results

    def _load_domain_patterns(self) -> Dict[str, Dict]:
        """Load predefined domain patterns and indicators."""
        return {
            'finance': {
                'keywords': ['price', 'cost', 'amount', 'payment', 'balance', 'transaction', 'credit', 'debit', 'loan', 'interest'],
                'patterns': [r'.*price.*', r'.*cost.*', r'.*amount.*', r'.*balance.*'],
                'strategies': ['log_transforms', 'ratios', 'moving_averages', 'volatility_measures']
            },
            'healthcare': {
                'keywords': ['age', 'weight', 'height', 'blood', 'pressure', 'heart', 'disease', 'diagnosis', 'treatment'],
                'patterns': [r'.*age.*', r'.*weight.*', r'.*height.*', r'.*blood.*', r'.*bmi.*'],
                'strategies': ['bmi_calculation', 'age_groups', 'vital_ratios', 'medical_scores']
            },
            'ecommerce': {
                'keywords': ['product', 'category', 'brand', 'rating', 'review', 'purchase', 'customer', 'order'],
                'patterns': [r'.*product.*', r'.*category.*', r'.*brand.*', r'.*rating.*'],
                'strategies': ['categorical_encoding', 'rating_aggregations', 'purchase_patterns', 'seasonality']
            },
            'transportation': {
                'keywords': ['speed', 'distance', 'time', 'route', 'vehicle', 'fuel', 'traffic', 'journey'],
                'patterns': [r'.*speed.*', r'.*distance.*', r'.*time.*', r'.*fuel.*'],
                'strategies': ['speed_ratios', 'time_features', 'distance_calculations', 'efficiency_metrics']
            },
            'real_estate': {
                'keywords': ['area', 'room', 'bedroom', 'bathroom', 'floor', 'location', 'neighborhood', 'sqft'],
                'patterns': [r'.*area.*', r'.*room.*', r'.*bedroom.*', r'.*sqft.*', r'.*floor.*'],
                'strategies': ['area_ratios', 'room_density', 'location_encoding', 'size_categories']
            },
            'text_analysis': {
                'keywords': ['text', 'comment', 'review', 'description', 'title', 'content', 'message'],
                'patterns': [r'.*text.*', r'.*comment.*', r'.*review.*', r'.*description.*'],
                'strategies': ['tfidf', 'sentiment_analysis', 'text_length', 'keyword_extraction']
            },
            'time_series': {
                'keywords': ['date', 'time', 'timestamp', 'year', 'month', 'day', 'hour'],
                'patterns': [r'.*date.*', r'.*time.*', r'.*timestamp.*'],
                'strategies': ['lag_features', 'rolling_statistics', 'seasonality', 'time_decomposition']
            }
        }

    def _analyze_column_names(self, columns: List[str]) -> Dict[str, Any]:
        """Analyze column names to extract domain insights."""
        insights = {
            'domain_keywords': {},
            'pattern_matches': {},
            'semantic_groups': {},
            'data_types_inferred': {}
        }

        # Check for domain keywords
        for domain, config in self.domain_patterns.items():
            keyword_matches = []
            for col in columns:
                col_lower = col.lower()
                for keyword in config['keywords']:
                    if keyword in col_lower:
                        keyword_matches.append((col, keyword))

            if keyword_matches:
                insights['domain_keywords'][domain] = keyword_matches

        # Check for regex patterns
        for domain, config in self.domain_patterns.items():
            pattern_matches = []
            for col in columns:
                col_lower = col.lower()
                for pattern in config['patterns']:
                    if re.search(pattern, col_lower):
                        pattern_matches.append((col, pattern))

            if pattern_matches:
                insights['pattern_matches'][domain] = pattern_matches

        # Group semantically similar columns
        semantic_groups = {
            'identifiers': [col for col in columns if any(id_word in col.lower()
                           for id_word in ['id', 'key', 'index', 'identifier'])],
            'measurements': [col for col in columns if any(measure_word in col.lower()
                            for measure_word in ['size', 'length', 'width', 'height', 'weight', 'area'])],
            'categories': [col for col in columns if any(cat_word in col.lower()
                          for cat_word in ['type', 'category', 'class', 'group', 'kind'])],
            'temporal': [col for col in columns if any(time_word in col.lower()
                        for time_word in ['date', 'time', 'year', 'month', 'day', 'hour'])]
        }

        insights['semantic_groups'] = {k: v for k, v in semantic_groups.items() if v}

        return insights

    def _detect_domains(self, df: pd.DataFrame, column_insights: Dict, analysis_results: Dict) -> List[str]:
        """Detect the most likely domains for the dataset."""
        domain_scores = {}

        # Score based on keyword matches
        for domain, matches in column_insights.get('domain_keywords', {}).items():
            domain_scores[domain] = len(matches) * 2  # Weight keyword matches highly

        # Score based on pattern matches
        for domain, matches in column_insights.get('pattern_matches', {}).items():
            domain_scores[domain] = domain_scores.get(domain, 0) + len(matches)

        # Additional scoring based on data characteristics
        if 'text_columns' in analysis_results and analysis_results['text_columns']:
            domain_scores['text_analysis'] = domain_scores.get('text_analysis', 0) + len(analysis_results['text_columns'])

        if 'datetime_columns' in analysis_results and analysis_results['datetime_columns']:
            domain_scores['time_series'] = domain_scores.get('time_series', 0) + len(analysis_results['datetime_columns']) * 2

        # Sort by score and return top domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

        # Return domains with score > 0
        return [domain for domain, score in sorted_domains if score > 0]

    def _get_domain_strategies(self, domain: str, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Get feature engineering strategies for a specific domain."""
        if domain not in self.domain_patterns:
            return {}

        strategies = {
            'recommended_techniques': self.domain_patterns[domain]['strategies'],
            'applicable_columns': [],
            'specific_recommendations': []
        }

        # Find columns that match domain patterns
        domain_config = self.domain_patterns[domain]
        applicable_columns = []

        for col in df.columns:
            col_lower = col.lower()
            # Check keywords
            if any(keyword in col_lower for keyword in domain_config['keywords']):
                applicable_columns.append(col)
            # Check patterns
            elif any(re.search(pattern, col_lower) for pattern in domain_config['patterns']):
                applicable_columns.append(col)

        strategies['applicable_columns'] = applicable_columns

        # Generate specific recommendations
        recommendations = []

        if domain == 'finance':
            if any('price' in col.lower() for col in applicable_columns):
                recommendations.append("Consider log transformation for price columns to handle skewness")
                recommendations.append("Create price ratios and percentage changes")

        elif domain == 'healthcare':
            if 'age' in [col.lower() for col in applicable_columns]:
                recommendations.append("Create age groups (child, adult, senior) for better interpretability")
            if any('weight' in col.lower() or 'height' in col.lower() for col in applicable_columns):
                recommendations.append("Calculate BMI if both weight and height are available")

        elif domain == 'real_estate':
            area_cols = [col for col in applicable_columns if 'area' in col.lower() or 'sqft' in col.lower()]
            if area_cols:
                recommendations.append("Create area per room ratios for efficiency metrics")

        elif domain == 'time_series':
            recommendations.append("Extract temporal components (year, month, day, weekday)")
            recommendations.append("Create lag features and rolling statistics for temporal patterns")

        strategies['specific_recommendations'] = recommendations

        return strategies

    def _generate_research_queries(self, domains: List[str], column_insights: Dict) -> List[str]:
        """Generate research queries for web search."""
        queries = []

        for domain in domains:
            # Basic domain query
            queries.append(f"feature engineering {domain} machine learning best practices")

            # Domain-specific queries
            if domain == 'finance':
                queries.append("financial time series feature engineering techniques")
                queries.append("price prediction feature engineering methods")
            elif domain == 'healthcare':
                queries.append("medical data feature engineering machine learning")
                queries.append("healthcare predictive modeling feature selection")
            elif domain == 'text_analysis':
                queries.append("NLP feature engineering text classification")
                queries.append("text preprocessing machine learning features")

        # Limit number of queries
        return queries[:self.max_search_results]

    def _generate_domain_recommendations(self, domain_analysis: Dict) -> List[str]:
        """Generate overall recommendations based on domain analysis."""
        recommendations = []

        detected_domains = domain_analysis.get('detected_domains', [])

        if not detected_domains:
            recommendations.append("No specific domain detected - using generic feature engineering approaches")
            recommendations.append("Consider manual domain specification for better feature strategies")
        else:
            recommendations.append(f"Detected domains: {', '.join(detected_domains)}")
            recommendations.append("Applying domain-specific feature engineering strategies")

        # Add specific recommendations based on column insights
        semantic_groups = domain_analysis.get('column_insights', {}).get('semantic_groups', {})

        if semantic_groups.get('identifiers'):
            recommendations.append("ID columns detected - consider excluding from modeling")

        if semantic_groups.get('temporal'):
            recommendations.append("Temporal columns detected - extract time-based features")

        if semantic_groups.get('measurements'):
            recommendations.append("Measurement columns detected - consider ratios and transformations")

        return recommendations

    def _parse_research_results(self, search_result: str, domain: str) -> Dict[str, Any]:
        """Parse research results and extract actionable insights."""
        # This is a simplified parser - in practice, you'd want more sophisticated NLP
        strategies = {
            'techniques': [],
            'best_practices': [],
            'common_features': [],
            'domain_specific_advice': []
        }

        # Look for common feature engineering keywords in the research
        technique_keywords = [
            'normalization', 'standardization', 'scaling', 'encoding', 'binning',
            'transformation', 'polynomial', 'interaction', 'aggregation', 'ratio'
        ]

        for keyword in technique_keywords:
            if keyword.lower() in search_result.lower():
                strategies['techniques'].append(keyword)

        # Extract domain-specific advice (simplified)
        if domain == 'finance' and ('return' in search_result.lower() or 'volatility' in search_result.lower()):
            strategies['domain_specific_advice'].append("Calculate returns and volatility measures")

        if domain == 'healthcare' and 'patient' in search_result.lower():
            strategies['domain_specific_advice'].append("Consider patient demographic features")

        # Add general insights
        strategies['research_summary'] = search_result[:500] + "..." if len(search_result) > 500 else search_result

        return strategies