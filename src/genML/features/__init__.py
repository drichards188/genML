"""
Feature Engineering Module

This module provides automated and intelligent feature engineering capabilities
for machine learning pipelines. It includes data type detection, feature processors,
and automated feature selection mechanisms.
"""

from .data_analyzer import DataTypeAnalyzer
from .feature_processors import (
    NumericalProcessor,
    CategoricalProcessor,
    TextProcessor,
    DateTimeProcessor
)
from .feature_engine import AutoFeatureEngine
from .feature_selector import AdvancedFeatureSelector
from .domain_researcher import DomainResearcher

__all__ = [
    'DataTypeAnalyzer',
    'NumericalProcessor',
    'CategoricalProcessor',
    'TextProcessor',
    'DateTimeProcessor',
    'AutoFeatureEngine',
    'AdvancedFeatureSelector',
    'DomainResearcher'
]