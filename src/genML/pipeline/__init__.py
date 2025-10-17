"""Convenience exports for the genML pipeline package."""

from src.genML.pipeline.dataset import load_dataset
from src.genML.pipeline.feature_engineering import engineer_features
from src.genML.pipeline.model_advisor import run_model_selection_advisor
from src.genML.pipeline.prediction import generate_predictions
from src.genML.pipeline.training import train_model_pipeline

__all__ = [
    "engineer_features",
    "generate_predictions",
    "load_dataset",
    "run_model_selection_advisor",
    "train_model_pipeline",
]
