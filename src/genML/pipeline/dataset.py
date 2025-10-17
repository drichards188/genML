"""Dataset loading utilities for the genML pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.genML.pipeline import config

logger = logging.getLogger(__name__)


def _discover_dataset_paths() -> tuple[Path | None, Path | None]:
    """Locate train/test CSV files in supported locations."""
    search_paths = [
        Path("datasets/current"),
        Path("datasets"),
        Path("."),
    ]

    for base_path in search_paths:
        train_path = base_path / "train.csv"
        test_path = base_path / "test.csv"

        if train_path.exists() and test_path.exists():
            logger.info("Using dataset from: %s", base_path)
            return train_path, test_path

    return None, None


def load_dataset() -> str:
    """
    Load train/test CSVs, persist them as pickle files for downstream stages,
    and emit a JSON summary.

    Returns:
        JSON string with dataset summary or error payload.
    """
    try:
        train_path, test_path = _discover_dataset_paths()

        if not train_path or not test_path:
            return json.dumps(
                {
                    "error": (
                        "train.csv and test.csv files not found. "
                        "Please place dataset files in datasets/current/ or project root directory."
                    ),
                    "status": "failed",
                }
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        train_only_cols = train_cols - test_cols

        id_patterns = {"id", "ID", "Id", "index", "Index"}
        potential_targets = [col for col in train_only_cols if col not in id_patterns]
        target_col = potential_targets[0] if potential_targets else None

        target_rate = None
        if target_col and target_col in train_df.columns:
            target_series = train_df[target_col]
            try:
                if target_series.dtype in ["object", "category"] or target_series.nunique() <= 20:
                    target_rate = target_series.value_counts(normalize=True).to_dict()
                else:
                    target_rate = {
                        "mean": float(target_series.mean()),
                        "std": float(target_series.std()),
                        "min": float(target_series.min()),
                        "max": float(target_series.max()),
                    }
            except Exception:
                target_rate = None

        summary = {
            "status": "success",
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "train_columns": list(train_df.columns),
            "test_columns": list(test_df.columns),
            "target_column": target_col,
            "missing_values": {
                "train": train_df.isnull().sum().to_dict(),
                "test": test_df.isnull().sum().to_dict(),
            },
            "target_stats": target_rate,
            "train_head": train_df.head().to_dict("records"),
            "test_head": test_df.head().to_dict("records"),
        }

        train_df.to_pickle(config.DATA_DIR / "train_data.pkl")
        test_df.to_pickle(config.DATA_DIR / "test_data.pkl")

        with open(config.REPORTS_DIR / "data_exploration_report.json", "w") as f:
            json.dump(summary, f, indent=2)

        return json.dumps(summary, indent=2)

    except Exception as exc:
        logger.exception("Error loading data")
        return json.dumps(
            {
                "error": f"Error loading data: {exc}",
                "status": "failed",
            }
        )
