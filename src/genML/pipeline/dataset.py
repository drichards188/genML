"""Dataset loading utilities for the genML pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.genML.pipeline import config, context
from src.genML.pipeline.problem import (
    find_sample_submission,
    legacy_spec,
    locate_data_dir,
    resolve_problem,
)

logger = logging.getLogger(__name__)


def load_dataset() -> str:
    """
    Load train/test data, persist them as pickle files for downstream stages,
    and emit a JSON summary.

    The data comes from the requested problem spec (see pipeline/problem.py).
    Without a spec, train.csv/test.csv are discovered in datasets/current/,
    datasets/ or the working directory and every problem field is inferred.

    Returns:
        JSON string with dataset summary or error payload.
    """
    # Check if ingestion pipeline is configured
    if config.INGESTION_CONFIG is not None:
        logger.info("Using ingestion pipeline mode")
        return _load_via_ingestion_pipeline()

    return _load_via_problem_spec()


def _load_via_ingestion_pipeline() -> str:
    """Load data using the flexible ingestion pipeline."""
    try:
        from .ingestion import run_ingestion_pipeline

        result_json = run_ingestion_pipeline(config.INGESTION_CONFIG)

        # Parse result to save report
        result = json.loads(result_json)
        with open(config.REPORTS_DIR / "data_exploration_report.json", "w") as f:
            json.dump(result, f, indent=2)

        return result_json

    except Exception as exc:
        logger.exception("Error in ingestion pipeline")
        return json.dumps(
            {
                "error": f"Ingestion pipeline failed: {exc}",
                "status": "failed",
                "pipeline_mode": "ingestion",
            }
        )


def _load_via_problem_spec() -> str:
    """Load data for the requested problem spec (or legacy CSV discovery).

    Resolves the problem spec against the data, records every inferred field,
    and saves train/test pickles plus ``resolved_spec.json`` for later stages.
    """
    try:
        spec = context.get_requested_spec() or legacy_spec()
        try:
            data_dir = locate_data_dir(spec)
        except FileNotFoundError as exc:
            return json.dumps(
                {
                    "error": str(exc),
                    "status": "failed",
                    "pipeline_mode": "problem_spec" if context.get_requested_spec() else "legacy_csv",
                }
            )

        logger.info("Using dataset from: %s", data_dir)
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        sample_path = find_sample_submission(data_dir)
        sample_df = pd.read_csv(sample_path) if sample_path else None

        problem = resolve_problem(spec, train_df, test_df, sample_df, data_dir=data_dir)
        context.set_active_problem(problem)
        target_col = problem.target

        target_rate = None
        target_series = train_df[target_col]
        try:
            if problem.is_classification:
                target_rate = {str(k): float(v) for k, v in target_series.value_counts(normalize=True).items()}
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
            "problem": problem.name,
            "data_dir": str(data_dir),
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "train_columns": list(train_df.columns),
            "test_columns": list(test_df.columns),
            "target_column": target_col,
            "id_column": problem.id_column,
            "task": problem.task,
            "metric": problem.metric,
            "submission_output": problem.output,
            "inferred": problem.inferred,
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
            json.dump(summary, f, indent=2, default=str)

        return json.dumps(summary, indent=2, default=str)

    except Exception as exc:
        logger.exception("Error loading data")
        return json.dumps(
            {
                "error": f"Error loading data: {exc}",
                "status": "failed",
                "pipeline_mode": "problem_spec",
            }
        )
