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

# The loaded tables travel between stages as files in the run's data/ folder:
# ``<name>_data.parquet`` (zstd), or ``<name>_data.pkl`` when a frame cannot be stored as Parquet and
# for runs made before 2026-11-16. A pruned run keeps only ``test_ids.parquet`` (the id column).
STAGE_TABLES = ("train", "test")
TEST_IDS_FILE = "test_ids.parquet"


def _stage_path(data_dir: Path, name: str, suffix: str) -> Path:
    return Path(data_dir) / f"{name}_data{suffix}"


def save_stage_tables(train_df: pd.DataFrame, test_df: pd.DataFrame, data_dir: Path | None = None) -> str:
    """Write the train and test tables for the next stages; returns the format used.

    Both go as Parquet, or both as pickle when either cannot be stored as Parquet (non-text column
    names, mixed-type columns): one format per run, so train and test read back alike."""
    data_dir = Path(data_dir or config.DATA_DIR)
    frames = {"train": train_df, "test": test_df}
    written = []
    try:
        for name, frame in frames.items():
            path = _stage_path(data_dir, name, ".parquet")
            frame.to_parquet(path, compression="zstd")
            written.append(path)
        fmt = "parquet"
    except (ValueError, TypeError, NotImplementedError, ImportError) as exc:
        logger.warning("Stage tables stored as pickle: Parquet refused them (%s)", exc)
        for path in written:
            path.unlink(missing_ok=True)
        for name, frame in frames.items():
            frame.to_pickle(_stage_path(data_dir, name, ".pkl"))
        fmt = "pickle"
    for name in frames:   # never leave the other format's stale copy behind
        stale = _stage_path(data_dir, name, ".pkl" if fmt == "parquet" else ".parquet")
        stale.unlink(missing_ok=True)
    return fmt


def _missing_text_as_nan(frame: pd.DataFrame) -> pd.DataFrame:
    """Parquet gives missing text back as None; the stages were built on CSV reads, which give NaN."""
    for column in frame.columns[frame.dtypes == object]:
        series = frame[column]
        if series.isna().any():
            frame[column] = series.mask(series.isna(), float("nan"))
    return frame


def load_stage_table(name: str, data_dir: Path | None = None, ids_only_ok: bool = False) -> pd.DataFrame:
    """A table written by ``save_stage_tables`` (or an older run's pickle).

    ``ids_only_ok``: for the test table, a pruned run's id column is enough (submissions need only ids)."""
    if name not in STAGE_TABLES:
        raise ValueError(f"name must be one of {STAGE_TABLES}")
    data_dir = Path(data_dir or config.DATA_DIR)
    parquet, pickle = _stage_path(data_dir, name, ".parquet"), _stage_path(data_dir, name, ".pkl")
    if parquet.exists():
        return _missing_text_as_nan(pd.read_parquet(parquet))
    if pickle.exists():
        return pd.read_pickle(pickle)
    ids = data_dir / TEST_IDS_FILE
    if name == "test" and ids_only_ok and ids.exists():
        return pd.read_parquet(ids)
    hint = " (this run was pruned: only its predictions, reports and test ids were kept)" \
        if (data_dir.parent / "pruned.json").exists() else ""
    raise FileNotFoundError(f"No {name} table in {data_dir}{hint}")


def stage_table_files(data_dir: Path, name: str) -> list:
    """The files that hold a stage table in ``data_dir`` (empty when there are none)."""
    return [p for p in (_stage_path(data_dir, name, ".parquet"), _stage_path(data_dir, name, ".pkl"))
            if p.exists()]


def load_dataset() -> str:
    """
    Load train/test data, persist them (Parquet) for downstream stages,
    and emit a JSON summary.

    The data comes from the requested problem spec (see pipeline/problem.py).
    Without a spec, train.csv/test.csv are discovered in datasets/current/,
    datasets/ or the working directory and every problem field is inferred.

    Returns:
        JSON string with dataset summary or error payload.
    """
    return _load_via_problem_spec()


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
        tables_report = None
        if spec.tables:
            from src.genML.pipeline.tables import assemble

            run = config.ACTIVE_RUN
            problem_dir = run.problem_dir if run is not None else config.OUTPUTS_DIR
            train_df, test_df, tables_report = assemble(spec.tables, data_dir, problem_dir / "tables_cache")
            print(f"Tables: {len(spec.tables)} joined into train {train_df.shape}, test {test_df.shape}"
                  + (" (cached)" if tables_report.get("cached") else ""))
        else:
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
            "tables": tables_report,
            "train_head": train_df.head().to_dict("records"),
            "test_head": test_df.head().to_dict("records"),
        }

        save_stage_tables(train_df, test_df)

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
