"""Build and validate submissions from the resolved problem.

What gets written is decided by the metric (``problem.output``), never by the
placeholder values in ``sample_submission.csv``. The sample file is only used to
validate the result: same columns, same row count, same IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.genML.pipeline.problem import ResolvedProblem
from src.genML.pipeline.targets import decode_classes, inverse_regression

logger = logging.getLogger(__name__)


class SubmissionValidationError(ValueError):
    pass


def _class_column_order(target_columns: List[str], classes: List[Any]) -> List[int]:
    """Map per-class submission columns to class indices by name."""
    by_name = {str(c): i for i, c in enumerate(classes)}
    order = []
    for column in target_columns:
        key = column
        for prefix in ("class_", "Class_", "prob_", "Status_"):
            if key not in by_name and column.startswith(prefix):
                key = column[len(prefix):]
        if key not in by_name:
            raise SubmissionValidationError(
                f"Submission column '{column}' does not match any class {classes}"
            )
        order.append(by_name[key])
    return order


def predict_outputs(problem: ResolvedProblem, model, X_test: np.ndarray,
                    transform_info: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Return {submission column: values} plus summary statistics."""
    target_columns = list(problem.submission.get("target_columns") or [problem.target])
    output = problem.output
    stats: Dict[str, Any] = {"output": output}

    if not problem.is_classification:
        values = inverse_regression(np.asarray(model.predict(X_test), dtype=float), transform_info)
        stats.update({"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values))})
        return {target_columns[0]: values}, stats

    classes = problem.classes or transform_info.get("classes")
    if not classes:
        raise SubmissionValidationError("Class labels are unknown; rerun feature engineering")

    if output == "label":
        codes = np.asarray(model.predict(X_test)).astype(int).ravel()
        labels = decode_classes(codes, classes)
        stats["label_counts"] = {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()}
        return {target_columns[0]: labels}, stats

    if not hasattr(model, "predict_proba"):
        raise SubmissionValidationError(
            f"Metric '{problem.metric}' needs probabilities but {type(model).__name__} has no predict_proba"
        )
    proba = np.asarray(model.predict_proba(X_test), dtype=float)

    if output == "top_k":
        k = problem.metric_obj.top_k or 3
        top = np.argsort(-proba, axis=1)[:, :k]
        lookup = np.asarray([str(c) for c in classes], dtype=object)
        joined = np.array([" ".join(row) for row in lookup[top]], dtype=object)
        return {target_columns[0]: joined}, {"output": output, "k": k}

    if problem.task == "binary" and len(target_columns) == 1:
        positive = proba[:, 1]
        stats.update({
            "positive_class": classes[1],
            "min": float(positive.min()),
            "max": float(positive.max()),
            "mean": float(positive.mean()),
        })
        return {target_columns[0]: positive}, stats

    # One probability column per class (multiclass log loss / AUC)
    order = _class_column_order(target_columns, classes)
    stats["classes"] = classes
    return {column: proba[:, index] for column, index in zip(target_columns, order)}, stats


def build_submission(problem: ResolvedProblem, model, X_test: np.ndarray, test_df: pd.DataFrame,
                     transform_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    outputs, stats = predict_outputs(problem, model, X_test, transform_info)
    columns: Dict[str, Any] = {}
    if problem.id_column:
        if problem.id_column not in test_df.columns:
            raise SubmissionValidationError(f"ID column '{problem.id_column}' is not in test.csv")
        columns[problem.id_column] = test_df[problem.id_column].to_numpy()
    columns.update(outputs)
    return pd.DataFrame(columns), stats


def validate_submission(submission: pd.DataFrame, problem: ResolvedProblem,
                        sample_path: Optional[Path]) -> List[str]:
    """Check a submission against the sample file; raise on hard errors.

    Returns a list of warnings for softer issues.
    """
    warnings: List[str] = []
    values = submission.drop(columns=[problem.id_column] if problem.id_column else [])
    if values.isna().any().any():
        raise SubmissionValidationError("Submission contains missing predictions")

    if problem.output in ("probability", "class_probabilities"):
        numeric = values.to_numpy(dtype=float)
        if numeric.min() < 0 or numeric.max() > 1:
            raise SubmissionValidationError("Probabilities fall outside [0, 1]")
        if numeric.std() == 0:
            warnings.append("All predicted probabilities are identical")

    if sample_path is None or not Path(sample_path).exists():
        warnings.append("No sample submission to validate against")
        return warnings

    sample = pd.read_csv(sample_path)
    if list(sample.columns) != list(submission.columns):
        raise SubmissionValidationError(
            f"Columns {list(submission.columns)} do not match sample {list(sample.columns)}"
        )
    if len(sample) != len(submission):
        raise SubmissionValidationError(f"{len(submission)} rows but the sample has {len(sample)}")
    if problem.id_column:
        if set(sample[problem.id_column]) != set(submission[problem.id_column]):
            raise SubmissionValidationError("Submission IDs do not match the sample submission IDs")
        if not (sample[problem.id_column].to_numpy() == submission[problem.id_column].to_numpy()).all():
            warnings.append("IDs are in a different order than the sample (usually accepted)")
    return warnings
