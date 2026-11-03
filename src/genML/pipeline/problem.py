"""Problem specifications: declare what is known about a problem, infer the rest.

A problem lives in ``problems/<name>/problem.yaml``. Any field left out is
inferred from the data, and every inference is recorded with its reason in the
resolved spec (``resolved_spec.json``) so guesses are visible and overridable.

Example::

    name: fraud-s6e11
    source: {type: kaggle, competition: playground-series-s6e11}
    target: is_fraud
    id_column: id
    task: binary
    metric: roc_auc
    columns:
      categorical: [merchant_category, auth_method, device_type]
    models: [XGBoost, CatBoost, LightGBM]
    budget: {n_trials: 20, timeout_per_model: 900}
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.genML.pipeline.inference import ID_NAME_PATTERNS, infer_target_column, infer_task_type
from src.genML.pipeline.metrics import Metric, default_metric_for, get_metric

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROBLEMS_DIR = PROJECT_ROOT / "problems"
DATASETS_DIR = PROJECT_ROOT / "datasets"
KAGGLEHUB_CACHE = Path.home() / ".cache" / "kagglehub" / "competitions"

VALID_TASKS = ("binary", "multiclass", "regression")
COLUMN_ROLES = ("categorical", "numerical", "text", "datetime", "drop")
SPEC_FIELDS = (
    "name", "description", "source", "target", "id_column", "task", "metric",
    "positive_class", "cv", "submission", "columns", "models", "budget",
    "feature_engineering",
)


@dataclass
class ProblemSpec:
    """What the user declares about a problem. ``None`` means "infer it"."""

    name: str
    source: Dict[str, Any] = field(default_factory=lambda: {"type": "csv"})
    description: str = ""
    target: Optional[str] = None
    id_column: Optional[str] = None
    task: Optional[str] = None
    metric: Optional[str] = None
    positive_class: Optional[Any] = None
    cv: Dict[str, Any] = field(default_factory=dict)
    submission: Dict[str, Any] = field(default_factory=dict)
    columns: Dict[str, List[str]] = field(default_factory=dict)
    models: Optional[List[str]] = None
    budget: Dict[str, Any] = field(default_factory=dict)
    feature_engineering: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemSpec":
        unknown = set(data) - set(SPEC_FIELDS)
        if unknown:
            raise ValueError(f"Unknown problem spec field(s): {sorted(unknown)}")
        if "name" not in data:
            raise ValueError("Problem spec needs a 'name'")
        spec = cls(**{key: value for key, value in data.items() if value is not None})
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.task is not None and self.task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{self.task}'")
        if self.metric is not None:
            metric = get_metric(self.metric)
            if self.task is not None and self.task not in metric.tasks:
                raise ValueError(f"metric '{self.metric}' does not support task '{self.task}'")
        unknown_roles = set(self.columns) - set(COLUMN_ROLES)
        if unknown_roles:
            raise ValueError(f"Unknown column role(s) {sorted(unknown_roles)}; use {COLUMN_ROLES}")
        source_type = self.source.get("type", "csv")
        if source_type not in ("csv", "kaggle"):
            raise ValueError(f"source.type must be 'csv' or 'kaggle', got '{source_type}'")
        if source_type == "kaggle" and not self.source.get("competition"):
            raise ValueError("source.type 'kaggle' needs source.competition")

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, {}, [], "")}


@dataclass
class ResolvedProblem:
    """A fully specified problem: declared fields plus recorded inferences."""

    name: str
    data_dir: str
    target: str
    id_column: Optional[str]
    task: str
    metric: str
    output: str
    positive_class: Optional[Any]
    cv: Dict[str, Any]
    submission: Dict[str, Any]
    column_types: Dict[str, str]
    drop_columns: List[str]
    models: Optional[List[str]]
    budget: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    source: Dict[str, Any]
    inferred: Dict[str, str] = field(default_factory=dict)
    classes: Optional[List[Any]] = None

    @property
    def metric_obj(self) -> Metric:
        return get_metric(self.metric)

    @property
    def is_classification(self) -> bool:
        return self.task in ("binary", "multiclass")

    @property
    def problem_type(self) -> str:
        """Legacy two-way label used by the training code."""
        return "classification" if self.is_classification else "regression"

    @property
    def scoring(self):
        return self.metric_obj.scoring_for(self.task)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolvedProblem":
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "ResolvedProblem":
        with open(path) as handle:
            return cls.from_dict(json.load(handle))


# --------------------------------------------------------------------------- #
# Loading specs
# --------------------------------------------------------------------------- #

def problem_path(name_or_path: str) -> Path:
    """Map a problem name or a YAML path to the spec file."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate if candidate.is_absolute() else (Path.cwd() / candidate)
    return PROBLEMS_DIR / name_or_path / "problem.yaml"


def load_problem(name_or_path: str) -> ProblemSpec:
    path = problem_path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"No problem spec at {path}")
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    return ProblemSpec.from_dict(data)


def list_problems() -> List[str]:
    if not PROBLEMS_DIR.exists():
        return []
    return sorted(p.parent.name for p in PROBLEMS_DIR.glob("*/problem.yaml"))


def save_problem(spec: ProblemSpec, path: Optional[Path] = None) -> Path:
    path = path or (PROBLEMS_DIR / spec.name / "problem.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump(spec.to_dict(), handle, sort_keys=False)
    return path


# --------------------------------------------------------------------------- #
# Locating data
# --------------------------------------------------------------------------- #

LEGACY_SEARCH_PATHS = (Path("datasets/current"), Path("datasets"), Path("."))


def _has_train_test(directory: Path) -> bool:
    return (directory / "train.csv").exists() and (directory / "test.csv").exists()


def locate_data_dir(spec: ProblemSpec, download: bool = False) -> Path:
    """Find the folder holding train.csv/test.csv for a spec.

    Order: an explicit ``source.path``; ``datasets/<name>/``; for Kaggle
    sources the kagglehub cache (downloading when ``download`` is set); for
    specs without a path the legacy search (datasets/current, datasets, cwd).
    """
    source = spec.source or {"type": "csv"}
    explicit = source.get("path")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path) if (PROJECT_ROOT / path).exists() else Path.cwd() / path
        if not _has_train_test(path):
            raise FileNotFoundError(f"source.path {path} has no train.csv/test.csv")
        return path

    named = DATASETS_DIR / spec.name
    if _has_train_test(named):
        return named

    if source.get("type") == "kaggle":
        competition = source["competition"]
        cached = KAGGLEHUB_CACHE / competition
        if _has_train_test(cached):
            return cached
        if download:
            from src.genML.pipeline.kaggle import download_competition

            return download_competition(competition)
        raise FileNotFoundError(
            f"Data for Kaggle competition '{competition}' not found in {named} or {cached}. "
            "Run with download enabled (python src/genML/main.py --problem NAME --download)."
        )

    for base in LEGACY_SEARCH_PATHS:
        if _has_train_test(base):
            return base.resolve()
    raise FileNotFoundError(
        "train.csv and test.csv not found. Place them in datasets/current/ or declare source.path."
    )


def find_sample_submission(data_dir: Path) -> Optional[Path]:
    exact = data_dir / "sample_submission.csv"
    if exact.exists():
        return exact
    matches = sorted(data_dir.glob("*submission*.csv"))
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def _infer_id_column(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     sample_df: Optional[pd.DataFrame]) -> tuple[Optional[str], str]:
    if sample_df is not None and len(sample_df.columns) > 0:
        first = sample_df.columns[0]
        if first in test_df.columns:
            return first, "first column of sample_submission.csv"
    for column in test_df.columns:
        if column in ID_NAME_PATTERNS or column.lower().endswith("id"):
            if test_df[column].is_unique:
                return column, f"unique test column named like an ID ('{column}')"
    return None, "no ID column found; submissions will use row order"


def resolve_problem(
    spec: ProblemSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
) -> ResolvedProblem:
    """Fill every unset field of ``spec`` and record why."""
    inferred: Dict[str, str] = {}

    # Target
    target = spec.target
    if target is None:
        target, reason = infer_target_column(list(train_df.columns), list(test_df.columns))
        if target is None:
            raise ValueError(f"Could not infer the target column ({reason}); declare 'target'")
        inferred["target"] = reason
    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' is not in train.csv")

    # ID column
    id_column = spec.id_column
    if id_column is None:
        id_column, reason = _infer_id_column(train_df, test_df, sample_df)
        inferred["id_column"] = reason

    # Task
    task = spec.task
    if task is None:
        task, reason = infer_task_type(train_df[target])
        inferred["task"] = reason

    # Sample submission shape (used for metric fallback and output columns)
    sample_value_type = None
    if sample_df is not None and len(sample_df.columns) >= 2:
        from src.genML.submission_formatter import SubmissionFormatter

        sample_value_type = SubmissionFormatter()._determine_value_type(sample_df[sample_df.columns[1]])

    # Metric
    metric_name = spec.metric
    if metric_name is None:
        metric, reason = default_metric_for(task, sample_value_type)
        metric_name = metric.name
        inferred["metric"] = reason
    metric = get_metric(metric_name)
    if task not in metric.tasks:
        raise ValueError(f"Metric '{metric_name}' does not support task '{task}'")

    # Submission layout
    submission = dict(spec.submission)
    output = submission.pop("output", None) or metric.output
    if "output" not in spec.submission:
        inferred["output"] = f"from metric '{metric_name}' ({output})"
    if "target_columns" not in submission:
        if sample_df is not None and len(sample_df.columns) >= 2:
            submission["target_columns"] = [c for c in sample_df.columns if c != id_column] or [target]
            inferred["submission.target_columns"] = "non-ID columns of sample_submission.csv"
        else:
            submission["target_columns"] = [target]
            inferred["submission.target_columns"] = "no sample submission; using the target name"
    if sample_df is not None:
        submission.setdefault("expected_rows", int(len(sample_df)))

    # CV
    cv = {"n_splits": 5, "shuffle": True, "random_state": 42, **spec.cv}
    if "method" not in cv:
        cv["method"] = "stratified_kfold" if task in ("binary", "multiclass") else "kfold"
        inferred["cv.method"] = f"default for {task}"

    # Column roles from the spec (everything else is typed by the analyzer)
    column_types: Dict[str, str] = {}
    for role in ("categorical", "numerical", "text", "datetime"):
        for column in spec.columns.get(role, []):
            column_types[column] = role
    drop_columns = list(spec.columns.get("drop", []))
    missing = [c for c in list(column_types) + drop_columns if c not in train_df.columns]
    if missing:
        logger.warning("Spec references columns not in train.csv: %s", missing)

    return ResolvedProblem(
        name=spec.name,
        data_dir=str(data_dir) if data_dir else "",
        target=target,
        id_column=id_column,
        task=task,
        metric=metric_name,
        output=output,
        positive_class=spec.positive_class,
        cv=cv,
        submission=submission,
        column_types=column_types,
        drop_columns=drop_columns,
        models=spec.models,
        budget=dict(spec.budget),
        feature_engineering=dict(spec.feature_engineering),
        source=dict(spec.source),
        inferred=inferred,
    )


def legacy_spec(name: str = "current") -> ProblemSpec:
    """Spec used when no problem is named: discover CSVs and infer everything."""
    return ProblemSpec(name=name, source={"type": "csv"})
