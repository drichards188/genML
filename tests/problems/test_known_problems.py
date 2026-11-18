"""Golden tests for every problem declared under problems/.

Kaggle data cannot be committed (competition rules), so each test locates the
data locally (datasets/<name>/ or the kagglehub cache) and skips when absent.

- ``test_inference_matches_spec``: pure inference (no declared fields except the
  metric) must reproduce the spec's target, ID column and task, and the metric
  must imply the right submission output. Fast; runs in the default suite.
- ``test_pipeline_smoke``: an end-to-end run on a small subsample, validated
  against the matching slice of sample_submission.csv. Marked ``slow``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.genML.features.data_analyzer import DataTypeAnalyzer
from src.genML.pipeline import context
from src.genML.pipeline.metrics import get_metric
from src.genML.pipeline.problem import (
    ProblemSpec,
    find_sample_submission,
    list_problems,
    load_problem,
    locate_data_dir,
    resolve_problem,
)
from src.genML.pipeline.tables import main_files, main_table

PROBLEMS = list_problems()

EXPECTED_OUTPUT = {
    "accuracy": "label",
    "roc_auc": "probability",
    "normalized_gini": "probability",
    "amex": "probability",
    "rmse": "value",
    "rmsle": "value",
    "log_loss": "class_probabilities",
    "map_at_3": "top_k",
}


def _data_dir_or_skip(spec: ProblemSpec):
    try:
        return locate_data_dir(spec)
    except FileNotFoundError as exc:
        pytest.skip(f"data for {spec.name} not available locally: {exc}")


def _read_main(spec: ProblemSpec, data_dir, train_rows: int, test_rows: int):
    """First rows of the train and test files that hold the target and ID.

    Single-table problems read train.csv / test.csv. Multi-table problems read their
    main table's files with its ``columns`` / ``drop`` settings, as the table loader
    does; child tables are rolled up per key and never hold the target or ID.
    """
    if spec.tables:
        names, table = main_files(spec.tables), spec.tables[main_table(spec.tables)]
    else:
        names, table = ("train.csv", "test.csv"), {}
    frames = []
    for name, nrows in zip(names, (train_rows, test_rows)):
        frame = pd.read_csv(data_dir / name, nrows=nrows)
        if table.get("columns"):
            frame = frame[[c for c in table["columns"] if c in frame.columns]]
        frames.append(frame.drop(columns=[c for c in table.get("drop") or [] if c in frame.columns]))
    return frames


@pytest.fixture(autouse=True)
def _clean_context():
    context.reset()
    yield
    context.reset()


def test_problems_are_declared():
    assert PROBLEMS, "no problem specs under problems/"


@pytest.mark.parametrize("name", PROBLEMS)
def test_spec_is_valid(name):
    spec = load_problem(name)
    assert spec.target and spec.task and spec.metric, f"{name} should declare target, task and metric"
    metric = get_metric(spec.metric)
    assert spec.task in metric.tasks


@pytest.mark.parametrize("name", PROBLEMS)
def test_inference_matches_spec(name):
    spec = load_problem(name)
    data_dir = _data_dir_or_skip(spec)
    train, test = _read_main(spec, data_dir, train_rows=50_000, test_rows=2_000)
    sample_path = find_sample_submission(data_dir)
    sample = pd.read_csv(sample_path, nrows=2_000) if sample_path else None

    inferred_only = ProblemSpec(name=spec.name, source=spec.source, metric=spec.metric)
    resolved = resolve_problem(inferred_only, train, test, sample, data_dir=data_dir)

    assert resolved.target == spec.target, resolved.inferred.get("target")
    assert resolved.id_column == spec.id_column, resolved.inferred.get("id_column")
    assert resolved.task == spec.task, resolved.inferred.get("task")
    assert resolved.output == EXPECTED_OUTPUT[spec.metric]

    # Short string labels must be categorical, never TF-IDF'd as free text
    features = train.drop(columns=[spec.target])
    analysis = DataTypeAnalyzer().analyze_dataset(features)
    short_label_text = [
        column for column in analysis["text_columns"]
        if features[column].dtype == object
        and features[column].dropna().astype(str).str.split().str.len().mean() < 3
    ]
    assert not short_label_text, f"short labels typed as text: {short_label_text}"
    if spec.id_column in features.columns:
        assert spec.id_column in analysis["id_columns"]


def _subsample(data_dir, spec, tmp_path, n_train=3000, n_test=500):
    """Write a small train/test/sample set with matching IDs."""
    train = pd.read_csv(data_dir / "train.csv", nrows=60_000)
    if spec.task in ("binary", "multiclass") and train[spec.target].nunique() < len(train):
        # Keep every class present in the subsample
        train = (
            train.groupby(spec.target, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(50, n_train // train[spec.target].nunique())),
                                      random_state=0))
        )
    train = train.sample(min(n_train, len(train)), random_state=0)
    test = pd.read_csv(data_dir / "test.csv").head(n_test)
    sample_path = find_sample_submission(data_dir)
    sample = pd.read_csv(sample_path)
    sample = sample[sample[spec.id_column].isin(test[spec.id_column])]

    out = tmp_path / "data"
    out.mkdir()
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    sample.to_csv(out / "sample_submission.csv", index=False)
    return out, sample


@pytest.mark.slow
@pytest.mark.parametrize("name", PROBLEMS)
def test_pipeline_smoke(name, tmp_path):
    from src.genML.pipeline import (
        engineer_features,
        generate_predictions,
        load_dataset,
        train_model_pipeline,
    )

    spec = load_problem(name)
    if spec.tables:
        pytest.skip(f"{name} spans several tables; the smoke subsample only writes train.csv/test.csv")
    data_dir = _data_dir_or_skip(spec)
    small_dir, sample = _subsample(data_dir, spec, tmp_path)
    spec.source = {**spec.source, "path": str(small_dir)}
    spec.models = ["XGBoost"]
    spec.budget = {"tuning": False}
    context.set_requested_spec(spec)

    for stage in (load_dataset, engineer_features, train_model_pipeline):
        result = json.loads(stage())
        assert result.get("status") == "success", result.get("error")

    prediction = json.loads(generate_predictions())
    assert prediction.get("status") == "success", prediction.get("error")

    submission = pd.read_csv(prediction["submission_file"])
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    values = submission[submission.columns[1]]
    expected = EXPECTED_OUTPUT[spec.metric]
    if expected == "probability":
        assert values.between(0, 1).all()
        assert values.nunique() > 2, "AUC submissions must be probabilities, not labels"
    elif expected == "label":
        assert set(values.unique()) <= set(pd.read_csv(data_dir / "train.csv", usecols=[spec.target])[spec.target].unique())
    elif expected == "value":
        assert np.isfinite(values).all() and values.nunique() > 2
