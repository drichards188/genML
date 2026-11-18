"""Tests for the AI-review-to-run bridge (src/genML/experiments.py)."""

import json

import pytest

from src.genML import experiments, jobs, service
from src.genML.experiments import check_patch, deep_merge
from tests.test_jobs_mcp import toy_env  # noqa: F401  (fixture)


# --------------------------------------------------------------------------- #
# Patch checks
# --------------------------------------------------------------------------- #

def test_check_patch_accepts_known_settings():
    patch = {
        "models": ["LightGBM", "xgboost", "GLM-LightGBM"],
        "budget": {"n_trials": 30, "seeds": 3, "tuning_folds": 2, "max_trees": 12000,
                   "search_space": {"LightGBM": {"num_leaves": [2, 8], "learning_rate": [0.01, 0.1]},
                                    "GLM-LightGBM": {"num_leaves": [2, 6]}}},
        "feature_engineering": {"base": "raw", "encodings": {"pairs": False, "count_encode": False,
                                                             "numeric_as_categorical": "all", "digits": "auto",
                                                             "digit_keys": ["cents", "lead2"]}},
        "ensemble": {"method": "stack", "top": 5},
    }
    assert check_patch(patch) == []


@pytest.mark.parametrize("patch, fragment", [
    ({"cv": {"n_splits": 10}}, "'cv' cannot be changed"),
    ({"target": "other"}, "'target' cannot be changed"),
    ({"models": ["GPT-Classifier"]}, "unknown model"),
    ({"models": []}, "non-empty list"),
    ({"budget": {"n_trials": 5000}}, "outside"),
    ({"budget": {"n_trials": "lots"}}, "must be int"),
    ({"budget": {"seeds": True}}, "must be int"),
    ({"budget": {"gpu_hours": 3}}, "not a known setting"),
    ({"ensemble": {"method": "magic"}}, "must be one of"),
    ({"feature_engineering": {"base": "fancy"}}, "must be one of"),
    ({"feature_engineering": {"encodings": {"digit_keys": ["mantissa"]}}}, "digit_keys must be"),
    ({"feature_engineering": {"encodings": {"digits": "sometimes"}}}, "encodings.digits must be"),
    ({"feature_engineering": {"encodings": {"numeric_as_categorical": "most"}}}, "must be one of"),
    ({"feature_engineering": {"encodings": {"max_pairs": -1}}}, "outside"),
    ({"budget": {"search_space": {"LightGBM": {"num_leaves": [1, 8]}}}}, "outside"),
    ({"budget": {"search_space": {"LightGBM": {"num_leaves": [8, 4]}}}}, "low must be below high"),
    ({"budget": {"search_space": {"LightGBM": {"num_leaves": [2.5, 8]}}}}, "whole numbers"),
    ({"budget": {"search_space": {"LightGBM": {"depth": [1, 3]}}}}, "is not tuned for LightGBM"),
    ({"budget": {"search_space": {"Random Forest": {"max_depth": [1, 3]}}}}, "have search spaces"),
    ({"budget": {"max_trees": 50}}, "outside"),
    ({"feature_engineering": {"python": "import os"}}, "not a known setting"),
    ({"feature_engineering": {"encodings": {"original": {"path": "nowhere.csv", "target": "y"}}}},
     "does not exist"),
    ([], "non-empty mapping"),
])
def test_check_patch_rejects(patch, fragment):
    errors = check_patch(patch)
    assert errors and any(fragment in error for error in errors), errors


def test_deep_merge_keeps_unmentioned_settings():
    base = {"budget": {"n_trials": 20, "timeout_per_model": 900},
            "feature_engineering": {"base": "raw", "encodings": {"pairs": "auto", "max_unique": 50}}}
    merged = deep_merge(base, {"budget": {"n_trials": 5}, "feature_engineering": {"encodings": {"pairs": False}}})
    assert merged["budget"] == {"n_trials": 5, "timeout_per_model": 900}
    assert merged["feature_engineering"]["encodings"] == {"pairs": False, "max_unique": 50}
    assert base["budget"]["n_trials"] == 20  # input untouched


# --------------------------------------------------------------------------- #
# Settings carry forward from the best run
# --------------------------------------------------------------------------- #

def test_changes_build_on_the_best_runs_settings(toy_env):
    """A change merges into the settings of the best run, so accepted changes carry forward even when
    problem.yaml drifts after the run (the autopilot's _experiment_settings relies on this)."""
    import yaml

    from src.genML.pipeline.problem import PROBLEMS_DIR, load_problem

    baseline = service.start_run("toy", start_worker=False)
    assert jobs.run_one()["status"] == "completed"
    spec_path = PROBLEMS_DIR / "toy" / "problem.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    spec["models"] = ["Logistic Regression"]
    spec_path.write_text(yaml.safe_dump(spec))

    best = experiments.current_best("toy")
    assert best["run_id"] == baseline["run_id"]
    settings, source = experiments.baseline_settings("toy", best, load_problem("toy"))
    assert source == f"run {baseline['run_id']}"
    assert settings["models"] == ["XGBoost"]
    merged = deep_merge(settings, {"ensemble": {"method": "none"}})
    changes = experiments._changes(settings, merged)
    assert changes and all(change.startswith("ensemble") for change in changes)
    assert experiments.fingerprint(merged) != experiments.fingerprint(settings)
