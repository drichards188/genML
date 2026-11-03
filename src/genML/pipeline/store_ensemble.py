"""Blend experiments across runs of one problem into a new run.

Every run stores its models' out-of-fold and test predictions (pipeline/oof.py).
``ensemble_from_store`` picks the best experiments that share the latest run's
folds, blends them (stack or hill climbing), and writes the result as a new run
folder with its own validated submission, so it can be compared and submitted
like any other run.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.genML.pipeline import config, context
from src.genML.pipeline.ensemble import hill_climb, stack
from src.genML.pipeline.oof import ExperimentStore
from src.genML.pipeline.problem import ResolvedProblem
from src.genML.pipeline.targets import inverse_regression

logger = logging.getLogger(__name__)


def ensemble_from_store(problem_name: str, method: str = "stack", top: int = 10,
                        outputs_root: Optional[Path] = None, notes: str = "") -> Dict[str, Any]:
    from src.genML.pipeline.prediction import generate_predictions
    from src.genML.run_context import RunContext, list_runs, OUTPUTS_ROOT

    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    completed = [r for r in list_runs(problem_name, outputs_root=root) if r.get("status") == "completed"]
    source_run = next((r for r in completed if (root / problem_name / r["run_id"] / "reports" /
                                                 "final_predictions.json").exists()), None)
    if source_run is None:
        raise FileNotFoundError(f"No completed run of '{problem_name}' with stored OOF predictions")
    source_dir = root / problem_name / source_run["run_id"]
    problem = ResolvedProblem.load(source_dir / "reports" / "resolved_spec.json")
    signature = json.loads((source_dir / "reports" / "final_predictions.json").read_text())["fold_signature"]

    problem_dir = root / problem_name
    store = ExperimentStore(problem_dir)
    experiments = [e for e in store.list(signature) if not str(e.meta.get("model", "")).startswith("Ensemble")]
    if len(experiments) < 2:
        raise ValueError(f"Need at least 2 stored experiments with folds {signature}; found {len(experiments)}")

    metric = problem.metric_obj
    y, transform_info = store.load_targets(signature)
    fold_ids = np.load(problem_dir / f"folds_{signature}.npy")
    n_classes = len(problem.classes) if problem.is_classification else 0
    task = problem.task

    if task == "regression":
        y_eval = inverse_regression(y.astype(float), transform_info)

        def native(pred):
            return metric.score(y_eval, inverse_regression(pred, transform_info), task)
    else:
        def native(pred):
            return metric.score(y, pred, task)

    ranked = sorted(experiments, key=lambda e: metric.signed(e.meta["oof_score"]), reverse=True)[:top]
    oofs, tests = [e.oof for e in ranked], [e.test for e in ranked]
    if method == "stack":
        blend_oof, blend_test, info = stack(y, oofs, tests, fold_ids, task, n_classes)
    elif method == "hill_climb":
        blend_oof, blend_test, info = hill_climb(oofs, tests, lambda p: metric.signed(native(p)))
    else:
        raise ValueError("method must be 'stack' or 'hill_climb'")
    score = native(blend_oof)
    best_single = ranked[0]
    info.update({"members": [e.experiment_id for e in ranked], "oof_score": score,
                 "best_single": {"experiment_id": best_single.experiment_id,
                                 "oof_score": best_single.meta["oof_score"]}})

    run = RunContext(problem_name, notes=notes or f"ensemble ({method}) of {len(ranked)} stored experiments",
                     outputs_root=root).activate()
    shutil.copy(source_dir / "data" / "test_data.pkl", config.DATA_DIR / "test_data.pkl")
    shutil.copy(source_dir / "reports" / "resolved_spec.json", config.REPORTS_DIR / "resolved_spec.json")
    joblib.dump(transform_info, config.FEATURES_DIR / "target_transform.pkl")
    np.save(config.FEATURES_DIR / "final_test_predictions.npy", blend_test)
    store.save(f"ensemble_{method}", run.run_id, blend_oof, blend_test, {
        "model": f"Ensemble ({method})", "metric": problem.metric, "oof_score": score,
        "greater_is_better": metric.greater_is_better, "fold_signature": signature,
        "members": info["members"], "ensemble": info, "task": task,
    })
    final = {"final_name": f"Ensemble ({method})", "final_score": score, "final_source": "store_ensemble",
             "metric": problem.metric, "greater_is_better": metric.greater_is_better,
             "fold_signature": signature, "ensemble": info}
    (config.REPORTS_DIR / "final_predictions.json").write_text(json.dumps(final, indent=2, default=str))
    (config.REPORTS_DIR / "model_training_report.json").write_text(json.dumps({
        "status": "success", "best_model": final["final_name"], "best_score": score,
        "scoring_metric": problem.metric, "task": task, "score_source": "out-of-fold",
        "final_predictions": final,
        "all_results": {e.meta["name"]: {"oof_score": e.meta["oof_score"]} for e in ranked},
    }, indent=2, default=str))

    context.reset()
    context.set_active_problem(problem)
    prediction = json.loads(generate_predictions())
    if prediction.get("status") != "success":
        return run.finish("failed", {"error": prediction.get("error")})
    summary = run.finish("completed", prediction)
    summary["improvement_over_best_single"] = metric.signed(score) - metric.signed(best_single.meta["oof_score"])
    return summary
