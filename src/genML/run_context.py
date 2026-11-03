"""Per-run output folders and the run index.

Every run writes to ``outputs/<problem>/<run_id>/``::

    outputs/fraud-s6e11/20261103_142501/
    ├── data/  features/  models/  predictions/  reports/
    ├── submission.csv
    ├── pipeline.log
    └── run.json            # summary: spec, metric, CV score, best model, timings

and appends one row to ``outputs/<problem>/runs.csv`` so runs can be compared.

Activating a run points the pipeline's path settings (``config.DATA_DIR`` and
friends) at the run folder. Runs inside one process are sequential; the job
worker runs each run in its own process, so concurrent runs never share state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.genML.pipeline import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
RUN_SUBDIRS = ("data", "features", "models", "predictions", "reports")
INDEX_FIELDS = (
    "run_id", "started_at", "finished_at", "status", "metric", "cv_score", "best_model",
    "n_features", "train_rows", "submission", "spec_hash", "git_commit", "notes",
)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None


def spec_hash(spec_dict: Dict[str, Any]) -> str:
    payload = json.dumps(spec_dict, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:10]


@dataclass
class RunContext:
    """Identity and folders of one pipeline run."""

    problem_name: str
    run_id: str = field(default_factory=new_run_id)
    outputs_root: Optional[Path] = None  # default: <project>/outputs
    notes: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def root(self) -> Path:
        return Path(self.outputs_root) if self.outputs_root else OUTPUTS_ROOT

    @property
    def problem_dir(self) -> Path:
        return self.root / self.problem_name

    @property
    def run_dir(self) -> Path:
        return self.problem_dir / self.run_id

    @property
    def log_file(self) -> Path:
        return self.run_dir / "pipeline.log"

    @property
    def submission_file(self) -> Path:
        return self.run_dir / "submission.csv"

    @property
    def summary_file(self) -> Path:
        return self.run_dir / "run.json"

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def activate(self) -> "RunContext":
        """Create the run folders and point the pipeline's paths at them."""
        for sub in RUN_SUBDIRS:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        config.OUTPUTS_DIR = self.run_dir
        config.DATA_DIR = self.run_dir / "data"
        config.FEATURES_DIR = self.run_dir / "features"
        config.MODELS_DIR = self.run_dir / "models"
        config.PREDICTIONS_DIR = self.run_dir / "predictions"
        config.REPORTS_DIR = self.run_dir / "reports"
        config.AI_TUNING_OVERRIDE_PATH = config.REPORTS_DIR / "ai_tuning_overrides.json"
        config.ACTIVE_RUN = self
        logger.info("Run %s for problem %s -> %s", self.run_id, self.problem_name, self.run_dir)
        return self

    def finish(self, status: str, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write run.json and append the run to outputs/<problem>/runs.csv."""
        results = results or {}
        resolved_path = self.run_dir / "reports" / "resolved_spec.json"
        resolved = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        training = _read_json(self.run_dir / "reports" / "model_training_report.json")
        features = _read_json(self.run_dir / "reports" / "feature_engineering_report.json")
        data = _read_json(self.run_dir / "reports" / "data_exploration_report.json")

        summary = {
            "run_id": self.run_id,
            "problem": self.problem_name,
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "metric": resolved.get("metric"),
            "cv_score": training.get("best_score"),
            "best_model": training.get("best_model"),
            "all_scores": {
                name: result.get("mean_score") for name, result in (training.get("all_results") or {}).items()
            },
            "n_features": len(features.get("features_used") or []) or None,
            "train_rows": (data.get("train_shape") or [None])[0],
            "submission": str(self.submission_file) if self.submission_file.exists() else None,
            "spec_hash": spec_hash({k: v for k, v in resolved.items() if k not in ("data_dir", "inferred")}),
            "git_commit": _git_commit(),
            "notes": self.notes,
            "error": results.get("error"),
        }
        self.summary_file.parent.mkdir(parents=True, exist_ok=True)
        self.summary_file.write_text(json.dumps(summary, indent=2, default=str))
        append_to_index(self.problem_dir / "runs.csv", summary)
        return summary


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def append_to_index(index_path: Path, summary: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not index_path.exists()
    with open(index_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({key: summary.get(key) for key in INDEX_FIELDS})


def list_runs(problem_name: str, outputs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Runs of a problem, newest first, read from each run's run.json."""
    runs = []
    for summary_file in (Path(outputs_root or OUTPUTS_ROOT) / problem_name).glob("*/run.json"):
        summary = _read_json(summary_file)
        if summary:
            runs.append(summary)
    return sorted(runs, key=lambda r: r.get("run_id", ""), reverse=True)


def load_run(problem_name: str, run_id: str, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    summary = _read_json(Path(outputs_root or OUTPUTS_ROOT) / problem_name / run_id / "run.json")
    if not summary:
        raise FileNotFoundError(f"No run {run_id} for problem {problem_name}")
    return summary
