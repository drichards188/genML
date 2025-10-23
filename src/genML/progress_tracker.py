"""
Real-time progress tracking for ML Pipeline.

This module provides a centralized progress tracking system that pipeline stages
can use to report their progress, which is then consumed by the dashboard UI.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Thread-safe progress tracker for ML pipeline execution.

    Tracks pipeline stages, model training progress, resource usage, and AI insights.
    Writes progress to JSON file for consumption by the dashboard.
    """

    def __init__(self, run_id: Optional[str] = None, dataset_name: str = "unknown"):
        """
        Initialize progress tracker.

        Args:
            run_id: Unique identifier for this pipeline run
            dataset_name: Name of the dataset being processed
        """
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_name = dataset_name
        self.started_at = datetime.now().isoformat()

        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

        # Progress data structure
        self._data = {
            "run_id": self.run_id,
            "dataset_name": dataset_name,
            "status": "running",
            "started_at": self.started_at,
            "completed_at": None,
            "current_stage": 0,
            "current_stage_name": "",
            "stage_progress_pct": 0,
            "current_task": "Initializing...",
            "eta_seconds": None,
            "stages": {},
            "models": [],
            "resources": {
                "gpu_memory_mb": 0,
                "gpu_memory_total_mb": 0,
                "cpu_percent": 0,
                "ram_mb": 0,
                "elapsed_seconds": 0
            },
            "ai_insights": {
                "model_selection": {},
                "feature_suggestions": [],
                "error_patterns": []
            }
        }

        # Set up progress directory
        self.progress_dir = Path("outputs/progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.progress_dir / "current_run.json"

        # Archive directory for historical runs
        self.archive_dir = self.progress_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)

        # Write initial state
        self._write_progress()

        logger.info(f"ProgressTracker initialized for run {self.run_id}")

    def _write_progress(self):
        """Write current progress to JSON file (thread-safe)."""
        try:
            # Calculate elapsed time
            start_dt = datetime.fromisoformat(self.started_at)
            elapsed = (datetime.now() - start_dt).total_seconds()
            self._data["resources"]["elapsed_seconds"] = int(elapsed)

            # Write to file
            with open(self.progress_file, 'w') as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write progress file: {e}")

    def track_stage_start(self, stage_num: int, stage_name: str, description: str = ""):
        """
        Track the start of a pipeline stage.

        Args:
            stage_num: Stage number (1-5)
            stage_name: Name of the stage
            description: Optional description of what this stage does
        """
        with self._lock:
            self._data["current_stage"] = stage_num
            self._data["current_stage_name"] = stage_name
            self._data["current_task"] = description or f"Starting {stage_name}..."
            self._data["stage_progress_pct"] = 0

            # Initialize stage data
            self._data["stages"][str(stage_num)] = {
                "name": stage_name,
                "description": description,
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "duration_seconds": None,
                "summary": {}
            }

            self._write_progress()
            logger.info(f"Stage {stage_num} started: {stage_name}")

    def track_stage_complete(self, stage_num: int, summary: Dict[str, Any] = None):
        """
        Mark a stage as complete.

        Args:
            stage_num: Stage number (1-5)
            summary: Optional summary data for this stage
        """
        with self._lock:
            stage_key = str(stage_num)
            if stage_key in self._data["stages"]:
                stage = self._data["stages"][stage_key]
                stage["status"] = "completed"
                stage["completed_at"] = datetime.now().isoformat()

                # Calculate duration
                start_dt = datetime.fromisoformat(stage["started_at"])
                end_dt = datetime.fromisoformat(stage["completed_at"])
                stage["duration_seconds"] = int((end_dt - start_dt).total_seconds())

                # Add summary
                if summary:
                    stage["summary"] = summary

                self._data["stage_progress_pct"] = 100

                self._write_progress()
                logger.info(f"Stage {stage_num} completed in {stage['duration_seconds']}s")

    def track_stage_progress(self, stage_num: int, progress_pct: int, task_description: str = ""):
        """
        Update progress within a stage.

        Args:
            stage_num: Stage number
            progress_pct: Progress percentage (0-100)
            task_description: Description of current task
        """
        with self._lock:
            self._data["stage_progress_pct"] = min(100, max(0, progress_pct))
            if task_description:
                self._data["current_task"] = task_description
            self._write_progress()

    def track_model_start(self, model_name: str, total_trials: int = None):
        """
        Track the start of model training.

        Args:
            model_name: Name of the model being trained
            total_trials: Total number of hyperparameter tuning trials (if applicable)
        """
        with self._lock:
            # Check if model already exists
            existing_model = next((m for m in self._data["models"] if m["name"] == model_name), None)

            if existing_model:
                existing_model["status"] = "training"
                existing_model["current_trial"] = 0
            else:
                model_data = {
                    "name": model_name,
                    "status": "training",
                    "started_at": datetime.now().isoformat(),
                    "mean_score": None,
                    "std_score": None,
                    "best_params": {},
                    "tuned": total_trials is not None and total_trials > 0,
                    "current_trial": 0,
                    "total_trials": total_trials,
                    "best_score": None,
                    "trial_history": []
                }
                self._data["models"].append(model_data)

            self._data["current_task"] = f"Training {model_name}..."
            self._write_progress()

    def track_model_trial(self, model_name: str, trial_num: int, score: float):
        """
        Track a hyperparameter tuning trial.

        Args:
            model_name: Name of the model
            trial_num: Trial number
            score: Score for this trial
        """
        with self._lock:
            model = next((m for m in self._data["models"] if m["name"] == model_name), None)
            if model:
                model["current_trial"] = trial_num
                model["trial_history"].append({
                    "trial": trial_num,
                    "score": score,
                    "timestamp": datetime.now().isoformat()
                })

                # Update best score
                if model["best_score"] is None or score > model["best_score"]:
                    model["best_score"] = score

                # Update current task
                if model["total_trials"]:
                    self._data["current_task"] = f"Tuning {model_name} - Trial {trial_num}/{model['total_trials']}"
                else:
                    self._data["current_task"] = f"Training {model_name} - Trial {trial_num}"

                self._write_progress()

    def track_model_complete(self, model_name: str, mean_score: float, std_score: float = 0.0,
                           best_params: Dict[str, Any] = None):
        """
        Mark model training as complete.

        Args:
            model_name: Name of the model
            mean_score: Mean cross-validation score
            std_score: Standard deviation of CV score
            best_params: Best hyperparameters found
        """
        with self._lock:
            model = next((m for m in self._data["models"] if m["name"] == model_name), None)
            if model:
                model["status"] = "completed"
                model["mean_score"] = mean_score
                model["std_score"] = std_score
                if best_params:
                    model["best_params"] = best_params
                model["completed_at"] = datetime.now().isoformat()

                self._write_progress()
                logger.info(f"Model {model_name} completed with score {mean_score:.4f}")

    def update_model_training_stage(self, models_trained: int, models_total: int):
        """
        Update overall model training stage progress.

        Args:
            models_trained: Number of models trained so far
            models_total: Total number of models to train
        """
        with self._lock:
            stage_key = "4"  # Model training is stage 4
            if stage_key in self._data["stages"]:
                self._data["stages"][stage_key]["summary"]["models_trained"] = models_trained
                self._data["stages"][stage_key]["summary"]["models_total"] = models_total

                # Update stage progress
                progress = int((models_trained / models_total) * 100) if models_total > 0 else 0
                self._data["stage_progress_pct"] = progress

                self._write_progress()

    def track_resources(self, gpu_memory_mb: float = 0, gpu_memory_total_mb: float = 0,
                       cpu_percent: float = 0, ram_mb: float = 0):
        """
        Update resource usage metrics.

        Args:
            gpu_memory_mb: Current GPU memory usage in MB
            gpu_memory_total_mb: Total GPU memory in MB
            cpu_percent: CPU utilization percentage
            ram_mb: RAM usage in MB
        """
        with self._lock:
            self._data["resources"]["gpu_memory_mb"] = int(gpu_memory_mb)
            self._data["resources"]["gpu_memory_total_mb"] = int(gpu_memory_total_mb)
            self._data["resources"]["cpu_percent"] = int(cpu_percent)
            self._data["resources"]["ram_mb"] = int(ram_mb)
            self._write_progress()

    def track_ai_insight(self, insight_type: str, data: Dict[str, Any]):
        """
        Track AI advisor insights.

        Args:
            insight_type: Type of insight (model_selection, feature_suggestions, error_patterns)
            data: Insight data
        """
        with self._lock:
            if insight_type in self._data["ai_insights"]:
                self._data["ai_insights"][insight_type] = data
                self._write_progress()
                logger.info(f"AI insight recorded: {insight_type}")

    def set_pipeline_status(self, status: str):
        """
        Set overall pipeline status.

        Args:
            status: Status string (running, completed, failed)
        """
        with self._lock:
            self._data["status"] = status
            if status in ["completed", "failed"]:
                self._data["completed_at"] = datetime.now().isoformat()
            self._write_progress()

    def archive_run(self):
        """Archive the current run to the archive directory."""
        try:
            archive_file = self.archive_dir / f"{self.run_id}.json"
            with open(self.progress_file, 'r') as src:
                data = json.load(src)
            with open(archive_file, 'w') as dst:
                json.dump(data, dst, indent=2)
            logger.info(f"Run archived to {archive_file}")
        except Exception as e:
            logger.error(f"Failed to archive run: {e}")

    def get_progress_data(self) -> Dict[str, Any]:
        """Get current progress data (thread-safe)."""
        with self._lock:
            return self._data.copy()
