"""MCP server exposing genML to LLM clients (Claude Code, Claude Desktop, ...).

Runs are long (minutes to hours), so tools never block on training: ``start_run``
queues a job and returns a ``job_id`` immediately, and ``get_run_status`` polls
it. A single worker process runs one job at a time on the GPU.

Register with Claude Code::

    claude mcp add genml -- /path/to/genML/.venv/bin/python -m src.genML.mcp_server

(run from the project root, or pass --directory/cwd so ``src`` is importable).

The stdio transport uses stdout for the protocol, so the server redirects file
descriptor 1 to stderr before importing the pipeline; native libraries that
print (LightGBM, XGBoost) cannot corrupt the protocol stream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _protect_stdout():
    """Keep the real stdout for MCP messages; send everything else to stderr."""
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INSTRUCTIONS = """genML runs tabular ML experiments for declared problems (Kaggle-style train/test CSVs).

Typical loop:
1. list_problems (or create_problem / inspect_dataset for new data).
2. start_run(problem, ...) returns a job_id immediately; runs take minutes to hours.
3. Poll get_run_status(job_id) every few minutes; do not busy-poll.
4. When done: compare_runs, get_run_report(section=models|features), get_feature_diagnostics.
5. Try an idea by passing overrides to start_run (models, n_trials, encodings, ensemble_method);
   start_ensemble blends the best stored experiments across runs.
6. submit_to_kaggle previews by default; it only submits with confirm=true. Ask the user before
   confirming: submissions count against a daily limit and are public to the competition.
CV scores are out-of-fold scores in the problem's own metric (check greater_is_better)."""


def build_server():
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    from src.genML import service

    server = MCPServer(name="genml", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    queues_work = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    @server.tool(annotations=read_only)
    def list_problems() -> list:
        """Declared problems with task, metric, completed runs and the best run's CV score."""
        return service.list_problems()

    @server.tool(annotations=read_only)
    def inspect_dataset(problem: str | None = None, data_path: str | None = None) -> dict:
        """Columns, detected types, target distribution and what inference decides (target, task,
        metric, submission output). Pass a declared problem or a folder with train.csv/test.csv."""
        return service.inspect_dataset(problem=problem, data_path=data_path)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                             open_world_hint=True))
    def create_problem(name: str, data_path: str | None = None, kaggle_competition: str | None = None,
                       target: str | None = None, id_column: str | None = None, task: str | None = None,
                       metric: str | None = None, description: str = "", overwrite: bool = False) -> dict:
        """Write problems/<name>/problem.yaml. For Kaggle, pass kaggle_competition (data is downloaded
        and the metric read from Kaggle); otherwise data_path. Unset fields are inferred and reported."""
        return service.create_problem(name, data_path=data_path, kaggle_competition=kaggle_competition,
                                      target=target, id_column=id_column, task=task, metric=metric,
                                      description=description, overwrite=overwrite)

    @server.tool(annotations=queues_work)
    def start_run(problem: str, models: list[str] | None = None, n_trials: int | None = None,
                  timeout_per_model: int | None = None, seeds: int | None = None,
                  encodings: dict | None = None, ensemble_method: str | None = None, notes: str = "") -> dict:
        """Queue a pipeline run and return its job_id at once. Optional overrides of the spec:
        models (e.g. ["XGBoost","CatBoost","LightGBM"]), n_trials and timeout_per_model (Optuna
        budget per model), seeds (seed averaging), encodings (e.g. {"target_encode":"auto",
        "count_encode":"auto","pairs":"auto"}), ensemble_method (stack|hill_climb|none).
        Describe the idea being tested in notes."""
        return service.start_run(problem, models=models, n_trials=n_trials,
                                 timeout_per_model=timeout_per_model, seeds=seeds, encodings=encodings,
                                 ensemble_method=ensemble_method, notes=notes)

    @server.tool(annotations=queues_work)
    def start_ensemble(problem: str, method: str = "stack", top: int = 10, notes: str = "") -> dict:
        """Queue a blend of the best stored out-of-fold experiments across all runs of a problem
        (method stack or hill_climb). Produces a new run with its own submission."""
        return service.start_ensemble(problem, method=method, top=top, notes=notes)

    @server.tool(annotations=read_only)
    def get_run_status(job_id: int) -> dict:
        """Status of a queued/running/finished job: stage, current model and trial, finished models'
        scores, and the CV score once complete. Poll every few minutes."""
        return service.get_run_status(job_id)

    @server.tool(annotations=read_only)
    def list_jobs(limit: int = 10) -> list:
        """Recent jobs (newest first)."""
        return service.list_jobs(limit=limit)

    @server.tool(annotations=read_only)
    def list_runs(problem: str, limit: int = 10) -> list:
        """Recent runs of a problem with status, CV score, final model and notes."""
        return service.list_runs(problem, limit=limit)

    @server.tool(annotations=read_only)
    def compare_runs(problem: str, run_ids: list[str] | None = None, limit: int = 10) -> dict:
        """Completed runs side by side: CV score, gap to the best run, each model's OOF score."""
        return service.compare_runs(problem, run_ids=run_ids, limit=limit)

    @server.tool(annotations=read_only)
    def get_run_report(problem: str, run_id: str, section: str = "summary") -> dict:
        """One section of a run: summary | models (scores, params, ensemble) | features (names,
        encodings) | spec (resolved problem spec) | errors (error and log tail)."""
        return service.get_run_report(problem, run_id, section)

    @server.tool(annotations=read_only)
    def get_feature_diagnostics(problem: str, run_id: str) -> dict:
        """Per-column signal on the raw training data (AUC or correlation with the target, category
        target-rate spread), to decide which features, interactions or encodings to try next."""
        return service.get_feature_diagnostics(problem, run_id)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                             idempotent_hint=True, open_world_hint=False))
    def cancel_run(job_id: int) -> dict:
        """Cancel a queued job or stop a running one."""
        return service.cancel_run(job_id)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                             open_world_hint=True))
    def submit_to_kaggle(problem: str, run_id: str | None = None, message: str = "",
                         confirm: bool = False) -> dict:
        """Preview (default) or submit a run's submission.csv to Kaggle. Only submits with
        confirm=true; get the user's go-ahead first (daily limit, public leaderboard)."""
        return service.submit_to_kaggle(problem, run_id=run_id, message=message, confirm=confirm)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_submissions(problem: str) -> list:
        """Submitted runs with CV score next to Kaggle public/private scores (refreshed from Kaggle)."""
        return service.get_submissions(problem)

    @server.resource("genml://runs/{problem}/{run_id}/log", mime_type="text/plain")
    def run_log(problem: str, run_id: str) -> str:
        """Full pipeline.log of a run."""
        path = service._run_dir(problem, run_id) / "pipeline.log"
        return path.read_text(errors="replace") if path.exists() else ""

    @server.resource("genml://runs/{problem}/{run_id}/reports/{name}", mime_type="application/json")
    def run_report_file(problem: str, run_id: str, name: str) -> str:
        """A JSON report of a run (e.g. model_training_report.json, resolved_spec.json)."""
        reports = service._run_dir(problem, run_id) / "reports"
        path = (reports / service._check_name(name, "report name"))
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        if not path.exists():
            raise service.ServiceError(f"No report {name}; see get_run_report for sections")
        return path.read_text()

    return server


def main() -> None:
    _protect_stdout()
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    build_server().run("stdio")


if __name__ == "__main__":
    main()
