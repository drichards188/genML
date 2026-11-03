# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

genML is a tabular ML pipeline for Kaggle-style problems (`train.csv` / `test.csv`). Each problem is declared in `problems/<name>/problem.yaml`; unset fields are inferred and every inference is recorded. Runs tune GPU gradient-boosting models on the problem's own metric, store out-of-fold (OOF) predictions on folds shared by all runs, stack them, and write a submission whose content is decided by the metric. Runs can be queued through a job worker and driven by LLM clients through an MCP server. A FastAPI + React dashboard shows live progress.

## Key Commands

### Environment
```bash
# CPU environment (enough for all tests); .venv is gitignored
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-dashboard.txt
VIRTUAL_ENV=.venv uv pip install kagglehub kaggle "mcp>=2"   # Kaggle download/submit, MCP server
source .venv/bin/activate

# GPU environment (RAPIDS)
mamba env create -f environment.yml && conda activate genml
```

### Running
```bash
python src/genML/main.py --list-problems
python src/genML/main.py --problem fraud-s6e11 [--download] [--notes "what changed"]
python src/genML/main.py --list-runs fraud-s6e11
python src/genML/main.py --show-run fraud-s6e11 RUN_ID
python src/genML/main.py --ensemble fraud-s6e11 [--method stack|hill_climb] [--top 10]
python src/genML/main.py --submit fraud-s6e11 [--run RUN_ID] [--message TEXT]   # outward-facing: ask first
python src/genML/main.py --submissions fraud-s6e11
python src/genML/main.py --init-kaggle playground-series-s6e11 --name fraud-s6e11
python src/genML/main.py            # legacy: datasets/current/, everything inferred

python -m src.genML.jobs worker | enqueue NAME | list | cancel JOB_ID
python -m src.genML.mcp_server      # MCP server (stdio)
python run_api.py                   # dashboard API on :8000; UI: cd dashboard && npm run dev
```

Kaggle auth: `KAGGLE_API_TOKEN` env var or `~/.kaggle/access_token`. Never write tokens into the repo.

### Testing
```bash
pytest                    # full suite, including slow end-to-end runs on local Kaggle data
pytest -m "not slow"      # skip end-to-end runs
pytest tests/problems     # golden tests for every declared problem
```

Every test runs in its own temp directory (`tests/conftest.py::isolate_outputs`), so the suite never writes to the real `outputs/`. Golden and smoke tests skip when a problem's data is not available locally; Kaggle data must never be committed.

## Architecture

### Problem specs (`src/genML/pipeline/problem.py`)
- `ProblemSpec`: fields `name, description, source, target, id_column, task, metric, positive_class, cv, submission, columns, models, budget, feature_engineering, ensemble`. `None` means infer.
- `resolve_problem()` fills every field and records `inferred[field] = reason`; the result (`ResolvedProblem`) is saved as `reports/resolved_spec.json` in the run folder.
- Inference rules (`pipeline/inference.py`): task follows AutoGluon (2 values binary; non-numeric multiclass; non-integer floats regression; integers multiclass only with few distinct values). Target is the train-only column, chosen deterministically.
- Metric is never inferred from sample-submission placeholders when declared; without a declaration `metrics.default_metric_for()` uses the sample only as a hint.
- `locate_data_dir()`: `source.path`, then `datasets/<name>/`, then the kagglehub cache, then legacy discovery.
- `GENML_PROBLEMS` and `GENML_OUTPUTS` env vars override the problems and outputs folders (used by tests and worker subprocesses).

### Metrics (`src/genML/pipeline/metrics.py`)
One registry entry per metric: sklearn scorer per task (higher is better), output kind (`label`, `probability`, `class_probabilities`, `top_k`, `value`), `greater_is_better`, Kaggle names, optional target transform (`rmsle` trains on log1p). `Metric.score()` scores OOF predictions on the metric's own scale. Add new metrics here; everything else (tuning, OOF scoring, submissions) follows.

### Run flow (`src/genML/flow.py`)
`run_pipeline()` creates a `RunContext`, then `MLPipelineFlow.kickoff()` runs five stages in order (`STAGES`): load data, feature engineering, model selection (records a skip; models come from the spec), training, prediction. A failed stage raises `PipelineAbort`; the run is still recorded as failed. CrewAI was removed; the flow is plain Python.

### Run folders (`src/genML/run_context.py`)
`RunContext.activate()` creates `outputs/<problem>/<run_id>/{data,features,models,predictions,reports}` and points `config.DATA_DIR` and friends at it. `finish()` writes `run.json` and appends to `outputs/<problem>/runs.csv`. Path globals are per process; concurrent runs are separate worker processes.

### Feature engineering
- `features/feature_engine.py` (`AutoFeatureEngine`), `data_analyzer.py` (type detection; AutoGluon text gate: string columns need >= 3 words on average and unique ratio > 0.01 to count as text), `feature_processors.py`, `feature_selector.py` (selection runs on at most `selection_max_rows`, default 100k), `domain_researcher.py`.
- `features/encoding.py`: opt-in via `feature_engineering.encodings`; out-of-fold target encoding (shared folds), count encoding, pair crosses, numeric-as-categorical, original-dataset means.
- `pipeline/targets.py`: label encoding (with decode), log1p for RMSLE, logit for bounded regression targets, and inverses.

### Training (`src/genML/pipeline/training.py`, `tuning.py`)
Optuna tunes each model on `problem.scoring`, within `budget` (`n_trials`, `timeout_per_model`, `tuning: false`). Then `_run_oof_stage()`: fixed folds (`pipeline/oof.py`, cached per problem as `folds_<signature>.npy`), `fit_oof()` per model (and per seed with `budget.seeds`), experiments saved to `outputs/<problem>/oof/`, scored with the native metric. The run's ensemble (`pipeline/ensemble.py`: `stack` with nested CV, or `hill_climb`) competes with the best single model on OOF score; the winner's test predictions go to `features/final_test_predictions.npy`. `budget.refit_full` refits the best single model on all rows instead.

XGBoost GPU uses `tree_method="hist", device="cuda"` (`gpu_hist` no longer exists in XGBoost 3).

### Submissions (`src/genML/pipeline/submission.py`, `src/genML/submissions.py`)
`build_submission()` writes what the metric needs; `validate_submission()` checks columns, row count and IDs against `sample_submission.csv`. `submissions.submit_run()` tags the Kaggle description with `[genml <run_id>]` and logs CV in `outputs/<problem>/submissions.csv`; `refresh_scores()` pulls public/private scores back.

### Cross-run ensembles (`src/genML/pipeline/store_ensemble.py`)
Blends the best stored experiments sharing the latest run's folds into a new run folder with its own submission.

### Jobs and MCP
- `src/genML/jobs.py`: SQLite queue (`outputs/jobs.db`), one worker, each job in its own spawned process (GPU concurrency 1, logs in `outputs/jobs/`).
- `src/genML/service.py`: compact JSON-returning functions used by the MCP server.
- `src/genML/mcp_server.py`: MCP tools over the service (`start_run` returns a job id; `get_run_status` polls; `submit_to_kaggle` previews unless `confirm=true`). It redirects fd 1 to stderr before importing the pipeline so native library output cannot corrupt the stdio protocol.

### Dashboard API (`src/genML/api/server.py`)
`/api/status`, `/ws/progress` (live progress from `outputs/progress/current_run.json`), `/api/problems`, `/api/problems/{problem}/runs[/{run_id}]`, and run-aware `/api/reports`, `/api/reports/{name}`, `/api/models`, `/api/logs/{run_id}` (query `problem` and `run_id`; default the live or newest run). Identifiers are validated against path traversal.

### Legacy
`src/genML/tools.py` re-exports the pipeline API for old scripts. `config.AI_ADVISORS_CONFIG` is a stub: the OpenAI advisors were never committed and were removed; LLM guidance comes from MCP clients.

## Conventions

- New problems: `--init-kaggle` or `service.create_problem`, then review the written YAML; add golden coverage automatically (tests parametrize over `problems/`).
- New metrics: add to the registry in `metrics.py` (scorer, output kind, direction, Kaggle names).
- New models: add a factory in `training.py`, an Optuna objective in `tuning.py` (accept `scoring=`), and a `SUPPORTED_MODEL_SUMMARIES` entry in `config.py`.
- Keep `requirements.txt`, `pyproject.toml` and `environment.yml` in sync.
- Source files use CRLF line endings in the working tree (git autocrlf); match them when scripting edits.
- 4-space indentation, snake_case, Python 3.10+ (Kaggle submit and MCP need 3.11+). Format with black/isort; lint with flake8.

## Dependencies

Core: pandas (<3), numpy, scikit-learn, xgboost, lightgbm, catboost, joblib, pyyaml, optuna. Optional extras in `pyproject.toml`: `dashboard` (fastapi, uvicorn, websockets, watchdog, psutil), `kaggle` (kagglehub, kaggle), `mcp`. GPU: `environment.yml` (RAPIDS cuML, CUDA 11.8). Dev: `requirements-dev.txt` (pytest, httpx, black, flake8, isort, mypy).
