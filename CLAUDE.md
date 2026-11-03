# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **CrewAI ML Pipeline** project that automates machine learning workflows using CrewAI Flows. It takes a Kaggle-style `train.csv`/`test.csv` dataset, engineers features, tunes and trains several models (with optional GPU acceleration), and writes a formatted submission. A FastAPI + React dashboard shows pipeline progress in real time.

## Key Commands

### Running the Pipeline
```bash
# Main execution (recommended for GPU support)
conda activate genml
python src/genML/main.py

# Alternative using CrewAI CLI (CPU only)
crewai run

# Using UV package scripts (CPU only)
uv run kickoff
uv run run_pipeline
```

### Running the Dashboard
```bash
# Terminal 1: Start API backend (http://localhost:8000)
python run_api.py            # options: --host, --port, --reload

# Terminal 2: Start dashboard UI (development mode, http://localhost:5173)
cd dashboard
npm install  # First time only
npm run dev

# Terminal 3: Run the pipeline to see live updates
python src/genML/main.py

# Production build
cd dashboard && npm run build && cd ..
python run_api.py  # Serves built dashboard at http://localhost:8000
```

### Development Commands
```bash
# Install dependencies (CPU)
pip install -r requirements.txt
pip install -r requirements-dashboard.txt   # dashboard backend
pip install -r requirements-dev.txt         # pytest, black, isort, flake8, mypy

# Install with GPU support (recommended)
mamba env create -f environment.yml
conda activate genml

# OR using UV (CPU only); add dashboard extras with --extra dashboard
uv sync

# View flow diagram
python -c "from src.genML.main import plot; plot()"
# OR
uv run plot
```

### Testing
```bash
# One-time CPU test environment (no conda/GPU needed); .venv is gitignored
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-dashboard.txt
source .venv/bin/activate

pytest                                           # full suite
pytest --cov=src/genML --cov-report=term-missing # with coverage
pytest tests/test_pipeline_modules.py            # one module
pytest tests/integration                         # integration tests
pytest -k feature_engine                         # by keyword
```

## Architecture

### Core Flow Structure
`MLPipelineFlow` in `src/genML/flow.py` runs a 5-stage sequential pipeline using CrewAI's `@start()` / `@listen()` decorators:

1. **Data Loading** (`load_data_task`): loads and validates the dataset
2. **Feature Engineering** (`feature_engineering_task`): automated feature generation
3. **Model Selection** (`model_selection_task`): AI model advisor guidance (skipped when AI advisors are disabled)
4. **Model Training** (`model_training_task`): multi-model training with Optuna tuning and ensembles
5. **Prediction** (`prediction_task`): generates final submissions

Each stage returns JSON for the next stage, reports progress through `progress_tracker.py`, and aborts downstream stages on failure.

### Pipeline Package (`src/genML/pipeline/`)
All production logic lives here.

- `__init__.py`: public API exports (`load_dataset`, `engineer_features`, `run_model_selection_advisor`, `train_model_pipeline`, `generate_predictions`)
- `config.py`: shared constants, paths, tuning/GPU/memory/parallel settings, hyperparameter ranges
- `dataset.py`: dataset discovery and loading with multi-path resolution
- `feature_engineering.py`: automated feature engineering (calls the AI feature ideation advisor when enabled)
- `model_advisor.py`: model selection guidance and problem type detection
- `training.py`: Optuna tuning, ensemble evaluation, GPU-aware and parallel training
- `prediction.py`: submission and prediction generation
- `ai_tuning.py`: AI-driven hyperparameter override utilities
- `tuning.py`: Optuna optimization strategies
- `utils.py`: memory management and GPU cleanup helpers
- `optional_dependencies.py`: runtime checks for optional libraries (CatBoost, LightGBM, TabNet, cuML)

**Backward compatibility**: `src/genML/tools.py` re-exports pipeline functions for older scripts and tests. New code should import from `src.genML.pipeline` directly.

### Automated Feature Engineering (`src/genML/features/`)
- `feature_engine.py`: main `AutoFeatureEngine` orchestrator
- `data_analyzer.py`: automatic data type detection (`DataTypeAnalyzer`)
- `feature_processors.py`: processors for numerical, categorical, text, and datetime data
- `feature_selector.py`: `AdvancedFeatureSelector` with statistical and model-based selection
- `domain_researcher.py`: `DomainResearcher` for domain-specific feature strategies

### Dashboard Backend (`src/genML/api/`)
- `server.py`: FastAPI app factory (`create_app`) with REST endpoints, WebSocket, and static serving of `dashboard/dist`
- `websocket_manager.py`: WebSocket connection manager; watches progress files and broadcasts updates
- `run_api.py` (project root): uvicorn runner for `create_app`

### Other Modules (`src/genML/`)
- `main.py`: entry point (data file checks, dataset name detection, kickoff, plot)
- `submission_formatter.py`: adaptive submission formatting for competition platforms
- `progress_tracker.py`: pipeline progress tracking for the dashboard
- `resource_monitor.py`: CPU/RAM/GPU usage sampling for the dashboard
- `logging_config.py`: centralized logging configuration
- `gpu_utils.py`: GPU detection and memory management utilities

### Not Yet Implemented in This Repository

The code has hooks for two features whose modules are **not committed**:

- **AI advisors** (`src/genML/ai_advisors/`): imported lazily by `feature_engineering.py`, `model_advisor.py`, and `training.py` (`OpenAIClient`, `FeatureIdeationAdvisor`, `ModelSelectionAdvisor`, `ErrorPatternAnalyzer`). Because the package is missing, `AI_ADVISORS_CONFIG["enabled"]` defaults to `False`. Keep it off until the package exists; each call site catches the import error, but the model selection stage would otherwise report a failure.
- **Ingestion pipeline** (`src/genML/pipeline/ingestion.py` exposing `run_ingestion_pipeline`): `dataset.py` and `main.py` switch to it when `INGESTION_CONFIG` is not `None`. With the module missing, setting `INGESTION_CONFIG` makes data loading fail, so leave it as `None`.

If you implement either, update this section and the configuration notes below.

### Directory Structure

```
src/genML/
├── main.py
├── flow.py
├── tools.py                 # Legacy API (re-exports pipeline)
├── submission_formatter.py
├── progress_tracker.py
├── resource_monitor.py
├── logging_config.py
├── gpu_utils.py
├── pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── feature_engineering.py
│   ├── model_advisor.py
│   ├── training.py
│   ├── prediction.py
│   ├── ai_tuning.py
│   ├── tuning.py
│   ├── optional_dependencies.py
│   └── utils.py
├── api/
│   ├── __init__.py
│   ├── server.py
│   └── websocket_manager.py
└── features/
    ├── __init__.py
    ├── feature_engine.py
    ├── data_analyzer.py
    ├── feature_processors.py
    ├── feature_selector.py
    └── domain_researcher.py

dashboard/src/
├── api/client.ts            # API client for backend
├── hooks/useWebSocket.ts    # WebSocket connection with auto-reconnect
├── hooks/useProgressData.ts
├── store/progressStore.ts   # Zustand state
├── types/pipeline.ts
├── App.tsx / App.css / index.css
└── main.tsx

tests/
├── conftest.py              # Shared fixtures
├── test_pipeline_modules.py
├── test_feature_engine.py
├── test_feature_processors.py
├── test_flow.py
├── test_reproducibility.py
├── test_submission_formatter.py
├── features/                # data_analyzer, domain_researcher, feature_selector, interactions
├── integration/             # end-to-end, full pipeline, training, domain scenarios, performance
└── validation/              # edge cases, feature quality
```

**Datasets** (gitignored): `datasets/current/` is the primary location; `datasets/` and the project root are fallbacks.

**Outputs** (gitignored):
```
outputs/
├── data/         # train_data.pkl, test_data.pkl
├── features/     # .npy arrays, feature_names.txt
├── models/       # trained models, cross_validation_results.csv
├── predictions/
├── reports/      # JSON reports from each stage
└── logs/
```

## Problem Type Detection

The system detects regression vs. classification from the target variable:
- **Classification**: ≤20 unique values, integer-like values, low unique ratio
- **Regression**: many unique values, continuous values, high unique ratio

This drives model selection and evaluation metrics.

## Data Loading

With `INGESTION_CONFIG = None` (the default and only working mode today), the pipeline looks for `train.csv` and `test.csv` in this order:
1. `datasets/current/` (recommended)
2. `datasets/`
3. Project root

Optionally include `sample_submission.csv` for submission format detection. Loading produces `outputs/data/train_data.pkl`, `outputs/data/test_data.pkl`, and `outputs/reports/data_exploration_report.json`.

## GPU Support

- `gpu_utils.py` detects CUDA and RAPIDS (cuML/cuDF) and falls back to CPU gracefully.
- `training.py` has GPU model factories for XGBoost, CatBoost, LightGBM, and TabNet; GPU models train sequentially to avoid OOM while CPU models can train in parallel (`PARALLEL_TRAINING_CONFIG`).
- `GPU_CONFIG` and `MEMORY_CONFIG` in `config.py` control device use and memory limits (tuned to avoid WSL2 crashes).
- Install via `mamba env create -f environment.yml`. The env pins `cuda-version<12.0`.

**Note**: Avoid `crewai run` with the GPU conda environment; it spawns a uv environment without RAPIDS. Use `python src/genML/main.py`.

## Feature Engineering

`AutoFeatureEngine` provides automatic data type detection (numerical, categorical, text, datetime), domain-specific strategies, configurable processing via the config dict in `engineer_features()`, and statistical plus model-based feature selection.

## Model Training

Models trained depend on problem type and installed libraries:
- Logistic Regression (classification) / Linear Regression (regression)
- Random Forest
- XGBoost (GPU if available)
- CatBoost (GPU if available)
- LightGBM (GPU if available)
- TabNet (optional, requires `pytorch-tabnet`)
- Stacking / voting ensembles

Training uses Optuna tuning (`TUNING_CONFIG`: trials, timeout, `models_to_tune`), StratifiedKFold or KFold cross-validation, optional AI tuning overrides from `outputs/reports/ai_tuning_overrides.json`, and automatic best-model selection.

## Output Files

- `submission.csv` and a timestamped `submission_YYYYMMDD_HHMMSS.csv` backup
- `outputs/models/best_model_*.pkl`
- `outputs/models/cross_validation_results.csv`
- `outputs/features/feature_names.txt`
- `outputs/reports/*.json`

## Error Handling

- Failed stages return `{"status": "failed", "error": "..."}`.
- Downstream stages detect failures and abort (`PipelineAbort`).
- Optional features (AI advisors, optional model libraries) degrade gracefully.

## Extending the System

**New feature processors**: add classes in `src/genML/features/feature_processors.py`.

**New models**: update the model factories in `src/genML/pipeline/training.py`, add a GPU factory if applicable, add hyperparameter ranges and a `SUPPORTED_MODEL_SUMMARIES` entry in `config.py`.

**New domains**: extend `DomainResearcher` in `src/genML/features/domain_researcher.py`.

**Custom submission formats**: extend `SubmissionFormatter` in `src/genML/submission_formatter.py`.

**New pipeline stages**:
1. Create the function in the appropriate `pipeline/` module
2. Export it from `pipeline/__init__.py` (and `tools.py` if legacy access is needed)
3. Add a `@listen()` method to `MLPipelineFlow` in `flow.py` with progress tracking
4. Update the dashboard stage list if the stage should be displayed

## Code Conventions

- All production logic lives in `src/genML/pipeline/`; import from `src.genML.pipeline`.
- 4-space indentation, snake_case, descriptive names; target Python 3.10+.
- Format with `black` and `isort`; lint with `flake8`; type-check with `mypy` (all in `requirements-dev.txt`).
- Keep the three dependency files in sync: `requirements.txt`, `pyproject.toml`, and `environment.yml`.

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: control GPU device selection
- `OPENAI_API_KEY`: only relevant once the AI advisors package exists

Keep secrets in `.env` (gitignored, along with `env`, `.env.*`, `*.env.local`). Never hardcode or commit API keys.

## Dependencies

**Core** (`requirements.txt` / `pyproject.toml`): Python 3.10+, `crewai[tools]>=0.193.2,<1.0.0`, pandas (<3.0), numpy, scikit-learn, xgboost, lightgbm, catboost, joblib, pyyaml, optuna.

**GPU** (`environment.yml`): RAPIDS cuML, CUDA toolkit, GPU-enabled XGBoost/CatBoost/LightGBM.

**Optional**: `pytorch-tabnet` (TabNet model).

**Dashboard backend** (`requirements-dashboard.txt` or `pyproject` extra `dashboard`): fastapi, uvicorn, websockets, watchdog, psutil.

**Dashboard frontend** (`dashboard/package.json`): Node.js 18+, React 18, TypeScript, Vite 5, Zustand, React Query, Axios, Recharts.

**Testing** (`requirements-dev.txt`): pytest, pytest-cov, pytest-mock, pytest-timeout.

## Dashboard

**Features**: WebSocket connection with auto-reconnect, pipeline status overview, current activity with progress bars, stage tracking for all 5 stages, model training progress (Optuna trials, CV scores), resource monitoring (GPU memory, CPU, RAM), and a raw JSON debug view.

**API Endpoints**:
- `GET /api/health`: health check
- `GET /api/status`: current pipeline status
- `GET /api/runs`, `GET /api/runs/{run_id}`: run history and details
- `GET /api/reports`, `GET /api/reports/{name}`: stage reports
- `GET /api/logs/{run_id}`: log file
- `GET /api/models`: model comparison data
- `WS /ws/progress`: real-time progress stream

**URLs**: development `http://localhost:5173` (Vite proxies `/api` and `/ws` to port 8000); production `http://localhost:8000`.

## Common Workflows

**New dataset**: place `train.csv`/`test.csv` in `datasets/current/`, run `python src/genML/main.py`, review `outputs/reports/`.

**Model tuning**: adjust `TUNING_CONFIG` (`n_trials`, `timeout`, `models_to_tune`) and hyperparameter ranges in `config.py`; enable ensembles for best performance.

**Debugging**: check console logs, JSON reports in `outputs/reports/`, data in `outputs/data/`, feature artifacts in `outputs/features/`, and the `error` field of failed stage results. The dashboard debug view shows raw progress data.

## Project Statistics

- Python source: ~8,900 lines across 28 files (pipeline ~3,500, features ~2,600, API ~660)
- Dashboard TypeScript: ~900 lines across 7 files
- Tests: ~5,600 lines

Largest files: `training.py` (1,379), `feature_processors.py` (674), `feature_engineering.py` (668), `feature_engine.py` (599), `data_analyzer.py` (520).

## Additional Resources

- `README.md`: user guide, quickstart, dashboard setup
- `SETUP.md`: environment setup
- `dashboard/README.md`: dashboard UI guide
