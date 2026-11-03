# genML

A tabular ML pipeline for Kaggle-style problems (`train.csv` / `test.csv`). You declare what you know about a problem in a small YAML file; the pipeline infers the rest, engineers features, tunes GPU gradient-boosting models, stores out-of-fold predictions, stacks them, and writes a submission in the format the competition's metric needs.

## Why problem specs

The pipeline used to guess everything from the files. On the competitions it was used for, the guesses failed in predictable ways: hard 0/1 labels sent to ROC AUC competitions (the sample submission's placeholder value decided the output type), accuracy used as the tuning metric for AUC problems, and short category labels treated as free text. Now:

- A **problem spec** (`problems/<name>/problem.yaml`) declares target, metric, columns, models and budget. Anything left out is inferred, and every inference is recorded with its reason in the run's `resolved_spec.json`.
- The **metric** decides what is optimized and what the submission contains (labels, probabilities, per-class probabilities, top-k labels or values). The sample submission only validates the result.
- **Golden tests** check inference against every declared problem, so fixing one problem cannot silently break another.

## Quick start

```bash
# Environment (CPU; see SETUP.md for the GPU conda environment)
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-dashboard.txt
VIRTUAL_ENV=.venv uv pip install kagglehub kaggle "mcp>=2"   # optional: Kaggle + MCP
source .venv/bin/activate

# Kaggle credentials: token from https://www.kaggle.com/settings/api
export KAGGLE_API_TOKEN=...            # or save it in ~/.kaggle/access_token

# Create a problem from a Kaggle competition (downloads data, reads the metric)
python src/genML/main.py --init-kaggle playground-series-s6e11 --name fraud-s6e11

# Run it
python src/genML/main.py --problem fraud-s6e11 --notes "baseline"
python src/genML/main.py --list-runs fraud-s6e11
```

Without `--problem`, the legacy mode still works: put `train.csv`/`test.csv` (and optionally `sample_submission.csv`) in `datasets/current/` and run `python src/genML/main.py`; every field is inferred.

## Problem specs

```yaml
name: fraud-s6e11
source: {type: kaggle, competition: playground-series-s6e11}   # or {type: csv, path: datasets/x}
target: is_fraud
id_column: id
task: binary            # binary | multiclass | regression
metric: roc_auc         # accuracy, balanced_accuracy, f1, mcc, roc_auc, log_loss, map_at_3, rmse, rmsle, mae, r2
columns:                # optional role overrides: categorical, numerical, text, datetime, drop
  categorical: [merchant_category]
models: [XGBoost, CatBoost, LightGBM]
budget: {n_trials: 15, timeout_per_model: 900, seeds: 1}   # also: tuning: false, refit_full: true
cv: {n_splits: 5}                                          # method inferred (stratified for classification)
feature_engineering:
  encodings: {target_encode: auto, count_encode: auto, pairs: auto, numeric_as_categorical: auto}
ensemble: {method: stack, top: 10}                         # stack | hill_climb | none
```

Specs for the seven competitions this project has been used on are in `problems/` (Titanic and Playground S5E8 to S6E11).

## How a run works

Each run writes to its own folder, `outputs/<problem>/<run_id>/` (data, features, models, reports, `submission.csv`, `pipeline.log`, `run.json`), and is appended to `outputs/<problem>/runs.csv`.

1. **Load**: resolve the spec against the data and save `resolved_spec.json`.
2. **Feature engineering**: automatic type detection and processing, plus optional leak-safe encodings (out-of-fold target encoding, count encoding, pair crosses, original-dataset means).
3. **Model selection**: models come from the spec.
4. **Training**: Optuna tuning on the problem's metric, then an out-of-fold stage. Every model is retrained on folds shared by all runs of the problem; its OOF and test predictions go to `outputs/<problem>/oof/`. A logistic or Ridge stacker blends the run's models. The final predictions are whichever scores best out of fold.
5. **Prediction**: build the submission from the metric's output type and validate it against `sample_submission.csv` (columns, rows, IDs).

CV scores reported everywhere are out-of-fold scores in the problem's own metric (lower is better for losses such as RMSE).

## Commands

```bash
python src/genML/main.py --list-problems
python src/genML/main.py --problem NAME [--download] [--notes TEXT]
python src/genML/main.py --list-runs NAME
python src/genML/main.py --show-run NAME RUN_ID
python src/genML/main.py --ensemble NAME [--method stack|hill_climb] [--top 10]   # blend across runs
python src/genML/main.py --submit NAME [--run RUN_ID] [--message TEXT]            # submit to Kaggle
python src/genML/main.py --submissions NAME                                       # CV vs leaderboard

python -m src.genML.jobs worker | enqueue NAME | list | cancel JOB_ID              # job queue
```

## Using it from Claude (MCP)

`src/genML/mcp_server.py` exposes the pipeline as MCP tools, so an LLM client can create problems, start runs, poll their status, compare runs, read diagnostics and propose the next experiment. Runs go through a job queue with one GPU worker, so tools return immediately with a `job_id`.

```bash
claude mcp add genml -- /path/to/genML/.venv/bin/python -m src.genML.mcp_server
```

Tools: `list_problems`, `inspect_dataset`, `create_problem`, `start_run`, `start_ensemble`, `get_run_status`, `list_jobs`, `list_runs`, `compare_runs`, `get_run_report`, `get_feature_diagnostics`, `cancel_run`, `submit_to_kaggle` (previews unless `confirm=true`), `get_submissions`. Run logs and reports are available as `genml://runs/...` resources.

## Dashboard

A React dashboard shows live progress (stages, models, Optuna trials, GPU/CPU/RAM).

```bash
python run_api.py                         # API on http://localhost:8000
cd dashboard && npm install && npm run dev  # UI on http://localhost:5173
```

The API also lists problems and runs: `/api/problems`, `/api/problems/{problem}/runs`, and `/api/reports`, `/api/models`, `/api/logs/{run_id}` for any run (`?problem=...&run_id=...`; default is the live or newest run). See `dashboard/README.md`.

## Tests

```bash
pytest                 # everything (includes end-to-end smoke tests on local Kaggle data)
pytest -m "not slow"   # skip the end-to-end runs
```

Golden tests in `tests/problems/` run against each declared problem's data when it is available locally (`datasets/<name>/` or the kagglehub cache) and skip otherwise; Kaggle data is never committed.

## Project layout

```
problems/<name>/problem.yaml   # declared problems
src/genML/
  main.py                      # CLI
  flow.py                      # stage orchestration, run_pipeline()
  run_context.py               # per-run folders and runs.csv
  jobs.py                      # SQLite job queue and GPU worker
  service.py, mcp_server.py    # service layer and MCP server
  submissions.py               # Kaggle submit and CV/leaderboard log
  pipeline/                    # problem specs, metrics, folds/OOF store, ensembling, stages
  features/                    # automatic feature engineering and encodings
  api/                         # FastAPI backend for the dashboard
dashboard/                     # React + TypeScript UI
tests/
```
