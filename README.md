# genML

genML solves tabular prediction problems (a training table and a test table, in one file each or spread over several related files) with as little human input as possible. You declare what you know about a problem in a small YAML file. The pipeline infers the rest, builds leak-safe features, tunes gradient-boosting and neural models on the problem's own metric, stores out-of-fold predictions on folds shared by every run, stacks them, and writes the submission the metric needs.

The **autopilot** runs that pipeline in a loop: Claude Code sessions plan and build experiments, a second model reviews each plan, and trusted code measures every result before anything is kept.

Kaggle competitions are the test bench, not the goal. A gain only counts if the same method would work on real data from the same domain, so genML never uses other people's predictions, never takes ideas from a problem's published solutions or discussions, and never exploits how a dataset was generated. The full rules are in [CLAUDE.md](CLAUDE.md) (Purpose); every decision and change is logged in [DECISIONS.md](DECISIONS.md).

## Quick start

```bash
# Environment (CPU and pip GPU builds; SETUP.md has details and the RAPIDS conda option)
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-dashboard.txt
VIRTUAL_ENV=.venv uv pip install kagglehub kaggle "mcp>=2" jsonschema
source .venv/bin/activate

# Kaggle credentials: token from https://www.kaggle.com/settings/api
export KAGGLE_API_TOKEN=...            # or save it in ~/.kaggle/access_token

# Create a problem from a Kaggle competition, download its data and run a baseline
python src/genML/main.py --init-kaggle playground-series-s6e11 --name fraud-s6e11
python src/genML/main.py --problem fraud-s6e11 --download --notes "first look"
python src/genML/main.py --baseline fraud-s6e11      # one run + a seed test: the judge's noise

# Let the autopilot improve it (needs the claude CLI, bubblewrap and socat; see SETUP.md)
python src/genML/main.py --autopilot fraud-s6e11 --cycles 10
python src/genML/main.py --autopilot-status fraud-s6e11
```

Without `--problem`, the legacy mode still works: put `train.csv` and `test.csv` (and optionally `sample_submission.csv`) in `datasets/current/` and run `python src/genML/main.py`; every field is inferred.

## Problem specs

```yaml
name: fraud-s6e11
source: {type: kaggle, competition: playground-series-s6e11}   # or {type: csv, path: datasets/x}
target: is_fraud
id_column: id
task: binary            # binary | multiclass | regression
metric: roc_auc         # accuracy, balanced_accuracy, f1, mcc, roc_auc, normalized_gini, amex,
                        # log_loss, map_at_3, rmse, rmsle, mae, r2
columns:                # optional role overrides: categorical, numerical, text, datetime, drop
  categorical: [merchant_category]
models: [XGBoost, LightGBM, CatBoost]
budget: {n_trials: 15, timeout_per_model: 900, seeds: 1}
cv: {method: stratified_kfold, n_splits: 5}   # or group_kfold / stratified_group_kfold (cv.group),
                                              # time_blocks (cv.time_column)
feature_engineering:
  base: raw             # original columns plus the encodings below
  encodings: {target_encode: auto, count_encode: auto, pairs: auto}
ensemble: {method: stack, top: 10}            # stack | hill_climb | none
```

Anything left out is inferred, and every inference is recorded with its reason in the run's `resolved_spec.json`. Data spread over several files is declared under `tables:` (a main table plus child tables joined one to one or rolled up per key, for example a customer's statements); see `problems/home-credit/problem.yaml`. `problems/` holds eleven specs: Titanic, Playground S5E8 to S6E11, Home Credit, IEEE-CIS Fraud, AmEx Default and Porto Seguro.

## How a run works

Each run writes to its own folder, `outputs/<problem>/<run_id>/` (data, features, models, reports, `submission.csv`, `pipeline.log`, `run.json`), and is added to `outputs/<problem>/runs.csv`.

1. **Load**: read the data (assembling multi-table problems with Polars), resolve the spec, save `resolved_spec.json`.
2. **Features**: type detection and processing, label-free derived features (arithmetic expressions, composite keys, past-only group statistics, a train/test drift filter) and leak-safe encodings. Target encodings are built per fold, so no fold's labels reach its own model.
3. **Training**: Optuna tunes each model on the problem's metric with early stopping, then every model is refit on the shared folds; its out-of-fold and test predictions go to `outputs/<problem>/oof/`. A logistic or ridge stacker blends the run's models, and the final predictions are whichever scores best out of fold. Models: XGBoost, LightGBM, CatBoost, GLM-LightGBM, logistic or linear regression, random forest, and (with the `nn` extra) RealMLP, TabM and TabPFN.
4. **Prediction**: build the submission from the metric's output type and check it against `sample_submission.csv`.

Scores reported everywhere are out-of-fold scores in the problem's own metric.

## The autopilot

Each cycle is resumable at every step (`outputs/<problem>/autopilot/state.json`):

1. **Plan** (or **explore** every third cycle, an idea from its own probes of the data): a Claude Code session reads what was measured so far, screens several ideas with quick probes (one fixed LightGBM on the best run's features, a minute or two each, on rows the judge does not decide on), and picks one experiment, either a settings change or new code.
2. **Review**: a second opinion from a fresh model that has none of the planning session's context (Claude; Codex can be put first again for a different model family) checks the plan against the rules and the evidence. On concerns the plan gets one revision; a revised plan rejected again fails the cycle.
3. **Build** (code ideas): a session implements the idea in its own git worktree, with tests.
4. **Verify**: edits to protected paths (the judge, metric, folds, the autopilot itself, existing tests) are reverted, new tests are required, the fast suite must pass.
5. **Run**: a normal pipeline run with the cycle's code.
6. **Judge**: trusted code checks that the folds, target and best predictions are unchanged, then accepts the change only if its gain beats the run-to-run noise (measured by `--baseline`) and wins most folds. It judges on 80% of the rows; the other 20% never decide anything and show, at the end, whether the accumulated gains were real. Time-ordered problems are judged forward in time (only the latest block, whose models saw only the past). Gains far above the noise pause the loop for a person, with a second opinion on whether it is a leak.
7. **Canary** (accepted code): the same code runs on a copy of the data with the labels shuffled. If it still beats a constant prediction, labels are leaking in somewhere and the change is rejected.
8. **Integrate**: accepted code moves onto the shared branch `autopilot/main` (one code base for every problem, refreshed from your working tree at each start); every cycle is written to `journal.md` and `outputs/<problem>/experiments.json`.

Runs built on banned ideas or a known leak are listed in `problems/<name>/excluded_runs.yaml` (`--exclude-run`); nothing picks them as a baseline, a best run or a blend member.

What each step may do is declared in the header of its prompt file (`.claude/skills/genml/modes/`) and enforced, not asked for: sessions get only their listed tools, no internet and no credentials, and run in Claude Code's sandbox; pipeline runs of agent code run in bubblewrap with the judge's files read-only. Without a working sandbox the autopilot does not start. Agents never submit to Kaggle; submitting is always a person's decision.

In Claude Code, the `/genml` skill starts, checks, steers and stops the autopilot.

## Commands

```bash
python src/genML/main.py --list-problems
python src/genML/main.py --problem NAME [--download] [--notes TEXT]
python src/genML/main.py --list-runs NAME | --show-run NAME RUN_ID
python src/genML/main.py --ensemble NAME [--method stack|hill_climb] [--top 10]     # blend across runs
python src/genML/main.py --baseline NAME [--repeats N]                              # baseline + noise
python src/genML/main.py --noise-test NAME --run RUN_ID --model XGBoost [--seeds 3]  # seed noise
python src/genML/main.py --exclude-run NAME RUN_ID --reason TEXT                    # a run that does not count
python src/genML/main.py --compare NAME EXPERIMENT_A EXPERIMENT_B                    # is B really better?
python src/genML/main.py --submit NAME [--run RUN_ID] [--message TEXT]              # submit to Kaggle
python src/genML/main.py --submissions NAME                                         # CV vs leaderboard

python src/genML/main.py --autopilot NAME --cycles 10 [--directive TEXT] [--until ISO] [--no-second-opinion]
python src/genML/main.py --autopilot-status NAME | --autopilot-stop NAME
python src/genML/main.py --queue-directive NAME TEXT                                # steer a later cycle
python src/genML/main.py --second-opinion NAME [--cycle C003]                       # a second model reviews a cycle
python src/genML/main.py --llm-status                                               # which CLIs can answer

python -m src.genML.jobs worker | enqueue NAME | list | cancel JOB_ID                # job queue
```

## Using it from an MCP client

`src/genML/mcp_server.py` exposes the pipeline as MCP tools, so an LLM client can create problems, start runs, poll their status and compare runs. Runs go through a job queue with one GPU worker, so tools return immediately with a `job_id`.

```bash
claude mcp add genml -- /path/to/genML/.venv/bin/python -m src.genML.mcp_server
```

Tools: `list_problems`, `inspect_dataset`, `create_problem`, `start_run`, `start_ensemble`, `get_run_status`, `list_jobs`, `list_runs`, `compare_runs`, `get_run_report`, `get_feature_diagnostics`, `cancel_run`, `submit_to_kaggle` (previews unless `confirm=true`), `get_submissions`.

## Dashboard

A React dashboard shows live progress (stages, models, Optuna trials, GPU/CPU/RAM) and lists problems and runs.

```bash
python run_api.py                           # API on http://localhost:8000
cd dashboard && npm install && npm run dev  # UI on http://localhost:5173
```

## Tests

```bash
pytest -m "not slow"   # the fast suite (a few minutes)
pytest                 # also end-to-end runs on local Kaggle data
```

Golden tests in `tests/problems/` run against each declared problem's data when it is available locally and skip otherwise; Kaggle data is never committed. Tests never call a real LLM: every backend is faked.

## Project layout

```
problems/<name>/                 # problem.yaml and notes.md per problem
DECISIONS.md                     # what was decided, why, and what changed
src/genML/
  main.py                        # CLI
  flow.py, run_context.py        # the four stages, per-run folders
  jobs.py                        # SQLite job queue and GPU worker (agent code runs sandboxed)
  experiments.py                 # settings checks and the experiment ledger
  autopilot/                     # loop, judge, worktrees, permissions, dossier, second opinion
  llm/                           # claude and codex CLIs as JSON-answering backends
  pipeline/                      # specs, metrics, tables, folds and OOF store, training, ensembles
  features/                      # type detection, encodings, derived features
  service.py, mcp_server.py      # service layer and MCP server
  api/                           # FastAPI backend for the dashboard
.claude/skills/genml/            # the /genml skill and the autopilot's prompt files
dashboard/                       # React + TypeScript UI
tests/
```
