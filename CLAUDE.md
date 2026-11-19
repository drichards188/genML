# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

genML is a tabular ML pipeline for Kaggle-style problems (a train and a test table, in one file each or spread over several related files). Each problem is declared in `problems/<name>/problem.yaml`; unset fields are inferred and every inference is recorded. Runs tune gradient-boosting and neural models on the problem's own metric, store out-of-fold (OOF) predictions on folds shared by all runs, stack them, and write a submission whose content is decided by the metric. The intended way to use it is the **autopilot**: Claude Code sessions plan and build experiments, a second model reviews each plan, and trusted code judges every result. Runs go through a job worker and can also be driven by LLM clients through an MCP server. A FastAPI + React dashboard shows live progress. Decisions and changes are logged in `DECISIONS.md`.

**Status: the project ended on 2026-11-18 (the owner's call).** No part of it measured better than
existing tools; README.md is the retrospective and DECISIONS.md has the reasons. Do not start new features,
referee work or autopilot runs unless the owner reopens the project. The rest of this file describes the
code as it stood at the end.

## Purpose: a process that solves real problems

genML exists to solve real tabular problems with as little human input as possible. Kaggle competitions,
Playground included, are only a test bench for that process: the leaderboard rank is not the goal, and a
gain counts only if the same method would work on real data from the same domain. So:

- Never pool other people's public OOF/test predictions or submissions. Measure only models genML trains.
- Never use ideas from the problem's published solutions, write-ups, discussions or public notebooks
  (for a past competition, nothing learned from its winners). Ideas must come from general domain or ML
  knowledge, or from genML's own analysis of the data, so a score measures genML and not hindsight.
- Never reverse-engineer how a dataset was made: no modelling of a synthetic-data generator (for example
  GPT-2/GReaT row-as-text likelihoods), no exploiting tokenizer or rounding artifacts, row order, ids or leaks.
- Features need a reason a domain practitioner would accept (round transaction amounts, card velocity),
  not only a higher CV on this dataset.
- Validation must match how the model would be used (time or group splits when the data has that
  structure). A gain that exists only because of the split does not count.
- Prefer general capabilities (model families, stacking, validation schemes, data handling) over
  problem-specific tricks. When unsure whether something is a hack, leave it out and say so.

## Key Commands

### Environment
```bash
# CPU environment (enough for all tests); .venv is gitignored
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-dashboard.txt
VIRTUAL_ENV=.venv uv pip install kagglehub kaggle "mcp>=2"   # Kaggle download/submit, MCP server
VIRTUAL_ENV=.venv uv pip install jsonschema                 # schema checks of second-opinion answers
VIRTUAL_ENV=.venv uv pip install torch pytabkit tabpfn        # optional RealMLP, TabM, TabPFN (extra: nn)
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
python src/genML/main.py --baseline fraud-s6e11 [--repeats N]                                # baseline run + the judge's noise
python src/genML/main.py --noise-test fraud-s6e11 --run RUN_ID --model XGBoost [--seeds 3]   # seed noise
python src/genML/main.py --exclude-run fraud-s6e11 RUN_ID --reason "..."                   # a run that does not count
python src/genML/main.py --prune-runs fraud-s6e11 [--dry-run]                             # free disk of runs nothing needs
python src/genML/main.py --compare fraud-s6e11 EXPERIMENT_A EXPERIMENT_B                    # is B really better?
python src/genML/main.py --llm-status                                                       # claude / codex CLIs usable here
python src/genML/main.py --second-opinion fraud-s6e11 [--cycle C003] [--backend claude-cli]  # a second model reviews an autopilot cycle
python src/genML/main.py --autopilot fraud-s6e11 --cycles 10 [--directive "..."] [--until ISO] [--allow-unsandboxed]   # plan/build/test loop (needs bubblewrap + socat)
python src/genML/main.py --autopilot-status fraud-s6e11 | --autopilot-stop fraud-s6e11
python src/genML/main.py --research fraud-s6e11 | --show-research fraud-s6e11   # cited domain brief (problems/<name>/research.yaml)
python src/genML/main.py --init-kaggle playground-series-s6e11 --name fraud-s6e11
python src/genML/main.py            # legacy: datasets/current/, everything inferred

python -m src.genML.jobs worker | enqueue NAME | list | cancel JOB_ID
python -m src.genML.referee create --problem ieee-fraud --name ieee-100k --sample 100000   # a contest
python -m src.genML.referee enter ieee-100k ag-medium --seeds 0 1 2 --time-limit 600 -- \
    ~/.venvs/genml-autogluon/bin/python contestants/autogluon_contestant.py --presets medium_quality
python -m src.genML.referee summary|check ieee-100k | compare ieee-100k BASE CANDIDATE
python -m src.genML.referee.benchmark [--contest ieee-100k] [--skip-autogluon]   # noise, planted gains and leaks
python -m src.genML.referee.seed_study --contest ieee-100k --entry ag-medium       # is single-run DeLong fooled by seeds?
python -m src.genML.referee.power_study [--contest ieee-100k] [--skip-runs]        # referee against simpler tests
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
- `ProblemSpec`: fields `name, description, source, target, id_column, task, metric, positive_class, cv, submission, columns, models, budget, feature_engineering, ensemble, tables`. `None` means infer.
- `cv.method`: `kfold`, `stratified_kfold` (defaults), `group_kfold` / `stratified_group_kfold` (`cv.group`: a column whose values never span folds, e.g. a customer or month) or `time_blocks` (`cv.time_column`: contiguous blocks in time order; the last block is reported as `forward_holdout` in `final_predictions.json`, and tuning uses the latest blocks).
- `tables` (`pipeline/tables.py`): data spread over several files. One main table (`train`/`test` files) and children with `parent`, `key` (shared key column; not `on`, which YAML reads as a boolean), `relation: one` (left join) or `many` (rolled up per key: count, numeric mean/std/min/max, categorical nunique, plus last/trend/`last_n` with `order_by`; chains roll up bottom-first; `aggregate.partitions` rolls up key slices one at a time for very large tables). Built with Polars; each CSV is cached as float32 Parquet and the assembled tables in `outputs/<problem>/tables_cache/<hash>/`. Test columns differing from train only in punctuation are aligned. `locate_data_dir` then looks for the main table's files instead of train.csv/test.csv.
- `resolve_problem()` fills every field and records `inferred[field] = reason`; the result (`ResolvedProblem`) is saved as `reports/resolved_spec.json` in the run folder.
- Inference rules (`pipeline/inference.py`): task follows AutoGluon (2 values binary; non-numeric multiclass; non-integer floats regression; integers multiclass only with few distinct values). Target is the train-only column, chosen deterministically.
- Metric is never inferred from sample-submission placeholders when declared; without a declaration `metrics.default_metric_for()` uses the sample only as a hint.
- `locate_data_dir()`: `source.path`, then `datasets/<name>/`, then the kagglehub cache, then legacy discovery.
- `GENML_PROBLEMS` and `GENML_OUTPUTS` env vars override the problems and outputs folders (used by tests and worker subprocesses).

### Metrics (`src/genML/pipeline/metrics.py`)
One registry entry per metric: sklearn scorer per task (higher is better), output kind (`label`, `probability`, `class_probabilities`, `top_k`, `value`), `greater_is_better`, Kaggle names, optional target transform (`rmsle` trains on log1p). `Metric.score()` scores OOF predictions on the metric's own scale. Add new metrics here; everything else (tuning, OOF scoring, submissions) follows. Includes `normalized_gini` (2 x AUC - 1) and `amex` (Amex default metric; matches the competition's reference implementation).

### Run flow (`src/genML/flow.py`)
`run_pipeline()` creates a `RunContext`, then `MLPipelineFlow.kickoff()` runs four stages in order (`STAGES`): load data, feature engineering, training, prediction (models come from the spec). A failed stage raises `PipelineAbort`; the run is still recorded as failed. CrewAI was removed; the flow is plain Python.

### Run folders (`src/genML/run_context.py`)
`RunContext.activate()` creates `outputs/<problem>/<run_id>/{data,features,models,predictions,reports}` and points `config.DATA_DIR` and friends at it. `finish()` writes `run.json` and appends to `outputs/<problem>/runs.csv`. Path globals are per process; concurrent runs are separate worker processes. The loaded tables travel between stages as `data/{train,test}_data.parquet` (`dataset.save_stage_tables` / `load_stage_table`; both as pickle when Parquet refuses a frame; older runs' pickles still load; missing text reads back as NaN, as from a CSV).
Pruning (`src/genML/pruning.py`, `--prune-runs`): deletes a run's `data/` tables (keeping `data/test_ids.parquet`, which a cross-run ensemble's submission needs) and its big matrices (`features/X_*.npy`, `fold_*.npy`) and writes `pruned.json`. Kept in full: the current best run, every run named in the autopilot state's anchors and best, runs an active job uses (its run, a noise job's `run_id`), and unfinished runs changed in the last 12 hours. The job worker prunes the job's problem after every job; the autopilot prunes a canary run and deletes its shuffled data once the verdict is recorded. Reports, predictions, the submission, `final_test_predictions.npy` and the OOF store always stay.

### Feature engineering
- `features/feature_engine.py` (`AutoFeatureEngine`), `data_analyzer.py` (type detection; AutoGluon text gate: string columns need >= 3 words on average and unique ratio > 0.01 to count as text), `feature_processors.py`, `feature_selector.py` (selection runs on at most `selection_max_rows`, default 100k).
- `features/encoding.py`: opt-in via `feature_engineering.encodings`; out-of-fold target encoding (shared folds), count encoding, pair crosses, numeric-as-categorical (`numeric_as_categorical: all` = per-value target encoding of every numeric column), digit keys (`digits: auto` and `digit_keys`: integer part, int/10, tenths, cents, leading digits of decimal columns, each target-encoded; `encoding.digit_key()`), original-dataset means. Target encoding uses integer level codes and `np.bincount` (`_codes`, `_level_means`). Target encodings are also nested per fold (`nested: true`, the default): `nested_target_encodings()` builds one block per outer fold k that never uses fold k's labels, saved as `features/fold_train_<k>.npy` / `fold_test_<k>.npy` + `fold_blocks.json`. `oof.FoldFeatures` swaps them in for every fold-k fit (tuning and OOF). The single matrix in `X_train.npy` would leak fold k's labels into the training rows when fold k validates; it is kept only for refitting on all rows.
- `feature_engineering.base: raw` skips the automated transforms (scaled/log/squared/binned copies) and uses the original columns (non-numeric as integer codes) plus encodings.
- `features/derived.py`: label-free features under `feature_engineering`: `derived` (columns from arithmetic expressions, parsed with an AST whitelist, never executed as code), `keys` (composite keys as categoricals), `group_stats` (per-group count/mean/std/min/max/nunique/deviation over train+test rows; past-only, each row seeing only its group's rows up to its own time, whenever `time_column` is set; `past_only: false` must be explicit) and `drift_filter` (drops columns whose single-column adversarial AUC train-vs-test exceeds the threshold). Applied before type analysis; target, id and the cv column are never dropped. Validated by `experiments.check_patch` so the AI can use them.
- `pipeline/targets.py`: label encoding (with decode), log1p for RMSLE, logit for bounded regression targets, and inverses.

### Training (`src/genML/pipeline/training.py`, `tuning.py`)
Optuna tunes each model on `problem.scoring`, within `budget` (`n_trials`, `timeout_per_model`, `tuning: false`). XGBoost, LightGBM and CatBoost share `tuning.optimize_gbdt`: it scores on the problem's own folds (the first `budget.tuning_folds`; default all, or 2 above 200k rows) with nested encodings, uses early stopping (ceiling `MAX_TREES`, or `budget.max_trees`), and the mean early-stopped tree count becomes a fixed parameter for the OOF stage, so OOF rows never decide when their own model stops. Search spaces (`GBDT_SPACE`) go down to stumps (depth 1, 2 leaves); `budget.search_space: {Model: {param: [low, high]}}` moves a range (`gbdt_space`, `check_search_space`, limits in `GBDT_LIMITS`). `GLM-LightGBM` (`pipeline/glm_boost.py`, binary only) fits a logistic regression on the features (target-encoded `*_te` columns as logits) and boosts LightGBM from its logit (`init_score`); it is tuned like LightGBM (`GBDT_KINDS`) and finds the encoded columns through `feature_names.txt` (`training.load_feature_names`). Untuned models are scored with `tuning.fold_cv_score` on the same folds. Folds come from `oof.run_folds`: random and stratified folds as before (their fingerprint is unchanged), grouped and time folds from the cv column's values in the feature stage, which records the signature in `features/fold_signature.txt` for later stages; sklearn-based tuning then gets them as a `PredefinedSplit`. `pipeline/neural.py` adds RealMLP and TabM (pytabkit, NaN imputed, RealMLP epochs scaled by data size via `default_epochs`) and TabPFN (pretrained, no training; a stratified `max_context` sample of each fold as context, `context_bags` samples averaged). They are not Optuna-tuned: fixed parameters come from `budget.model_params` (`neural.MODEL_PARAM_RULES`, checked by `check_model_params`), they are screened on the tuning folds only, and they register only when usable (`neural.available_models()`, also in the advisor dossier as `models_available`). TabPFN's weights download only after the owner accepts Prior Labs' license and sets `TABPFN_TOKEN`. `training.build_model_factories()` builds every model factory (also used by diagnostics). Then `_run_oof_stage()`: fixed folds (`pipeline/oof.py`, cached per problem as `folds_<signature>.npy`), `fit_oof()` per model (and per seed with `budget.seeds`), experiments saved to `outputs/<problem>/oof/`, scored with the native metric. The run's ensemble (`pipeline/ensemble.py`: `stack` with nested CV, or `hill_climb`) competes with the best single model on OOF score; the winner's test predictions go to `features/final_test_predictions.npy`. `budget.refit_full` refits the best single model on all rows instead.

XGBoost GPU uses `tree_method="hist", device="cuda"` (`gpu_hist` no longer exists in XGBoost 3).

### Submissions (`src/genML/pipeline/submission.py`, `src/genML/submissions.py`)
`build_submission()` writes what the metric needs; `validate_submission()` checks columns, row count and IDs against `sample_submission.csv`. `submissions.submit_run()` tags the Kaggle description with `[genml <run_id>]` and logs CV in `outputs/<problem>/submissions.csv`; `refresh_scores()` pulls public/private scores back.

### Diagnostics (`src/genML/pipeline/diagnostics.py`)
Whether an OOF gain is real. `paired_compare()` bootstraps the score difference between two experiments on the shared rows (row noise) and gives per-fold differences; `seed_noise_test()` refits a run's model with extra seeds (reusing its features and tuned params) and writes `reports/noise_test.json` with the seed spread. `verdict()` accepts B over A only if the gain exceeds `3 * sqrt(se_rows**2 + 2 * sigma**2)` and B wins at least 80% of folds; `compare_experiments()` uses the newest noise test of the problem. `queue_baseline_runs()` (`--baseline`) queues one run plus a `noise` job (the seed test on its best single model, `best_single_model()`), or with `repeats` of 2 or more, runs of the same settings with `budget.seed` 0..n-1 (`training.run_seed`: the Optuna sampler, the models built while tuning and the OOF seeds move together; experiments cannot set it) and `run_noise()` reads their final scores from `noise_runs.json`: that spread includes the tuning path, which a seed-only refit misses.

### LLM connection (`src/genML/llm/`)
- `llm.ask(system, prompt, schema, purpose=, problem=, backends=)` returns a dict matching a JSON schema. Two backends (`llm/backends.py`), both subscription CLIs that bill your plan, never per token: `claude-cli` (`claude -p --json-schema`, no tools, no settings or MCP servers, custom system prompt, API keys stripped from its env) and `codex-cli` (`codex exec --output-schema`, read-only sandbox in an empty temp dir). They are tried in order (`config.LLM_CONFIG['order']`, `claude-cli` only since 2026-11-17, when Codex was no longer signed in; or `backends=`); unavailable, failed or off-schema answers fall through to the next. Overrides: `GENML_LLM_ORDER`, `GENML_CLAUDE_MODEL`, `GENML_CODEX_MODEL`. The paid-API backends were removed on 2026-11-10.
- Every call is logged in `outputs/<problem>/llm/` (full request/answer JSON plus `calls.jsonl`).
- Tests fake every backend (`tests/test_llm.py`, `tests/test_second_opinion.py`); never call real CLIs from tests.

### Settings checks and experiment ledger (`src/genML/experiments.py`)
`check_patch()` allows only `models`, `budget`, `feature_engineering` and `ensemble` with known keys and value ranges (never `cv`: folds must stay shared). Changes merge into the settings of the best run so far (its `resolved_spec.json`; `baseline_settings()`, `deep_merge()`), so accepted changes carry forward without editing problem.yaml; `fingerprint()` and `_changes()` detect no-ops and repeats. `record_entry()` / `load_ledger()` keep `outputs/<problem>/experiments.json` (one entry per measured idea, predicted and measured gain); `experiment_log()` is the compact form the dossier shows; `current_best()` is the completed run with the best final OOF score, skipping runs listed in `problems/<name>/excluded_runs.yaml` (`excluded_runs()`, `exclude_run()`, `--exclude-run`: runs built on banned ideas or a known leak; also skipped by store ensembles, marked in `--list-runs`, the dossier and MCP `list_runs`, and never the default for `--submit`). `check_patch()` always refuses `encodings.nested: false`, `base: auto` and (with `target=`) derived features, keys or group statistics that read the target; `leak_problems()` applies the same rules to complete settings. Entries E001 to E003 come from the old run advisor (removed 2026-11-10); later ones from autopilot cycles.

### Dossier and second opinion (`src/genML/autopilot/dossier.py`, `second_opinion.py`)
- `build_dossier()` collects what was measured (scores, paired comparisons, seed noise, `diagnostics.range_edges`, prediction rank correlations, run history, the experiment log, the leaderboard, `problems/<name>/notes.md`) plus `SPEC_CAPABILITIES` (the settings a plan can change without code) and `models_available`. It only reads. The loop writes it to each cycle's context folder.
- `second_opinion.ask_second_opinion(problem, question, material)` asks a fresh model with no tools and none of the sessions' context (`second_opinion.ORDER`: `claude-cli`; put `codex-cli` first again for a different model family when a ChatGPT plan is signed in) for a verdict `ok | concerns | reject` with typed concerns (`banned_source`, `leak`, `generator_artifact`, `no_domain_reason`, `weak_evidence`, `repeat`, `cost`, `other`). Its system prompt embeds the `## Purpose` section of `modes/_shared.md`, so both models read the same rules. Questions: `PLAN_QUESTION` (the loop's review step), `GAIN_QUESTION` (suspicious gains), `CYCLE_QUESTION` (`--second-opinion`, `loop.review_cycle`).

### Autopilot (`src/genML/autopilot/`, skill `.claude/skills/genml/`)
A loop of headless Claude Code sessions on the subscription (`claude -p` with tools) that can write code, judged by trusted code. Each cycle (`loop.Autopilot.run_cycle`) is resumable at every step from `outputs/<problem>/autopilot/state.json`:
- **plan** (or **explore** every `explore_every`-th cycle: an idea from its own probes of the data and OOF errors, never a known public one) writes `.autopilot/plan.json`. Before choosing, the session screens ideas with the probe harness (`autopilot/probe.py`): `probe.load()` reads `.autopilot/probe_context.json` and `probe_rows.npy`, which the loop writes from the best run (`probe.write_context`), and `compare(label, add=, drop=, weights=, variant_params=)` fits one fixed LightGBM (two seeds, about 1 to 2 minutes on ieee-fraud) on the best run's own feature matrix with its nested encodings, and returns the gain, its noise and the gain as a share of the judge's bar. Probes never score a held-back row, and for time splits never use the judge's block (they train on the blocks before the second-to-last and score that block; `probe.probe_rows`), so screening many ideas does not wear out the judge's rows; for random folds they score half of fold 0 so the codes do not reveal the held-back rows. Each call is logged in the worktree's `.autopilot/probes.jsonl`; the loop copies it after every session to `outputs/<problem>/autopilot/probes.jsonl` (`_collect_probes`), plans cite probes in `idea.probe_ids` (validated), the ledger records `probe_gain`, the reviewer sees `probe_results`, and `.autopilot/probe_history.json` shows later cycles earlier probes next to what their runs measured. Probe numbers are advisory; the judge decides. The dossier's `measurements.judge_bar` (`loop._judge_bar`: the last judged cycle's bar, else a seed-noise lower bound) is the bar on the judge's own rows; the seed-noise threshold beside it is over all rows and lower.
- **review** (`step_review`, unless `--no-second-opinion`): the second opinion on the plan. On `concerns` or `reject` the plan session gets one revision (mode `revise`, the old plan kept as `plan_original.json`); a revised plan rejected again fails the cycle; a reviewer that cannot answer lets the cycle go on.
- **build** (code ideas) implements it in a per-cycle git worktree with tests; **verify** reverts edits to `workspace.PROTECTED` paths (judge, metric, folds/OOF store, autopilot, `.claude/`, `problems/`, `tests/conftest.py`, the protected test files) and to tests that existed at the cycle start, commits, requires new tests, runs the fast suite, and resumes the build session to **fix** failures (`max_fix_attempts`).
- **run** queues a normal job with `options.code_dir` = the worktree. Settings ideas first get `workspace.reset_worktree` (only verified code runs) and every patch passes `experiments.check_patch` (code ideas may name new settings, `known_only=False`) and `leak_problems` on the merged settings.
- **judge** (`judge.judge_run`): integrity checks against the anchors, then a paired comparison with the current best on the **acceptance rows** only; the bar is `3 * sqrt(se_rows**2 + 2 * sigma**2)` with `sigma` the run-to-run noise, plus winning 80% of the folds. Gains above `suspicious_ratio` (10) times the bar, or `--suspicious-gain`, pause the loop for a person with a second opinion on whether it is a leak (`second_opinion_gain.json`).
- **canary** (code cycles judged accepted or suspicious, unless `canary=False`): `canary.prepare_canary_data` copies the data with the target column shuffled (every other column untouched, read and written as text) and the same code and settings run on it into their own outputs root (job options `source` and `outputs_root`); `judge.canary_verdict` rejects the cycle if the run beats a constant prediction (a leak), and holds it for a person if the canary cannot finish.
- **integrate** puts accepted code on the shared branch `autopilot/main` (fast-forward, or a merge when another problem's loop moved it; a conflict pauses), records the cycle in `experiments.json` (`experiments.record_entry`) and `journal.md`.

Domain research (`autopilot/research.py`, `research_gate.py`, mode `research.md`; `--research`, `--show-research`, `--no-research`): when `problems/<name>/research.yaml` exists (`domain`, `sources_before`, optional `extra_hosts`, `max_items`), the loop runs it once before the cycles (again when the file changes; a failure is logged and the cycles go on). It is the only session with web access, and it is built so it cannot reach answers: it is **blind** (it gets only the owner's domain description, which `blindness_problems` refuses if it names the problem, its competition, data files, columns or the target; it runs in an empty folder with a random name, with no shell), it reads **reference sources only** (WebFetch allowed to OpenAlex, Semantic Scholar and arXiv APIs, arXiv pages, old Wikipedia revisions and owner-listed hosts; no web search; answer hosts such as Kaggle, GitHub and blogs denied, and refused in `extra_hosts`), and only **sources older than the problem** (`sources_before`, a competition's start date). A PreToolUse hook (`research_gate.py`, stdlib only, fails closed, self-tested before each session; run from a read-only copy in `~/.cache/genml-autopilot/gate/`, which the loop makes for a worktree before each session because a sandboxed session cannot write there, with the interpreter's real path, so the command never shows the checkout's path, which in a worktree names the problem) checks every fetch: searches must carry the date limit and single records are looked up and must be older; it is needed because Claude Code lets some documentation sites through without a permission rule. Afterwards trusted code audits the session log (every successful fetch must pass the gate again, or the brief is refused), checks each cited source's date (`check_citation`; `openalex:`, `doi:`, `arxiv:`, `s2:`, `wikipedia:<oldid>`, `url:` for owner hosts), and a second model (`second_opinion.ORDER`) drops dataset artifacts and items that name a competition. The brief (`outputs/<problem>/research/brief.json`, session record in `research/sessions/`) reaches every plan, explore and revise session as `domain_brief.json`; plans cite items in `idea.brief_items` (validated against the brief, recorded in the ledger, shown to the plan's reviewer). A brief item is a reason to test an idea, not evidence that it helps.

Anchors (`judge.make_anchors`, at the first start): the best clean run (`experiments.current_best`, excluded runs skipped); its settings must pass `leak_problems` (`base: raw`, nested encodings, nothing reads the target); a finished noise measurement is required (`--baseline`: by default one run followed by a `noise` job, the seed test, whose per-seed predictions `noise_test_oofs.npy` are rescored on the acceptance rows as `sigma_seed`; with `--repeats 2` or more, the spread of the seeded runs on the acceptance rows is `sigma_run`, which also covers the tuning path; the bar uses the larger); folds, target and transform hashes; the row split `autopilot/rows.npy` (`choose_rows`: random and grouped folds hold out a stratified 20% of rows; time blocks judge only the latest block, whose models saw only the past, half deciding and half held out). Holdout scores (`holdout_log`, `holdout_best`, shown by `--autopilot-status`) never decide anything and are kept out of what agents read.

Code base: one shared branch `autopilot/main` for every problem. Each start (`refresh_code_base`, `workspace.refresh_shared_base`) snapshots the owner's working tree (uncommitted work included; no index, HEAD or branch changes) and moves the branch to it, or merges it in when the branch has accepted code the owner has not merged yet (a conflict refuses to start: merge `autopilot/main` first).

Permissions are declared, not prompted (`autopilot/policy.py`): each loop mode file starts with a YAML header (`tools`, `bash` commands, `read` folders, `network: none`, `credentials: hidden`, `sandbox: required`) that `agent.run_agent` turns into `claude -p --restricted --tools ... --settings <file>`; the settings file (saved next to the session log) holds allow/deny rules and Claude Code's sandbox (no network, credential files unreadable, writes only in the worktree, `failIfUnavailable`; the session keeps Claude Code's own temp folder because the sandbox's sockets fail on long paths). Runs of agent code follow `autopilot/run_policy.yaml`: the job worker (`jobs._launch`) wraps them in bubblewrap (no network, credentials masked and removed from the environment, writes only to `outputs/<problem>/` minus `autopilot/`, `outputs/progress/`, the worktree and a private temp folder that `TMPDIR` points at) and hands over the job as a JSON file (`jobs execute-json`). Without bubblewrap and socat the loop refuses to start and agent-code jobs fail, unless the owner passes `--allow-unsandboxed`.

Prompts are the mode files in `.claude/skills/genml/modes/` (`_shared.md` is added to every loop session, never to the research session, which must not learn the problem); problem and machine details in them are placeholders filled by `loop.Autopilot._facts` (rows, data files, fold counts, run time, RAM, GPU). The protected tests `test_autopilot_policy.py`, `test_second_opinion.py`, `test_honest_judging.py`, `test_autopilot_ops.py` and `test_probe.py` pin these rules. Usage limits are waited out and the same session resumed. Worktrees and canary data live in `~/.cache/genml-autopilot/<repo>/`. Agents never submit to Kaggle.

### Referee (`src/genML/referee/`, `contestants/`)
Scores other tools' models honestly. A **contest** (`contest.py`, `outputs/referee/<name>/`) splits a real
training table (a declared problem's, tables joined as genML does, optionally sampled): with a time column
the latest of `cv.n_splits` blocks is hidden, so contestants predict forward in time; otherwise a stratified
20% is hidden. `public/` holds `train.parquet`, `hidden.parquet` (target removed) and `contest.json`;
`private/` holds the hidden labels and the row codes (half the hidden rows decide, half are holdout that is
reported but never decides), and the hashes of the public files, checked on every load. A **contestant**
(`arena.run_entry`) is any command given `--contest --out --seed --workdir [--time-limit]`; it writes one
prediction per hidden row and may write `self_report.json` (its own validation score). It runs under
bubblewrap with no network, the private folder and other entries replaced by empty folders, credentials
masked, writes only in its entry folder, at `nice 15`, in a memory-capped scope. Predictions edited after
the run, or entries made before a public file changed, are not counted. The **verdict** (`verdict.compare`)
scores seed-averaged predictions on the deciding rows; `se_total = sqrt(se_rows**2 + sd_a**2/n_a +
sd_b**2/n_b)` (paired bootstrap plus each entry's per-seed spread) and a graded result: `proven` (> 3
se_total, 80% of 5 row chunks), `likely` (> 2 se_total, 60%; also the ceiling when no seed noise was
measured), `no_evidence`, `worse`, `suspicious` (> 10x the proven bar). It reports `smallest_provable_gain`,
the holdout difference (flagged when it disagrees) and, per entry (`entry_summary`), `optimism`: the tool's
claim about itself minus its referee score. Because contestants never see the hidden labels, leaks inside
their pipelines cannot raise their referee score (no shuffled-label rerun is needed); **checks**
(`checks.check_contest`) look for leaks in the data itself: hidden rows duplicated among training rows,
a column that predicts the label almost alone, an id carrying signal on the hidden rows. A flag means
"explain this", not "reject". `contestants/autogluon_contestant.py` runs AutoGluon from its own environment
(`~/.venvs/genml-autogluon`, AutoGluon 1.6; never genML's), drops only the id column, seeds AutoGluon's
data split and bagged models, and reports its own validation score on the metric's scale. `contestants/lightgbm_contestant.py` is a
fixed LightGBM (seconds per run; `--drop`, `--drop-prefix`, `--noise-columns`) for calibrating verdicts.
`benchmark.py` runs noise-only comparisons, planted real gains (column groups withheld) and planted leaks,
and compares the checks with Deepchecks (`external/deepchecks_check.py`, `~/.venvs/genml-deepchecks`).
`seed_study.py` takes one entry run with many seeds (every difference is noise) and reports how often
DeLong's paired AUC test calls single runs, or seed averages, different, next to the referee's grades on
disjoint groups of runs (`verdict.grade_groups`, the grading `compare` uses, for caller-made groups).
`power_study.py` judges 3-run against 3-run comparisons five ways (DeLong on one run and on averages,
a t-test on per-run scores, DeLong plus seed variance, the referee) on same-settings nulls, models that
differ only in their noise columns (`lightgbm_contestant.py --noise-seed`) and planted gains (withheld
columns), and checks each call against the holdout rows (results: DECISIONS.md, 2026-11-18).

### Cross-run ensembles (`src/genML/pipeline/store_ensemble.py`)
Blends the best stored experiments sharing the latest run's folds into a new run folder with its own submission.

### Jobs and MCP
- `src/genML/jobs.py`: SQLite queue (`outputs/jobs.db`), one worker (a lock file makes a second one exit), each job in its own spawned process (GPU concurrency 1, logs in `outputs/jobs/`). A starting worker first settles jobs a dead worker left `running` (`_reap_orphans`: waits for a job process that still runs, then records how it ended); `cancel_job` stops the whole process group and kills it after 30 s; PIDs are checked against `/proc/<pid>/cmdline` before anything is waited on or killed. A job that cannot start is marked failed. A job with `options.code_dir` runs that checkout's code (autopilot worktrees, sandboxed); `options.outputs_root` and `options.source` (trusted callers only) send a run to other outputs and data (the canary). Each job runs in a systemd user scope (`systemd-run --user --scope`, which keeps the job's PID; `jobs.memory_scope`) with `MemoryMax` = the machine's RAM (`MemTotal`; under WSL the VM's) and `MemorySwapMax=0`: a job that outgrows memory is killed and marked `killed: out of memory` instead of swapping (swap growing on a full host disk took WSL down on 2026-11-13). Autopilot sessions run in the same kind of scope (`agent.run_agent`, collected when it ends), because their probes run inside them. `GENML_JOB_MEMORY_CAP` overrides the size or turns it `off` (tests do); without a usable user scope the job runs uncapped and its log says so. The job log ends with the scope's peak memory. Start the worker as a user service so it outlives a session: `systemd-run --user --setenv=PATH="$PATH" --unit genml-worker --working-directory=$PWD .venv/bin/python -m src.genML.jobs worker`.
- `src/genML/service.py`: compact JSON-returning functions used by the MCP server.
- `src/genML/mcp_server.py`: MCP tools over the service (`start_run` returns a job id; `get_run_status` polls; `submit_to_kaggle` previews unless `confirm=true`). It redirects fd 1 to stderr before importing the pipeline so native library output cannot corrupt the stdio protocol.

### Dashboard API (`src/genML/api/server.py`)
`/api/status`, `/ws/progress` (live progress from `outputs/progress/current_run.json`), `/api/problems`, `/api/problems/{problem}/runs[/{run_id}]`, and run-aware `/api/reports`, `/api/reports/{name}`, `/api/models`, `/api/logs/{run_id}` (query `problem` and `run_id`; default the live or newest run). Identifiers are validated against path traversal.

### Legacy
`python src/genML/main.py` with no arguments still runs on `datasets/current/` with everything inferred. The old AI-advisor modules (`tools.py`, `submission_formatter.py`, `ai_tuning.py`, `model_advisor.py`, `domain_researcher.py`) were removed on 2026-11-10; see DECISIONS.md.

## Conventions

- Record every decision that changes behaviour, scope or the rules in `DECISIONS.md` (newest first: decision, reason, what changed), in the same change. Planned work goes under its "Open" heading.
- New problems: `--init-kaggle` or `service.create_problem`, then review the written YAML; add golden coverage automatically (tests parametrize over `problems/`).
- New metrics: add to the registry in `metrics.py` (scorer, output kind, direction, Kaggle names).
- New settings the autopilot may use: validate them in `experiments.check_patch()` and describe them in `autopilot/dossier.SPEC_CAPABILITIES`; otherwise plans can only reach them as code ideas. Examples in `SPEC_CAPABILITIES` must be neutral: never a problem's column names or an idea from its published solutions.
- Autopilot prompts live in `.claude/skills/genml/modes/*.md` with `{{PLACEHOLDERS}}` filled by `loop.Autopilot._values`; each loop mode starts with its permission header (`policy.check_session_policy`), and a step that needs more access gets it there, never in the prompt text; keep the result-file JSON in a mode file and its validator in `loop.py` (`validate_plan`, `validate_explore`, `validate_build`) in step. Never weaken `workspace.PROTECTED` or the judge to let a cycle pass.
- New models: add a factory in `training.build_model_factories()`, an Optuna objective in `tuning.py` (accept `scoring=`; gradient-boosted trees go through `suggest_gbdt_params` / `optimize_gbdt`), add the name to `TUNING_CONFIG['models_to_tune']` and to `experiments.KNOWN_MODELS` so experiments may use it.
- Keep `requirements.txt`, `pyproject.toml` and `environment.yml` in sync.
- Source files use CRLF line endings in the working tree (git autocrlf); match them when scripting edits.
- 4-space indentation, snake_case, Python 3.10+ (Kaggle submit and MCP need 3.11+). Format with black/isort; lint with flake8.

## Dependencies

Core: pandas (<3), numpy, scikit-learn, xgboost, lightgbm, catboost, joblib, pyyaml, optuna, polars and pyarrow (multi-table problems, Parquet caches), psutil (resource monitor). Optional extras in `pyproject.toml`: `dashboard` (fastapi, uvicorn, websockets, watchdog, psutil), `kaggle` (kagglehub, kaggle), `mcp`, `nn` (torch, pytabkit, tabpfn), `llm` (jsonschema, to check second-opinion answers; the `claude` and `codex` CLIs need no package). GPU: `environment.yml` (RAPIDS cuML, CUDA 11.8). Dev: `requirements-dev.txt` (pytest, httpx, black, flake8, isort, mypy). System packages for the autopilot sandbox: `bubblewrap` and `socat` (`sudo apt install bubblewrap socat`).
