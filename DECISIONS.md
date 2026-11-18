# Decisions and changes

One place for what was decided, why, and what changed as a result. Newest first. Each entry: the
decision, the reason, and what it changed (files, behaviour). CLAUDE.md describes how the system works
now; this file records how it got there. Git history has the line-level detail.

Add an entry whenever a decision changes behaviour, scope or the rules, in the same change.

## Open (decided to do, not done yet)

1. Strip from the code the capabilities that came from the copied fraud-s6e11 recipe (see 2026-11-12),
   block the routes the copying came through, then run a new fraud-s6e11 baseline (`--baseline`). All
   waiting for the owner's go-ahead. The interactive Claude Code transcripts still hold the copied
   material and agent Python can read them (the sandbox hides only credentials).
2. home-credit and amex-default need `--baseline` before their first autopilot start (ieee-fraud has
   one since 2026-11-16). Titanic
   and the S5 Playground specs do not declare `feature_engineering.base: raw`, so the judge refuses them
   until they do.
3. Refocus genML on being an honest referee for other tools' results (the owner's call, 2026-11-18).
   No AI idea generator has been shown to beat many cheap tests plus an honest scorer, the modelling
   pipeline duplicates AutoGluon, and genML's distinct part is its scoring: the judge on rows a
   contestant never sees, the noise bar, time-forward splits, the shuffled-label check, the sandbox and
   the provenance rules. Next: AutoGluon in its own environment as the first contestant (a command that
   gets the training rows and returns predictions for the hidden rows), then a benchmark of planted
   failures (leaks, noise-only gains, repeated looks at the hidden rows) and planted real gains (a useful
   column withheld from the baseline) on a sample of a real dataset, comparing AutoGluon's own
   validation, Deepchecks and genML's judge. Pass: genML catches clearly more with few false alarms and
   passes real gains above a measured size; otherwise stop. Verdicts become graded (proven, likely but
   unproven, no evidence, leak). New pipeline features and idea-generation work stop; the autopilot
   becomes one contestant among others.

Known limits, accepted for now: modules where a leak could hide (encoding, derived features, training,
ensembles) are not protected from agent edits; the canary catches leaks in accepted code, but settings
ideas are only covered by the settings rules, not a canary run. A canary doubles the run time of an
accepted code cycle. The second opinion is the same model family as the sessions (Claude only since
2026-11-17).

## 2026-11-17

**ieee-fraud C003 (lagged entity fraud history) was accepted, the first cycle with the probe harness.**
An explore cycle. The session found the best run's errors concentrated where card1 is too coarse an
entity, showed with its own scripts that label-free velocity on finer card keys explains none of it, and
then screened 13 variants with the harness (about 50 s each; keys, symmetric vs past-only, label delay,
and a shuffled-label control at +0.0016, inside its noise). The reviewer asked for a realistic label delay
(chargebacks arrive weeks late), so the revision built the 30-day-lag version (probe P09 +0.0073) and not
the stronger no-lag one (P11 +0.0106). New opt-in setting `feature_engineering.encodings.history` in
`features/encoding.py` (`nested_history_encodings`): per row, the smoothed fraud rate and counts of its
card-key entity from training rows at least `lag_days` older, never from its own block or the outer fold;
needs `time_blocks`. Run 20261117_210907_410: stack 0.9231 to 0.9331 AUC on the acceptance rows
(+0.0100 vs bar 0.0081, 5 of 5 folds); held-back rows +0.0092; canary clean. Prediction +0.005, so the
session's shrink rule was too pessimistic here (C002: probe +0.0046, run +0.0047). Caveats: the reviewer
noted that a card-plus-address-plus-email key matches a feature widely shared for this competition; the
record traces it to the cycle's own residual probes and it has a domain reason, so it was not refused,
but the model may have seen that work in training. On the Kaggle test months the history goes stale (no
labels after train ends), which a live system would not have. The code is on `autopilot/main` (commit
177d040), not in the owner's tree; entry E002 in `outputs/ieee-fraud/experiments.json`.

**Plans screen ideas with cheap probes before a run is spent (`autopilot/probe.py`).** A cycle tests one
idea with a 75-minute run against a strict bar, and C001 showed sessions already probe by hand. Those
probes rebuilt features from the raw CSV (a subset of columns, no identity table), scored the judge's own
latest block (the half it decides on and the half it holds back), ran without a memory cap, and were
judged against the wrong bar: the dossier showed the seed test's threshold over all rows (0.0022), while
the judge's bar on its own rows is 0.0070. What changed:
- `probe.py`: `load()` and `compare(label, add=, drop=, weights=, variant_params=)` fit one fixed LightGBM
  (two seeds) on the best run's own feature matrix with its nested encodings, and report the gain, its
  noise (row bootstrap and seed spread) and the gain as a share of the judge's bar. `raw()` gives raw
  training columns but never the target. Every call is logged.
- Rows (`probe_rows`, trusted code): a probe never scores a held-back row. Time splits: train on the
  blocks before the second-to-last and score that block; the judge's block is not used, because choosing
  many ideas on the judge's own rows overfits them (the reusable-holdout problem, Dwork et al. 2015).
  Random folds: score a random half of fold 0's rows that are not held back, so the row codes do not
  reveal the held-back rows (the rules tell sessions not to look for them); the judge decides on 80% of
  rows there, so probes cannot avoid its rows (a known limit).
- Loop: writes `probe_context.json`, `probe_rows.npy` and `probe_history.json` (earlier probes next to
  what their runs measured) into each cycle's context; copies each session's probe log to
  `outputs/<problem>/autopilot/probes.jsonl`; plans cite `idea.probe_ids` (validated against the log);
  the ledger records `probes` and `probe_gain`; the reviewer gets `probe_results` and is asked whether the
  predicted gain exceeds the probe's. Plan sessions get 75 minutes (was 45) and the prompts ask them to
  screen the idea and at least two alternatives.
- The dossier carries `measurements.judge_bar` (the last judged cycle's bar, else a seed-noise lower bound),
  and the prompts quote it; the seed-noise threshold is marked as over all rows.
- Sessions run in a systemd scope like jobs (machine memory, no swap; `jobs.memory_scope`), since their
  probes run inside them.
- The usage-limit pattern matched any "429", including digits inside a session id; it now needs a bare 429.
Calibration on ieee-fraud (block 4, 98k rows, about 1.5 minutes a probe uncontended, 5.6 GB): a pure-noise
feature +0.0006 (noise 0.0008); C002's recency weights +0.0046 (noise 0.0013) against the full run's
+0.0047 on the judge's rows; C001's card-velocity features +0.0025 (noise 0.0010), which C001's own probe
had found flat and no run has measured. One idea with a full-run result is not a calibration, so the
ledger now records probe and measured gains side by side. Probe numbers are advisory (a session could
write the log itself); the judge still decides. Not done yet: the loop re-running a cited probe in the
job sandbox to confirm it, and multi-seed averaging to lower the bar.

**ieee-fraud C002 (recency-weighted training) was measured and rejected below the bar.** Sample weights
`exp(-age/tau)` with tau = 60 days, counted back from each fit's newest training row, raised the stack
from 0.9231 to 0.9278 AUC on the acceptance rows (+0.0047, 5 of 5 folds), under the bar of 0.0070,
which is mostly seed noise (sigma_seed 0.0015 on these rows; row noise 0.0011). The judge is unchanged. The code stays on branch
`autopilot/ieee-fraud/C002` (commit bd03293); entry E001 in `outputs/ieee-fraud/experiments.json`.

**The loop makes the research gate's copy before each session.** C002's plan and build sessions saw
three gate tests fail inside their sandbox (`OSError: Read-only file system` on
`~/.cache/genml-autopilot/gate/<hash>`). A worktree's `research_gate.py` has CRLF line endings
(git autocrlf), so its content hash differs from the owner's LF file and its copy did not exist yet;
the sandbox lets a session write only inside the worktree. Verify runs outside the sandbox and passed,
but the plan briefed the build to deselect a protected test as "environmental", which is the habit to
avoid. `Autopilot._agent` now calls `_prepare_gate_copy(worktree)` (`research.gate_copy(script)`, which
now takes a checkout's script) before every session; a failure is logged and the session still runs.
Giving sessions write access to the gate folder was rejected: the copy is what the research hook runs.

**A cycle never commits the placeholder files a killed sandbox leaves behind.** WSL was shut down
during C002's build (after the owner's `fstrim`; not a crash). Claude Code's sandbox guards shell, git,
editor and MCP config files (`.bashrc`, `.gitconfig`, `.gitmodules`, `.mcp.json`, `.vscode` and others)
by mounting an empty read-only file over each, creating it when missing and removing it when the session
ends; the killed session left nine of them in the worktree, and `commit_cycle`'s `git add -A` would have
put them in the cycle's commit and, if accepted, on `autopilot/main` (an empty `.mcp.json` or
`.gitmodules` in the owner's tree breaks tools that read them). `commit_cycle` now first deletes empty
files with those names at the worktree root that did not exist at the cycle's start
(`workspace.drop_sandbox_placeholders`, reported as `sandbox_leftovers`); files with content are left to
the usual checks. The owner's snapshot is unchanged. C002 resumes at its build step.

**The second opinion comes from Claude; Codex is out of the default order (Codex is no longer
signed in).** C001's plan review had already fallen back to Claude because Codex refused the model in
`~/.codex/config.toml` ("not supported when using Codex with a ChatGPT account"). `second_opinion.ORDER`
(plan, gain and cycle reviews, and the research review) and `config.LLM_CONFIG['order']` are now
`claude-cli` only, so no call waits on Codex first and `--llm-status` shows only Claude. The Codex
backend stays: putting `codex-cli` first in `ORDER` (or `GENML_LLM_ORDER`) brings it back. What is lost:
the reviewer is now the same model family as the sessions it checks, so blind spots they share go
unchecked. It still gets only the material it is handed, with no tools and none of the session's
context, under the same rules, and the judge, the canary and the settings rules still decide what counts.
Tests: `test_claude_answers_by_default_without_trying_codex`, `test_codex_is_not_in_the_default_order`;
the Codex-to-Claude fallback is still tested with Codex named explicitly.

## 2026-11-16

**Jobs may use all of the machine's RAM but never swap (the owner's call: the cap sized to WSL's
memory).** On 2026-11-13 the ieee-fraud baseline pushed WSL into swap while the Windows disk that holds
the swap file was nearly full; swap writes failed and WSL went down with everything in it. Each job now
runs in a systemd user scope (`systemd-run --user --scope`, which keeps the job's PID, so cancelling and
orphan checks are unchanged) with `MemoryMax` = `MemTotal` and `MemorySwapMax=0`. A job that outgrows
memory is killed and marked `killed: out of memory`; WSL and the other processes stay up. The cap is
applied per job, not to the worker's service, because the autopilot and the MCP server start workers
on their own (`ensure_worker`). `GENML_JOB_MEMORY_CAP` overrides the size or turns it off (the tests do);
without a usable user scope a job runs uncapped and its log says so. Each job log now ends with the
scope's peak memory, because no run had ever recorded how much memory it needed. Checked live: a capped
scope with no swap kills a process over its limit and systemd reports `oom-kill`. Tests:
`tests/test_run_storage.py` (one test starts a real scope when the machine has a user systemd).

**Stage tables as Parquet, and runs nothing needs are pruned.** An ieee-fraud run turned 125 MB of
input into 4.8 GB: 1.75 GB of uncompressed pickles handing the tables between stages, 2.1 GB of feature
matrices and 1 GB of per-fold encoding blocks, kept forever, one set per autopilot cycle. The tables now
go between stages as zstd Parquet (`dataset.save_stage_tables` / `load_stage_table`; both as pickle when
Parquet refuses either frame, so train and test always read back alike; older runs' pickles still load).
Parquet returns missing text as None; the loader turns it back into NaN, as a CSV read gives, because
code such as the original-dataset means compares `astype(str)` values across sources. For multi-table
problems (whose tables came from Polars with None) missing text is now NaN too: within a run nothing
changes, but category codes can differ from runs made before today. After every job the worker prunes
the job's problem (`pruning.py`, also `--prune-runs PROBLEM [--dry-run]`): runs that are not the current
best, not named in the autopilot state's anchors or best, not used by an active job and not unfinished
and recently changed lose their tables (the test ids stay, for cross-run ensembles) and big matrices,
and get a `pruned.json`. A canary's shuffled data copy and its run's matrices go once its verdict is
recorded. Expected effect for ieee-fraud: about 14 GB at the peak of a cycle instead of 5 to 6 GB more
per cycle. Not done (discussed): a feature cache shared by runs with the same data, settings and code.

**ieee-fraud baseline, the first run under the cap.** Run `20261116_212212_486` (77 minutes) and its
seed test: OOF AUC 0.9303 (stack of XGBoost 0.9290, LightGBM 0.9286, CatBoost 0.9162), forward holdout
0.9254 on the latest time block, seed sigma 0.00048 on all rows. Peak memory 18.2 GiB for the run and
7.8 GiB for the seed test, of the 23.5 GiB cap, never swapping; the scope's peak counts file cache too,
which the kernel drops before it kills (mid-tuning: 11.9 GiB program memory, 2.7 GiB cache). The run
folder is 3.2 GB instead of 4.8 GB. The autopilot's anchors: 0.9231 on the acceptance rows, run noise
0.0015 there. Then one autopilot cycle (`--cycles 1`).

**The research gate's hook command no longer depends on where the checkout lives.** The first ieee-fraud
cycle (C001, recency-weighted training) failed verify three times on a protected test the agent could
neither have broken nor fixed: `test_the_gate_hook_blocks_and_fails_closed` checks that the hook command,
which the blind research session sees, holds none of the domain's words, and the command held the
checkout's path, which in an autopilot worktree is `worktrees/<problem>/<cycle>` ("ieee-fraud" contains
"fraud"). Every cycle of a problem whose name shares a word with its domain would have failed the same
way. Real research sessions run from the owner's checkout and were not affected. `gate_command` now runs
a read-only copy of the gate in `~/.cache/genml-autopilot/gate/<content hash>/` (rewritten when its bytes
differ) with the interpreter's real path (the gate needs only the standard library), so the command is
the same from any checkout. The verify report now starts with pytest's FAILED/ERROR lines and totals
(pytest prints its warnings after the tracebacks, so the failure in the log showed only warnings), and a
failed cycle's message takes the start of the report instead of its end. Checked: the fast suite passes
from a copy of the checkout under `worktrees/ieee-fraud/C001`. C001's code stays on its branch
`autopilot/ieee-fraud/C001`, never run. The plan's review came from Claude: Codex refused the model set
in the owner's `~/.codex/config.toml` (`gpt-6-astra`, "not supported when using Codex with a ChatGPT
account"; it answered on 2026-11-12).

## 2026-11-13

**Hints to the 2019 IEEE-CIS winners' solution removed from files the agents read**, before the first
ieee-fraud autopilot start. A docstring in `features/derived.py`, a test example in
`tests/test_real_data.py` and three lines of this file described the excluded idea; they are neutral now
(the owner's 2026-11-12 rule: agents must not see the content of excluded ideas). Still readable by
agent Python and not fixed here: the interactive Claude Code transcripts (Open 1).

## 2026-11-12

**A research step gives the autopilot the web for domain knowledge, never for answers (the owner's
call).** The owner wants agents to learn what carries signal in a domain (for example in card fraud)
instead of relying only on what the model remembers, without letting them look at a problem's answers.
The research step (`autopilot/research.py`, `research_gate.py`, `modes/research.md`) is the only
session with web access, and three barriers keep answers out of reach. Blind: the session gets only the
owner's domain description from `problems/<name>/research.yaml` (refused if it names the problem, its
competition, files, columns or target), runs in an empty folder with a random name and has no shell.
Reference sources only: WebFetch to scholarly APIs (OpenAlex, Semantic Scholar, arXiv), arXiv pages,
old Wikipedia revisions and owner-listed hosts; no web search; Kaggle, GitHub, blogs and forums denied
and refused as owner hosts. Older than the problem: only sources published before `sources_before` (for
fraud-s6e11, 2026-11-01, the competition's start from the Kaggle API), so none can hold its answers.

A live test showed why permission rules alone are not enough: with only `api.openalex.org` allowed, a
headless `--restricted` session still fetched `docs.python.org` (Claude Code pre-approves some
documentation sites), while unlisted hosts were refused. So a PreToolUse hook, the gate, checks every
fetch: searches must carry the date limit, single records are looked up and must be older, anything else
or any error blocks. A second live test confirmed the hook runs in this mode, blocks the pre-approved
site and undated searches, allows dated ones and can make its own lookups. The gate takes only the
cutoff and owner hosts as arguments, because the session sees the hook's command when it blocks. After
the session, trusted code audits its log (every successful fetch must pass the gate again, or the whole
brief is refused), re-checks every cited source's date, and the second opinion (Codex first) drops
dataset artifacts and items that name a competition. Plan, explore and revise sessions read the result
as `domain_brief.json` and cite items in `idea.brief_items` (validated, recorded in the ledger, shown to
the plan's reviewer); `_shared.md` now counts the brief as general domain knowledge, not as evidence.
Commands: `--research`, `--show-research`, `--no-research`. Tests: `tests/test_research.py` (gate rules,
citations, blindness, audit, a full run with a fake CLI, the loop) and new cases in the protected
`test_autopilot_policy.py` (research is not a loop mode, its header and settings, the hook fails
closed, loop headers still refuse the web). Also fixed: the explore prompt's example probe no longer asks
about the cents digit.

Limits, accepted: owner-listed hosts are not date-checked; papers about a public dataset that a
synthetic set was generated from pass the date check (other people's domain research, which the rules
allow); the models still remember older competitions' public work, which no gate can remove.

**Tainted data deleted (the owner's call: delete, not quarantine).** On 2026-11-06 an interactive session
read this competition's discussion and a public notebook and copied the notebook's step-by-step recipe
into the fraud notes. Every fraud-s6e11 run after 2026-11-03 descends from it (E001, the anchor, included),
so their scores measure looking, not genML. Deleted:

- fraud-s6e11: runs 20261106_113915_138, 20261106_133032_053, 20261107_081643_935, 20261107_081645_566,
  20261110_105950_455, 20261110_123514_203, 20261110_151421_987 and 20261110_155142_905 with their stored
  OOF predictions and progress files; the LLM logs (`llm/`); `experiments.json` (E001 to E005);
  `noise_runs.json`; the autopilot state and its 2026-11-07 archive; `submission_latest.csv`; their rows
  in `runs.csv`; their jobs (1 to 7, 11, 13, 14, 17, 18) and job logs.
- ieee-fraud: run 20261109_132836_471 (built on the 2019 winners' published write-up), its OOF
  predictions, `run.out` and `submission_latest.csv`.
- `.playwright-mcp/` (saved Kaggle pages: this competition's discussion, the public notebook and other
  Playground write-ups), the old branches `autopilot/fraud-s6e11/{main,C001,C002}`, and the autopilot
  sessions' own transcripts.

Kept: `submissions.csv` (those submissions stay on Kaggle; the deleted runs' descriptions now say only
that they were deleted), the 2026-11-03 fraud run (made before the copying, but with the old
target-encoding leak, so it cannot be an anchor) and the folds. `excluded_runs.yaml` and the notes give
one neutral reason and name no ideas, so nothing an agent reads says what the copied runs did. No fraud run
is now both clean and leak-free; a new baseline is needed.

**Autopilot sessions can run their Python probes again.** In fraud-s6e11's C001 and C002 every probe was
refused, so both plans were untested guesses. The only allowed interpreter is the worktree-relative
`.venv/bin/python`, and Claude Code keeps the folder a `cd` moved into for the rest of the session: after
`cd .autopilot` the relative path no longer existed, and the absolute path, `../.venv/bin/python` and
`python3` matched no allow rule. Now every shell command starts in the worktree
(`CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1` in `agent.agent_env`), the same `.venv` commands are also
allowed by their absolute worktree path (`policy.bash_allow_rules`), and `_shared.md` tells sessions that a
`cd` does not carry over and that other interpreters are refused. Nothing new may run: the extra rule names
the same program, and the deny rules and the sandbox are unchanged. Checked live with a real session under
the plan step's permissions: the relative path after a `cd`, the absolute path and a `timeout` wrapper all
ran the probe; `python3` was refused; inside the probe the Kaggle token was hidden, the network was blocked
and the outputs folder was not writable. Tests: two new cases in `test_autopilot_policy.py`.

The owner's rule, clarified the same day: an idea counts when genML found it through its own work, even if
others published the same idea; it does not count when we got it by looking. (The explore prompt's
example probe about the cents digit was replaced and the excluded runs' content that agents could see
was deleted the same day, above.)

## 2026-11-10

**The first baselines ran on the CPU; they were stopped and rerun on the GPU.** The job worker was
started with `systemd-run --user`, whose PATH lacks `/usr/lib/wsl/lib`, so `nvidia-smi` was not found and
GPU detection quietly sent every model to the CPU (3 to 4 times slower; CatBoost also changes its metric
and sampling on the CPU, so such a baseline cannot be compared with GPU runs). `gpu_utils.nvidia_smi()`
now falls back to the WSL path, and every documented `systemd-run` command passes `--setenv=PATH`.
Still open: record each run's devices and have the judge refuse a comparison across devices.

**A baseline is one end-to-end run, not three** (the owner's call, to save GPU time). `--baseline` now
queues one run and then a seed test on it (its best single model refit with two more seeds, minutes, as
a `noise` job); the judge rescores those seeds on its own acceptance rows for its noise figure. This
misses the variation from re-tuning that repeated full runs would show, so the bar is somewhat lower;
`--repeats 3` still measures it that way when wanted. The two extra runs per problem queued earlier
(jobs 6, 7, 9 and 10) were cancelled.

**The review's remaining items are done** (the agreed order, items 2 to 6):
- *Tainted runs*: `problems/<name>/excluded_runs.yaml` (`--exclude-run`) lists runs that do not count;
  nothing picks them as a best run, baseline, blend member or default submission. Excluded: ieee-fraud
  20261109_132836_471 (an idea from the 2019 winners' write-up) and fraud-s6e11 E002 and E003 (a public
  discussion's ideas). fraud-s6e11's best clean run is now E001 (CV 0.94761). Its notes were cut to facts
  and own measurements (claims about the labels, public notebooks' folds and the original dataset are
  gone), a spec comment that repeated a discussion claim was removed, and its old autopilot state moved
  to `outputs/fraud-s6e11/autopilot_archived_20261110/`.
- *Leak paths*: derived features, keys and group statistics may not read the target (also through a
  chain); experiments may not set `encodings.nested: false` or `base: auto`; the loop now runs
  `check_patch` on every patch and `leak_problems` on the merged settings; anchoring refuses a leaky
  baseline. Accepted code is rerun on shuffled labels (the canary) and rejected if it still beats a
  constant prediction.
- *Honest noise bar*: `--baseline` queues seeded repeats of the same settings (`budget.seed`, which only
  trusted code can set); their spread is the judge's noise, required before anchoring. The judge decides
  on 80% of the rows and keeps 20% that never decide (shown by `--autopilot-status`); time-split problems
  are judged forward on the latest block only; "suspicious" is 10 times the noise bar instead of a fixed
  0.004.
- *Operations*: one shared branch `autopilot/main` for every problem, refreshed from the owner's
  working tree at each start (merged when it holds accepted code; a conflict refuses to start). The job
  worker locks the queue, settles jobs a dead worker left running, force-kills hung jobs and checks PIDs
  before acting on them; it runs as the user service `genml-worker`.
- *Prompts*: row counts, file names, fold counts, run time and machine are filled in per problem.
Tests: `test_honest_judging.py` and `test_autopilot_ops.py` (both protected); the fast suite passes.
CrewAI was uninstalled from `.venv` (nothing imports it).

**Both sandboxes checked live after installing bubblewrap and socat.** A real Claude Code session with
the plan step's permissions could not read the Kaggle token, the Claude login or `~/.ssh` (shown empty),
could not write the outputs or the main checkout, and had no network; it could write its own folder and
read the data. A pipeline run of worktree code under bubblewrap could write its problem's outputs but
not the autopilot state, had no network or credentials, and trained XGBoost on the GPU; a real Titanic
job went through the sandboxed job worker end to end. Two fixes came out of it: sessions keep Claude
Code's own temp folder (its sandbox sockets fail on long paths such as a worktree's), and runs get a
private temp folder through `TMPDIR` instead of a replaced `/tmp` (which hid anything under `/tmp`).

**Removed dead code; nothing the pipeline or the autopilot uses was lost.**
Why: leftovers of the old AI-advisor era made the code larger and harder for agents to read.
Removed `tools.py` (legacy re-exports), `submission_formatter.py` (its one live helper moved to
`problem.sample_value_kind`), `pipeline/ai_tuning.py` (overrides file never written),
`pipeline/model_advisor.py` (the model-selection stage only recorded a skip; `detect_problem_type`
moved to `training.py`), `features/domain_researcher.py` (a "detected domains" label no decision used;
its web research was never called), about 300 lines of AI feature hooks in `feature_engineering.py`,
the AI insight fields of the progress tracker, and unused settings in `config.py` (`AI_ADVISORS_CONFIG`,
`AI_SUPPORTED_FEATURE_OPERATIONS`, `AI_TUNING_*`, `SUPPORTED_MODEL_SUMMARIES`). The flow now has four
stages (load, features, training, prediction); the dashboard shows "of 4". Tests for the removed code
went with it. Backup of the tree before the removal: commit object `a2c2f37` (no branch points to it).

**Fixed tuned linear and logistic regression.** They were tuned as `ridge_alpha`, which no estimator
accepts, so every tuned run of them failed and was dropped silently; tuning also standardized the
columns while the fitted models did not. Now tuning searches `alpha` (Ridge) and `C` (logistic), and the
factories standardize inside each fit. Test: `tests/test_linear_models.py`.

**The second planner became a second opinion.** The old run advisor (`--advise`, `--apply-review`,
`--experiments`, `--scorecard`, `advisor.py`) duplicated what the autopilot does with less: settings
only, run by hand, with its own acceptance rule. What it had that the autopilot lacked was a review by a
different model family. Now:
- `autopilot/second_opinion.py` asks Codex (Claude when Codex cannot answer) for `ok | concerns | reject`.
- The loop asks it about every plan before a build or run (new step `review`; on concerns the plan gets
  one revision, mode `revise`; a revised plan rejected again fails the cycle; no reviewer means go on).
- It also triages every suspicious gain for the person who must check it.
- `--second-opinion NAME [--cycle ID]` asks about any cycle; `--no-second-opinion` turns the review off.

Moved `build_dossier` and `SPEC_CAPABILITIES` to `autopilot/dossier.py` (protected, read-only now; it
used to rewrite `experiments.json`). Removed the paid-API backends and their spend cap from `llm/`
(everything runs on the Claude and ChatGPT subscriptions) and the experiment launcher from
`experiments.py`; the ledger and settings checks stay. Deleted `advisor_playbook.md` instead of giving
it to the planners: it drew on competition solutions and repeated ideas that `problems/fraud-s6e11/
notes.md` says came from that problem's public discussion. Two `SPEC_CAPABILITIES` examples came from
the 2019 IEEE winners' write-up and named an ieee-fraud column; they are neutral now. Checked with one real
Codex call: a plan to pool public predictions came back `reject`, citing the rule.

**Docs brought up to date**: README, SETUP and CLAUDE.md describe the project as it is now (autopilot as
the main path, four stages, sandbox, second opinion). `psutil` added to the core dependencies (it was
imported on every run but declared only for the dashboard); `environment.yml` gets the `pandas<3` pin.
The `/genml` skill starts the loop with `systemd-run --user`, so it no longer ends with the session.

**Autopilot permissions are declared and enforced, not prompted.**
Why: the rules (no public notebooks or predictions, no Kaggle submissions) were prompt text only; the
sessions had web tools and Kaggle commands, and agent code ran with the owner's permissions, able to
submit or rewrite the judge's pinned hashes. Changed: each loop mode file has a permission header
(`policy.py`), turned into `claude -p --restricted --tools ... --settings` with Claude Code's sandbox
(no network, credential files unreadable, writes only in the worktree, refuses to start without the
sandbox). Pipeline runs of agent code run in bubblewrap per `autopilot/run_policy.yaml` (no network,
credentials hidden, the autopilot state read-only). The `external/` download folder is gone; settings
ideas run on a reset worktree; a job that cannot start fails instead of killing the worker. Needs the
system packages `bubblewrap` and `socat`; `--allow-unsandboxed` is the owner's override. Rules pinned
by `tests/test_autopilot_policy.py`.

**Multi-table problems are covered by the golden inference test.** It reads each spec's main table;
the slow smoke test skips multi-table problems (its subsample writes only train.csv/test.csv).

## 2026-11-09

**No ideas from a problem's published solutions, write-ups, discussions or public notebooks.** Scores
must measure genML, not hindsight. The ieee-fraud baseline had used an idea from the 2019 winners'
write-up; that run is excluded from the honest record and the spec was cleaned. fraud-s6e11's E002 and E003
used ideas from a public discussion, so its current best is not a clean measure either. Recorded in
CLAUDE.md (purpose), the autopilot prompts, the advisor prompt and each problem's notes.

**Added multi-table problems, group and time folds, derived features and neural models.** home-credit,
ieee-fraud and amex-default specs; `pipeline/tables.py`, `features/derived.py`, `pipeline/neural.py`;
`cv.method: group_kfold | stratified_group_kfold | time_blocks`.

## 2026-11-07

**genML is a process for real problems; Kaggle is only a test bench.** Leaderboard rank is not the
goal. Out of scope: pooling other people's public OOF or test predictions, and modelling or exploiting
how a synthetic dataset was generated. The autopilot's first cycle (C001, pooling public predictions)
was cancelled for this reason.

**Autopilot loop added** (plan, build in a worktree, verify, run, trusted judge with pinned anchors).

## 2026-11-03

**Redesign around declared problems and out-of-fold predictions.** Problem specs with inference of the
rest, a metric registry that decides the submission's content (fixing AUC entries that were sent as
hard labels), one run folder per run, shared folds, an OOF store with stacking, leak-safe nested target
encodings, Kaggle submit, a job queue, the MCP server. CrewAI was replaced with plain Python.
Commits 0f0620f to f57c2e9.
