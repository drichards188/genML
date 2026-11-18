---
# Permissions for this step's session (src/genML/autopilot/policy.py enforces them).
# Reads code, data and stored predictions, runs Python probes, writes the plan file.
tools: [Read, Glob, Grep, Write, Bash, TodoWrite]
bash: [".venv/bin/python *", "git status*", "git diff*", "git log*", "git show*", "ls*", "head *", "tail *",
       "wc *", "du *", "file *", "free*", "nvidia-smi*", "sort *", "uniq *"]
read: [outputs, data, notes]
network: none
credentials: hidden
sandbox: required
---
# Plan the next experiment ({{CYCLE}}, problem {{PROBLEM}})

Current best: {{BEST}}. Cycles finished so far: {{CYCLES_DONE}}. Deadline: {{DEADLINE}}.

Directive from the person who owns this project (if it is not "none", the idea must serve it):
{{DIRECTIVE}}

Your job: choose the single experiment most likely to raise the score for the hours it costs, and
describe it well enough that it can be built and run. You do not build it in this step.

1. Read {{CONTEXT_DIR}}/journal.md, ledger.json and cycles.json: what was tried, what was measured, what
   failed and why. Do not repeat a measured idea unless you change something that matters, and say what.
2. Read dossier.json (the best run's models, scores, paired comparisons, seed noise, tuned parameters at
   the edge of their range, prediction agreement between models, leaderboard history), best_settings.yaml
   (the settings the next run builds on) and the notes in {{NOTES_FILE}}. If
   {{CONTEXT_DIR}}/domain_brief.json exists, read it too: signals that published domain research found,
   with sources. An idea may build on its items; list their ids in `idea.brief_items`, map each signal to
   the columns that carry it here, and still give evidence from this problem (the dossier or a probe).
3. Read the code the idea would touch (CLAUDE.md maps it), so the build brief names files and functions.
4. Screen, about 30 to 40 minutes: test the idea and at least two alternatives or variants with the probe
   harness (see "Probes" in the rules) before choosing, and cite the probes that test the chosen idea in
   `idea.probe_ids`. Ideas it cannot express (another model family, tuning) get a small script of their own
   in {{SCRATCH}}, or no probe; say which in `evidence`.
5. Choose. A settings idea uses only settings the pipeline already has (dossier.json `spec_capabilities`,
   best_settings.yaml). A code idea needs new code, which the next step builds in this worktree.

The bar: a change counts only if its paired gain beats the judge's bar, {{JUDGE_BAR}} (3 x the combined
row and run-to-run noise on the judge's rows; dossier.json `measurements.judge_bar`; the `seed_noise`
threshold there is over all rows and lower) and it wins at least {{FOLDS_NEEDED}} of {{N_FOLDS}} folds,
judged on {{JUDGED_ON}}. Prefer ideas whose expected gain clearly clears it. A run of this problem
takes {{RUN_MINUTES}}; a build takes up to 2 hours.

Choose "conclude" only if the deadline has passed or nothing plausible is left; say why.

Write {{RESULT_FILE}} as JSON with these fields:

```json
{
  "decision": "continue",
  "reason": "one or two sentences: why this idea now (or why stop)",
  "idea": {
    "title": "short name",
    "kind": "settings or code",
    "hypothesis": "what should improve and why",
    "evidence": "what supports it: numbers from the dossier, the ledger, the notes or your probes",
    "novelty": "general (standard practice in ML or in this domain), adapted (a general idea changed for this problem) or new",
    "predicted_gain": 0.0004,
    "spec_patch_yaml": "budget:\n  n_trials: 30\n",
    "build_brief": "code ideas: what to implement and where, the setting that switches it on, how to test it (with the leak test when labels are used), memory and time limits",
    "est_run_minutes": 30,
    "brief_items": ["B03"],
    "probe_ids": ["P04", "P02"]
  },
  "journal": "a few lines for the journal: what this step learned that later cycles should know"
}
```

- `predicted_gain` is the expected paired gain in the metric's units (it may be negative). When a probe
  tested the idea, start from its gain and shrink it by what probe_history.json shows.
- `probe_ids` lists the probes (ids in {{CONTEXT_DIR}}/probes.jsonl) that test the idea, the one closest to
  what will be built first; leave it out when no probe could test it.
- `spec_patch_yaml` is merged into best_settings.yaml: only `models`, `budget`, `feature_engineering` and
  `ensemble`; mappings merge, lists and values replace. Settings ideas must give it; for code ideas it is an
  optional draft (the build step writes the final one).
- `brief_items` lists the domain brief items the idea builds on (leave it out or empty when none; only
  ids that are in domain_brief.json).
- Leave `idea` out when concluding.
