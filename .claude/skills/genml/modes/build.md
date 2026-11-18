---
# Permissions for this step's session (src/genML/autopilot/policy.py enforces them).
# Edits code and tests in the worktree, runs Python and the test suite, smoke-tests on the data.
tools: [Read, Edit, Write, Glob, Grep, Bash, TodoWrite]
bash: [".venv/bin/python *", ".venv/bin/pytest *", "git status*", "git diff*", "git log*", "git show*",
       "ls*", "head *", "tail *", "wc *", "du *", "file *", "mkdir *", "free*", "nvidia-smi*", "sort *",
       "uniq *"]
read: [outputs, data, notes]
network: none
credentials: hidden
sandbox: required
---
# Build the planned idea ({{CYCLE}}, problem {{PROBLEM}})

The plan step chose this idea:

```json
{{IDEA_JSON}}
```

Implement it in this worktree so that one run of the pipeline can measure it. Current best: {{BEST}}.

1. Read the code you will change and its tests first. Use the pipeline's extension points (CLAUDE.md):
   - feature encodings: `features/encoding.py`, options under `feature_engineering.encodings`; anything
     built from labels goes through the nested per-fold blocks (`nested_target_encodings`, `FoldFeatures`);
   - models: a factory in `training.build_model_factories()`, tuning in `tuning.py`, the name in
     `config.TUNING_CONFIG['models_to_tune']` and `experiments.KNOWN_MODELS`;
   - budget or ensemble settings: the code that reads them, plus `experiments.BUDGET_RULES` / `ENSEMBLE_RULES`;
   - register every new setting in `experiments.check_patch()` and describe it in `autopilot/dossier.SPEC_CAPABILITIES`,
     so later plans can use it without new code.
   There are no downloads: anything the run needs is computed by the pipeline from the competition data.
2. Make it opt-in: a setting switches it on, and without the setting the pipeline behaves exactly as before.
3. Tests are required, in a new file under tests/: unit tests for the new code and, if the feature uses
   labels, a leak test in the style of
   `tests/test_new_capabilities.py::test_digits_and_per_value_encoding_are_target_encoded_and_nested`
   (flip fold k's labels; fold k's block must not change). Then run the fast suite until it passes:
   `.venv/bin/python -m pytest -m "not slow" -q -x`.
4. Smoke-test on the real data before finishing: a script in {{SCRATCH}} that runs the new code on a sample
   (for example 50k rows of {{TRAIN_FILE}} from {{DATA_DIR}}), checking that it runs, how long it takes, its
   peak memory (the full run has {{TRAIN_ROWS}} rows and must stay well under {{MEMORY_LIMIT}}) and that its
   output looks right.
   Do not start a full pipeline run; the orchestrator does that.
5. Write the settings patch that switches the idea on, relative to {{CONTEXT_DIR}}/best_settings.yaml
   (only `models`, `budget`, `feature_engineering`, `ensemble`; mappings merge, lists and values replace).
   Keep the run near this problem's usual {{RUN_MINUTES}}.

Write {{RESULT_FILE}} as JSON:

```json
{
  "summary": "what you built and where",
  "spec_patch_yaml": "feature_engineering:\n  encodings:\n    my_setting: true\n",
  "tests": ["tests/test_my_feature.py::test_builds", "tests/test_my_feature.py::test_no_leak"],
  "leak_test": "tests/test_my_feature.py::test_no_leak",
  "smoke": "rows, seconds, peak memory, a sanity number",
  "notes": "what the verdict should be read with: risks, doubts, anything left out"
}
```

Set `leak_test` to null only when the new code uses no labels, and say so in `notes`.
