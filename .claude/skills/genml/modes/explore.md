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
# Explore: find a new idea in the data ({{CYCLE}}, problem {{PROBLEM}})

Current best: {{BEST}}. Cycles finished so far: {{CYCLES_DONE}}. Deadline: {{DEADLINE}}.

This is an explore cycle. Ideas taken from the notes, public notebooks or earlier cycles are not allowed
here: the idea must come from evidence you find yourself, in the data or in the best model's errors. It may
fail. That is fine, as long as it is a real test of something the pipeline does not do yet.

1. Read {{CONTEXT_DIR}}/journal.md, ledger.json, cycles.json, dossier.json, best_settings.yaml and the notes
   in {{NOTES_FILE}}, so you know what has been done. If {{CONTEXT_DIR}}/domain_brief.json exists, its items
   (signals from published domain research) are questions worth probing: whether a signal shows up in the
   best model's errors here. The idea must still rest on what your probes find; list the item ids you
   tested in `idea.brief_items`.
2. Run at least three probes: Python scripts saved in {{SCRATCH}}, on the training data and on the best
   experiment's OOF predictions. The best experiment's predictions are
   {{OUTPUTS_DIR}}/oof/<experiment_id>/oof.npy ({{OOF_KIND}}, rows in the order of {{TRAIN_FILE}}); the encoded
   target is {{OUTPUTS_DIR}}/oof/_targets/y_<fold_signature>.npy; the fold of each row is
   {{OUTPUTS_DIR}}/folds_<fold_signature>.npy. Look for structure the model misses, for example:
   - where the residuals (label minus predicted probability) concentrate: by feature ranges, rare
     categories, pairs of features, and segments a domain expert would recognise (amount bands, merchant
     or product categories, time of day);
   - entity behaviour, if the data has keys such as a customer, card, device or merchant: how often an
     entity appears, and how a row differs from that entity's usual rows;
   - interactions with a domain meaning (ratios such as amount over balance, distance from typical values);
   - how train and test differ (adversarial validation on a sample), and whether predictions are calibrated;
   - where model families disagree on the errors, which points to a model with a different inductive bias.
   Do not probe how the dataset was made (row order, ids, digit or tokenizer quirks of a synthetic
   generator); see the purpose in the rules.
   Record the numbers each probe gave, including probes that found nothing.
3. Screen the candidate ideas these findings suggest with the probe harness (see "Probes" in the rules):
   at least two `compare` calls, on the idea and its variants or rivals. Cite the ones that test the
   chosen idea in `idea.probe_ids`, the closest to what will be built first.
4. Turn the strongest finding into one experiment that a single run can confirm or refute.

Write {{RESULT_FILE}} with the same JSON as a plan step (`decision`, `reason`, `idea`, `journal`; see the
field list below), and inside `idea` also:

```json
"probes": [
  {"script": "{{SCRATCH}}/probe_residual_segments.py", "question": "do residuals concentrate in a segment a domain expert would name?", "finding": "numbers"}
]
```

`idea` fields: `title`, `kind` (settings or code), `hypothesis`, `evidence` (why the finding suggests the
idea), `novelty` (`new` or `adapted`; never `general` here), `predicted_gain` (metric units),
`spec_patch_yaml` (settings ideas; merged into best_settings.yaml, only models, budget,
feature_engineering, ensemble), `build_brief` (code ideas: what to implement and where, the setting that
switches it on, tests including a leak test when labels are used), `est_run_minutes`, `probes` (at least two),
`probe_ids` (ids from {{CONTEXT_DIR}}/probes.jsonl of the harness probes that test the idea; leave it out when
the harness cannot express it), `brief_items` (domain brief item ids the probes tested; optional).
