---
# Permissions for this step's session (src/genML/autopilot/policy.py enforces them).
# Same as plan: reads code, data and stored predictions, runs Python probes, rewrites the plan file.
tools: [Read, Glob, Grep, Write, Bash, TodoWrite]
bash: [".venv/bin/python *", "git status*", "git diff*", "git log*", "git show*", "ls*", "head *", "tail *",
       "wc *", "du *", "file *", "free*", "nvidia-smi*", "sort *", "uniq *"]
read: [outputs, data, notes]
network: none
credentials: hidden
sandbox: required
---
# Revise the plan after a second opinion ({{CYCLE}}, problem {{PROBLEM}})

Before your plan costs a build or a run, a second reviewer ({{REVIEWER}}, a different AI model) checked
it against the project's rules and the measurements. It saw a summary of the evidence, not the data.

```json
{{OPINION_JSON}}
```

Your plan as it was is in {{CONTEXT_DIR}}/plan_original.json.

1. Judge each concern on the evidence. Where you can, check it with the probe harness (see "Probes" in
   the rules) or a short script in {{SCRATCH}}.
2. If a concern shows the idea breaks a rule (an idea from this problem's published solutions, write-ups,
   discussions or public notebooks, other people's predictions, how the dataset was generated, labels
   reaching a fold's own model), drop the idea: choose a different one, or conclude.
3. Otherwise fix what the concerns point at (evidence, domain reason, cost), or keep the idea and say in
   `reason` why a concern does not apply. Do not keep an idea only because it was yours.

Write {{RESULT_FILE}} again: the full plan with the same JSON fields as the plan step (`decision`,
`reason`, `idea`, `journal`, `probes` inside `idea` if this is an explore cycle, `probe_ids` for the
harness probes that test the idea, and `brief_items` if the idea builds on the domain brief). In `journal`, say
what the second opinion changed and why.
