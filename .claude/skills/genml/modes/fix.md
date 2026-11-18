---
# Permissions for this step's session (src/genML/autopilot/policy.py enforces them).
# Same as build: edits code and tests in the worktree, runs Python and the test suite.
tools: [Read, Edit, Write, Glob, Grep, Bash, TodoWrite]
bash: [".venv/bin/python *", ".venv/bin/pytest *", "git status*", "git diff*", "git log*", "git show*",
       "ls*", "head *", "tail *", "wc *", "du *", "file *", "mkdir *", "free*", "nvidia-smi*", "sort *",
       "uniq *"]
read: [outputs, data, notes]
network: none
credentials: hidden
sandbox: required
---
# Fix the build (attempt {{ATTEMPT}}, {{CYCLE}}, problem {{PROBLEM}})

Your build did not pass verification. This is what the orchestrator found:

```
{{PROBLEMS}}
```

Find and fix the cause in this worktree. The rules still hold: do not edit protected paths or existing
tests (they are reverted), keep the new behavior opt-in, and add tests rather than weakening any. If the
full run failed on the real data, reproduce the failure on a sample in {{SCRATCH}} first, then fix it and
check memory and time again.

Run the fast suite until it passes (`.venv/bin/python -m pytest -m "not slow" -q -x`), then write
{{RESULT_FILE}} again with the same fields as before (`summary`, `spec_patch_yaml`, `tests`, `leak_test`,
`smoke`, `notes`), and describe the fix in `notes`.
