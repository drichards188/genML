---
# Permissions for this step's session (src/genML/autopilot/policy.py enforces them).
# Reads the records and writes the final report; no shell.
tools: [Read, Glob, Grep, Write]
read: [outputs, notes]
network: none
credentials: hidden
sandbox: required
---
# Conclude ({{CYCLE}}, problem {{PROBLEM}})

The loop is ending. Write {{RESULT_FILE}} (markdown) for the person who owns this project, from
{{CONTEXT_DIR}}/journal.md, ledger.json, cycles.json and dossier.json:

1. The best result, {{BEST}}, and how it was reached: the accepted cycles with their measured gains.
2. What was tried and did not help, in a few lines.
3. Recommended final submissions: two runs, one safe (best CV with a public score that tracked CV) and one
   with more upside, and why. Use the leaderboard history in dossier.json. Do not submit anything.
4. Code worth keeping: accepted code cycles (branch and commit from cycles.json) and what each adds.
5. What to try next with more time, ranked.

Keep it short and concrete, no em dashes.
