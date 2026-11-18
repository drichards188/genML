"""Autopilot: a loop of Claude Code sessions that build and test new ideas, judged by trusted code.

Each cycle (``loop.run_cycle``):

1. **plan** (agent): read the measurements, the journal and the code; probe the data;
   pick one experiment (a settings change, or new code). Every few cycles the plan is
   an **explore** step that must propose an idea not already in the notes or the ledger.
2. **build** (agent, code ideas only): implement it in its own git worktree, with tests.
3. **verify** (trusted): revert edits to protected files and to existing tests, commit,
   run the fast test suite; on failure the build session is resumed to **fix** it.
4. **run** (trusted): queue a normal run with the worktree's code and the idea's settings.
5. **judge** (trusted, ``judge.py``): check the measurement inputs against anchors taken
   before the agents ran, then the paired comparison against the best run.
6. **integrate**: accepted code is fast-forwarded onto ``autopilot/<problem>/main``,
   which the next cycle starts from; every result is appended to the journal and ledger.

The agents run on the Claude subscription (``claude -p`` with tools); the judge, the
folds, the metric and Kaggle submissions are outside their reach (``workspace.PROTECTED``).
"""
