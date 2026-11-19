# genML

**Ended 18 November 2026. Nothing in it measured better than existing tools.**

genML was a tabular ML pipeline for Kaggle-style problems that grew an AI autopilot and, at the end, a
referee for scoring other tools' models.

## Timeline

| When | What |
|---|---|
| Sep 2025 | First version: boosted-tree pipeline, LLM advisors, dashboard |
| | Redesign: declared problem specs, shared folds, stacking, leak-safe encodings |
| | Autopilot: Claude Code sessions build ideas, trusted code judges them on hidden rows |
| | Autopilot on IEEE-CIS fraud: 3 cycles, 1 gain accepted |
| 18 Nov 2026 | Referee built, tested against simpler methods, project ended |

## Why it ended

| Part | Result |
|---|---|
| Pipeline | Duplicates AutoGluon |
| Autopilot | One accepted gain in three cycles, possibly remembered from public work |
| Referee | No better than a plain t-test over reruns (found 26% of planted gains, the t-test 32%) |

## Worth knowing

- Compare tools over several runs. Identical AutoGluon reruns differed by up to 0.03 AUC (AUC measures how
  well a model ranks positives above negatives: 0.5 is chance, 1.0 is perfect).
- AutoGluon's own validation overstated its score on later rows by 0.04 AUC on average.
- LLM agents make provenance hard: one copied public recipe tainted every run built after it.

The full decision log is in `DECISIONS.md`. `SETUP.md` and `CLAUDE.md` describe how to run the code.
