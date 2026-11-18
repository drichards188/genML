---
name: genml
description: Run and steer genML's autopilot for Kaggle tabular problems. Use for "/genml" commands: start, check, steer or stop the autopilot loop (Claude Code sessions that plan, build and test ideas, judged by trusted code), review what it did, or set up a new competition.
argument-hint: "[autopilot|status|stop|review|research|new] <problem or competition> [cycles] [directive]"
---

# genML autopilot

genML's autopilot (`src/genML/autopilot/`) is a loop of headless Claude Code sessions on the
subscription. Each cycle plans one experiment, builds new code when the idea needs it (in its own git
worktree, with tests), runs it through the normal job queue, and lets trusted code judge it (anchored
folds, target and metric; paired test). Accepted code goes onto `autopilot/<problem>/main`. The mode
files in `modes/` are the prompts the loop gives each session (`_shared.md` is added to every one).

Read the first word of the arguments and follow the matching mode. With no arguments, show `status` for
every problem that has an autopilot folder (`outputs/*/autopilot/state.json`).

## autopilot `<problem> [cycles] [directive...]`

Start the loop in the background.

1. Check the baseline has finished: `--baseline <problem>` queues one run of the spec's settings followed by a seed test (minutes: the run's best model refit with two more seeds), and the judge needs both
   (`python -m src.genML.jobs list` shows them). If there is none, offer to queue it and say it costs one
   full run. Check also that no autopilot is running for it (`--autopilot-status <problem>`; a live `pid` means it is running).
   Check the sandbox tools are installed (`which bwrap socat`). If not, ask the person to run
   `sudo apt install bubblewrap socat`; the loop refuses to start without them. Offer
   `--allow-unsandboxed` only if the person asks, and say what it means: agent code gets their permissions.
2. Start it as a user service, so it survives this session and the terminal:
   `systemd-run --user --setenv=PATH="$PATH" --unit genml-autopilot-<problem> --working-directory "$PWD" .venv/bin/python src/genML/main.py --autopilot <problem> --cycles <N, default 10> [--directive "..."] [--until 2026-11-30T12:00]`
   (its output: `journalctl --user -u genml-autopilot-<problem>`; the loop also writes
   `outputs/<problem>/autopilot/autopilot.log`). Only if `systemd-run --user` is unavailable, fall back to
   `nohup ... > outputs/<problem>/autopilot/run.out 2>&1 &` and tell the person it may end with this session.
3. Tell the person the cycle count, the directive, and how to check (`/genml status <problem>`) or stop it.
   Each plan gets a second opinion from a fresh Claude model before it is built or run; `--no-second-opinion`
   turns that off. If the problem has `problems/<problem>/research.yaml` and no brief for it yet, the loop
   first runs the domain research step (up to an hour; `--no-research` skips it).

The first start snapshots the current working tree (uncommitted work included) as the code base and
pins the judge's inputs. It never submits to Kaggle.

## status `<problem>`

Run `.venv/bin/python src/genML/main.py --autopilot-status <problem>`, then read the last entries of
`outputs/<problem>/autopilot/journal.md` and the tail of `outputs/<problem>/autopilot/autopilot.log`.
Summarize: state, current step, best score versus the start, what each recent cycle tried and measured,
anything paused or failing and why.

## stop `<problem>`

`.venv/bin/python src/genML/main.py --autopilot-stop <problem>`: the loop stops after the current cycle.
To stop at once, kill the process; it resumes from the same step on the next start.

## review `<problem> [cycle]`

Explain one cycle in depth (the newest by default) from `outputs/<problem>/autopilot/cycles/<id>/`
(including `second_opinion_*.json` and, if the plan was revised, `plan_revised.json`):
plan.json, build.json, verify logs, judge.json, and the code it committed
(`git diff <start_commit> <commit>` from state.json). Say whether the result looks trustworthy: a gain
above the suspicious limit, or a large CV gain from label-derived features, needs a public leaderboard
check before anything builds on it. If the loop is paused, say what the person must decide, and offer to
submit the candidate for a leaderboard check (ask first; submissions are outward-facing).

## research `<problem>`

Domain research: a blind session reads only scholarly sources published before the problem started and
writes a cited brief that plan and explore sessions read (`src/genML/autopilot/research.py`). It needs
`problems/<problem>/research.yaml` (`domain`: a description of the domain that names nothing about the
problem, its competition, files or columns; `sources_before`: the problem's start date, for a competition
its opening date). Run `.venv/bin/python src/genML/main.py --research <problem>` (tens of minutes, on the
subscription, plus one second-opinion review) or show the current brief with `--show-research <problem>`.

## new `<competition-slug> [name]`

Follow `modes/new.md`.

## Rules for this skill

- Ask before any Kaggle submission, every time.
- Never edit `src/genML/autopilot/`, the judge files or tests to make a cycle pass.
- Merging accepted autopilot code into the person's branch is their call; show
  `git diff <base_commit> autopilot/<problem>/main --stat` and let them decide.
