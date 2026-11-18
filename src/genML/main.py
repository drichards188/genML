#!/usr/bin/env python3
"""
Command line entry point for the genML pipeline.

Usage:
    python src/genML/main.py                              # legacy: datasets/current/, everything inferred
    python src/genML/main.py --problem fraud-s6e11        # run a declared problem
    python src/genML/main.py --problem NAME --download    # fetch Kaggle data first if missing
    python src/genML/main.py --problem NAME --notes "added count encoding"
    python src/genML/main.py --list-problems
    python src/genML/main.py --list-runs fraud-s6e11      # compare runs of a problem
    python src/genML/main.py --show-run fraud-s6e11 RUN_ID
    python src/genML/main.py --init-kaggle playground-series-s6e11 --name fraud-s6e11
    python src/genML/main.py --ensemble fraud-s6e11 [--method stack|hill_climb] [--top 10]
    python src/genML/main.py --submit fraud-s6e11 [--run RUN_ID] [--message "..."]
    python src/genML/main.py --submissions fraud-s6e11   # CV vs leaderboard for submitted runs
    python src/genML/main.py --noise-test fraud-s6e11 --run RUN_ID --model XGBoost [--seeds 3]
    python src/genML/main.py --compare fraud-s6e11 EXPERIMENT_A EXPERIMENT_B   # is B really better?
    python src/genML/main.py --prune-runs fraud-s6e11 [--dry-run]   # free disk of runs nothing needs
    python src/genML/main.py --llm-status                 # which Claude/Codex CLIs can run here
    python src/genML/main.py --second-opinion fraud-s6e11 [--cycle C003] [--backend claude-cli]   # a second model reviews a cycle
    python src/genML/main.py --autopilot fraud-s6e11 --cycles 10 [--directive "..."]   # plan/build/test loop
    python src/genML/main.py --autopilot-status fraud-s6e11 | --autopilot-stop fraud-s6e11
    python src/genML/main.py --research fraud-s6e11 | --show-research fraud-s6e11   # cited domain brief

Each run writes to outputs/<problem>/<run_id>/ (data, features, models,
reports, submission.csv, pipeline.log, run.json) and appends a row to
outputs/<problem>/runs.csv.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.genML.flow import create_ml_pipeline_flow, run_pipeline
from src.genML.pipeline.problem import LEGACY_SEARCH_PATHS, list_problems, load_problem
from src.genML.run_context import list_runs, load_run


def check_data_files() -> bool:
    """Legacy mode: confirm train.csv and test.csv can be found."""
    for base_path in LEGACY_SEARCH_PATHS:
        if (base_path / "train.csv").exists() and (base_path / "test.csv").exists():
            print(f"✅ Data files found in {base_path}")
            return True
    print("❌ train.csv and/or test.csv not found.")
    print("   Either place them in datasets/current/, or declare a problem in problems/<name>/problem.yaml")
    print("   and run with --problem <name> (see --init-kaggle for Kaggle competitions).")
    return False


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the genML pipeline")
    parser.add_argument("--problem", help="Problem name (problems/<name>/problem.yaml) or path to a spec")
    parser.add_argument("--download", action="store_true", help="Download Kaggle data if it is missing")
    parser.add_argument("--notes", default="", help="Free text stored with the run")
    parser.add_argument("--list-problems", action="store_true", help="List declared problems and exit")
    parser.add_argument("--list-runs", metavar="PROBLEM", help="List runs of a problem and exit")
    parser.add_argument("--show-run", nargs=2, metavar=("PROBLEM", "RUN_ID"), help="Print a run summary")
    parser.add_argument("--init-kaggle", metavar="COMPETITION",
                        help="Create a problem spec for a Kaggle competition and exit")
    parser.add_argument("--name", help="Problem name for --init-kaggle (default: the competition slug)")
    parser.add_argument("--ensemble", metavar="PROBLEM",
                        help="Blend the best stored experiments across runs into a new run")
    parser.add_argument("--method", default="stack", choices=["stack", "hill_climb"], help="Blend method")
    parser.add_argument("--top", type=int, default=10, help="Experiments to blend (best by OOF score)")
    parser.add_argument("--submit", metavar="PROBLEM", help="Submit a run to Kaggle (newest completed by default)")
    parser.add_argument("--run", metavar="RUN_ID", help="Run to submit with --submit")
    parser.add_argument("--message", default="", help="Kaggle submission description for --submit")
    parser.add_argument("--submissions", metavar="PROBLEM",
                        help="Refresh and print CV vs leaderboard scores of submitted runs")
    parser.add_argument("--baseline", metavar="PROBLEM",
                        help="Queue the baseline run and the noise measurement the autopilot's judge needs "
                             "(required before --autopilot)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs for --baseline (default 1, followed by a seed test; 2+ measure the noise from "
                             "repeated runs instead)")
    parser.add_argument("--exclude-run", nargs=2, metavar=("PROBLEM", "RUN_ID"),
                        help="Mark a run as not counting (problems/<name>/excluded_runs.yaml); needs --reason")
    parser.add_argument("--reason", default="", help="Why the run does not count (--exclude-run)")
    parser.add_argument("--prune-runs", metavar="PROBLEM",
                        help="Delete the tables and feature matrices of runs nothing needs any more "
                             "(keeps the best run, autopilot anchors, runs queued jobs use)")
    parser.add_argument("--dry-run", action="store_true", help="With --prune-runs: only report what would go")
    parser.add_argument("--noise-test", metavar="PROBLEM",
                        help="Refit a run's model (--run, --model) with extra seeds to measure seed noise")
    parser.add_argument("--model", help="Model name for --noise-test (e.g. XGBoost)")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds for --noise-test, including the run's own")
    parser.add_argument("--compare", nargs=3, metavar=("PROBLEM", "EXPERIMENT_A", "EXPERIMENT_B"),
                        help="Paired test: is stored experiment B really better than A?")
    parser.add_argument("--llm-status", action="store_true",
                        help="Show which subscription CLIs (claude, codex) can answer here")
    parser.add_argument("--second-opinion", metavar="PROBLEM",
                        help="Ask a second model (claude-cli by default) to review an autopilot cycle (--cycle, default newest)")
    parser.add_argument("--cycle", help="Autopilot cycle for --second-opinion, e.g. C003")
    parser.add_argument("--backend", action="append",
                        help="Backend(s) for --second-opinion, in order (claude-cli, codex-cli)")
    parser.add_argument("--autopilot", metavar="PROBLEM",
                        help="Run the autopilot loop: Claude Code sessions plan, build and test ideas; trusted code judges")
    parser.add_argument("--cycles", type=int, default=1, help="Cycles to run with --autopilot")
    parser.add_argument("--directive", help="What the next autopilot plan should work on (free text)")
    parser.add_argument("--until", help="ISO time after which --autopilot starts no new cycle")
    parser.add_argument("--plan-only", action="store_true", help="--autopilot: stop each cycle after the plan")
    parser.add_argument("--explore-every", type=int, help="--autopilot: every Nth cycle must find a new idea")
    parser.add_argument("--suspicious-gain", type=float,
                        help="--autopilot: also hold accepted gains above this (metric units); default: only gains "
                             "more than 10x the noise bar are held")
    parser.add_argument("--agent-model", help="--autopilot: Claude model for the sessions (default: the CLI's)")
    parser.add_argument("--no-second-opinion", action="store_true",
                        help="--autopilot: skip the second model's review of plans and suspicious gains")
    parser.add_argument("--allow-unsandboxed", action="store_true",
                        help="--autopilot: run sessions and agent code without bubblewrap (they get your permissions)")
    parser.add_argument("--no-research", action="store_true",
                        help="--autopilot: do not run the domain research step, even if research.yaml asks for it")
    parser.add_argument("--research", metavar="PROBLEM",
                        help="Run the domain research step now (problems/<name>/research.yaml): a blind session reads "
                             "only dated scholarly sources and writes a cited brief for the autopilot")
    parser.add_argument("--show-research", metavar="PROBLEM", help="Show the problem's domain brief")
    parser.add_argument("--autopilot-status", metavar="PROBLEM", help="Show the autopilot's cycles and results")
    parser.add_argument("--autopilot-stop", metavar="PROBLEM", help="Stop the autopilot after its current cycle")
    parser.add_argument("--queue-directive", nargs=2, metavar=("PROBLEM", "TEXT"),
                        help="Queue a directive for a later autopilot cycle (one per cycle, oldest first)")
    return parser.parse_args(argv)


def _print_runs(problem: str) -> None:
    from src.genML.experiments import excluded_runs

    runs = list_runs(problem)
    if not runs:
        print(f"No runs for '{problem}' yet")
        return
    excluded = excluded_runs(problem)
    print(f"{'run_id':22s} {'status':10s} {'metric':9s} {'cv_score':>10s}  {'best_model':20s} notes")
    for run in runs:
        score = run.get("cv_score")
        score_text = f"{score:.5f}" if isinstance(score, (int, float)) else "-"
        print(f"{run['run_id']:22s} {run.get('status', ''):10s} {str(run.get('metric') or ''):9s} "
              f"{score_text:>10s}  {str(run.get('best_model') or ''):20s} "
              + (f"[EXCLUDED: {excluded[run['run_id']]}] " if run["run_id"] in excluded else "")
              + (run.get("notes") or ""))


def main(argv=None):
    """Parse arguments and run the requested command."""
    args = _parse_args(argv)
    os.chdir(project_root)

    if args.list_problems:
        names = list_problems()
        print("Declared problems:" if names else "No problems declared under problems/")
        for name in names:
            spec = load_problem(name)
            print(f"  {name}: {spec.description or ''} [{spec.task or '?'} / {spec.metric or '?'}]")
        return 0

    if args.list_runs:
        _print_runs(args.list_runs)
        return 0

    if args.show_run:
        print(json.dumps(load_run(*args.show_run), indent=2))
        return 0

    if args.ensemble:
        from src.genML.pipeline.store_ensemble import ensemble_from_store

        summary = ensemble_from_store(args.ensemble, method=args.method, top=args.top, notes=args.notes)
        print(f"Ensemble run {summary['run_id']}: {summary['status']}  OOF {summary.get('metric')} = "
              f"{summary.get('cv_score')}  (vs best single: {summary.get('improvement_over_best_single'):+.6f})")
        print(f"   Submission: {summary.get('submission')}")
        return 0 if summary["status"] == "completed" else 1

    if args.submit:
        from src.genML.submissions import submit_run

        row = submit_run(args.submit, run_id=args.run, message=args.message)
        print(f"✅ Submitted run {row['run_id']} (CV {row['metric']} {row['cv_score']}): {row['kaggle_response']}")
        print(f"   Check the score later with: --submissions {args.submit}")
        return 0

    if args.submissions:
        from src.genML.submissions import refresh_scores

        rows = refresh_scores(args.submissions)
        if not rows:
            print(f"No submissions logged for '{args.submissions}'")
        for row in rows:
            print(f"{row['submitted_at']}  {row['run_id']:22s} CV {row['cv_score']:>10s}  "
                  f"public {row.get('public_score') or '-':>9s}  {row.get('status') or ''}  {row['best_model']}")
        return 0

    if args.exclude_run:
        from src.genML.experiments import exclude_run

        try:
            path = exclude_run(*args.exclude_run, reason=args.reason)
        except ValueError as exc:
            print(f"❌ {exc}")
            return 1
        print(f"Run {args.exclude_run[1]} no longer counts for {args.exclude_run[0]} ({path})")
        return 0

    if args.prune_runs:
        from src.genML.pruning import format_report, prune_problem

        print(format_report(prune_problem(args.prune_runs, dry_run=args.dry_run)))
        return 0

    if args.baseline:
        from src.genML.pipeline.diagnostics import queue_baseline_runs

        record = queue_baseline_runs(args.baseline, repeats=args.repeats, notes=args.notes or "")
        print(f"✅ Queued the baseline of {args.baseline}:")
        for run in record["runs"]:
            print(f"   job {run['job_id']}: run {run['run_id']} (seed offset {run['seed']})")
        if record.get("seed_test"):
            print(f"   job {record['seed_test']['job_id']}: seed test on that run (the judge's noise figure)")
        print("   The autopilot can start once these have finished.")
        return 0

    if args.noise_test:
        from src.genML.pipeline.diagnostics import format_noise, seed_noise_test

        if not args.run or not args.model:
            print("❌ --noise-test needs --run RUN_ID and --model NAME")
            return 1
        report = seed_noise_test(args.noise_test, args.run, args.model, n_seeds=args.seeds)
        print(format_noise(report))
        return 0

    if args.compare:
        from src.genML.pipeline.diagnostics import compare_experiments, format_comparison

        print(format_comparison(compare_experiments(*args.compare)))
        return 0

    if args.llm_status:
        from src.genML.llm import backend_status

        for row in backend_status():
            mark = "ready" if row["available"] else "unavailable"
            print(f"  {row['backend']:14s} {row.get('billing', ''):12s} {mark:11s} {row['reason']}")
        return 0

    if args.second_opinion:
        from src.genML.autopilot.loop import review_cycle
        from src.genML.autopilot.second_opinion import format_opinion

        try:
            opinion = review_cycle(args.second_opinion, cycle_id=args.cycle, backends=args.backend)
        except Exception as exc:
            print(f"❌ {exc}")
            return 1
        print(f"Cycle {opinion['cycle']}\n")
        print(format_opinion(opinion))
        print(f"\n   Saved to {opinion['saved_to']}")
        return 0

    if args.autopilot:
        from src.genML.autopilot.loop import Autopilot, clear_stop, format_status, make_config

        clear_stop(args.autopilot)
        config = make_config(args.autopilot, cycles=args.cycles, directive=args.directive, until=args.until,
                             plan_only=args.plan_only, explore_every=args.explore_every,
                             suspicious_gain=args.suspicious_gain, model=args.agent_model,
                             allow_unsandboxed=args.allow_unsandboxed or None,
                             second_opinion=False if args.no_second_opinion else None,
                             research=False if args.no_research else None)
        try:
            state = Autopilot(config).run()
        except (RuntimeError, ValueError) as exc:
            print(f"❌ {exc}")
            return 1
        print(format_status(state))
        return 0 if state.get("status") in ("stopped", "concluded") else 1

    if args.research:
        from src.genML.autopilot import policy
        from src.genML.autopilot.agent import AgentSettings, UsageLimited
        from src.genML.autopilot.research import ResearchError, format_brief, run_research

        if not args.allow_unsandboxed:
            ok, reason = policy.sandbox_available()
            if not ok:
                print(f"❌ No sandbox for the research session: {reason} (or pass --allow-unsandboxed)")
                return 1
        try:
            brief = run_research(args.research, agent=AgentSettings(model=args.agent_model),
                                 backends=args.backend, allow_unsandboxed=args.allow_unsandboxed)
        except (ResearchError, policy.PolicyError, UsageLimited) as exc:
            print(f"❌ {exc}")
            return 1
        print(format_brief(brief))
        return 0

    if args.show_research:
        from src.genML.autopilot.research import format_brief, load_brief

        print(format_brief(load_brief(args.show_research)))
        return 0

    if args.autopilot_status:
        from src.genML.autopilot.loop import format_status, load_state

        print(format_status(load_state(args.autopilot_status)))
        return 0

    if args.queue_directive:
        from src.genML.autopilot.loop import queue_directive

        print(f"Queued {queue_directive(*args.queue_directive)}")
        return 0

    if args.autopilot_stop:
        from src.genML.autopilot.loop import request_stop

        print(f"Stop requested ({request_stop(args.autopilot_stop)}); the loop stops after its current cycle.")
        return 0


    if args.init_kaggle:
        from src.genML.pipeline.kaggle import init_problem_from_kaggle

        path, resolved = init_problem_from_kaggle(args.init_kaggle, name=args.name)
        print(f"✅ Wrote {path}")
        print(f"   target={resolved.target} id={resolved.id_column} task={resolved.task} "
              f"metric={resolved.metric} output={resolved.output}")
        if resolved.inferred:
            print("   Inferred (review these):")
            for key, reason in resolved.inferred.items():
                print(f"     {key}: {reason}")
        return 0

    if not args.problem and not check_data_files():
        return 1

    start_time = time.time()
    try:
        summary = run_pipeline(problem=args.problem, download=args.download, notes=args.notes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
        return 1

    elapsed = time.time() - start_time
    run_dir = project_root / "outputs" / summary["problem"] / summary["run_id"]
    print("\n" + "=" * 60)
    if summary["status"] == "completed":
        score = summary.get("cv_score")
        print(f"🎉 Run {summary['run_id']} completed in {elapsed:.0f}s")
        print(f"   Best model: {summary.get('best_model')}  CV {summary.get('metric')}: "
              f"{score:.5f}" if isinstance(score, (int, float)) else f"   CV score: {score}")
        print(f"   Submission: {summary.get('submission')}")
    else:
        print(f"❌ Run {summary['run_id']} failed: {summary.get('error')}")
    print(f"   Run folder: {run_dir}")
    print(f"   Log: {run_dir / 'pipeline.log'}")
    print("=" * 60)
    return 0 if summary["status"] == "completed" else 1


def kickoff():
    """Entry point for the `kickoff` / `run_pipeline` project scripts."""
    return main()


def plot():
    """Print the stage order."""
    create_ml_pipeline_flow().plot()


if __name__ == "__main__":
    sys.exit(main())
