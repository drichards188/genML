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
    return parser.parse_args(argv)


def _print_runs(problem: str) -> None:
    runs = list_runs(problem)
    if not runs:
        print(f"No runs for '{problem}' yet")
        return
    print(f"{'run_id':22s} {'status':10s} {'metric':9s} {'cv_score':>10s}  {'best_model':20s} notes")
    for run in runs:
        score = run.get("cv_score")
        score_text = f"{score:.5f}" if isinstance(score, (int, float)) else "-"
        print(f"{run['run_id']:22s} {run.get('status', ''):10s} {str(run.get('metric') or ''):9s} "
              f"{score_text:>10s}  {str(run.get('best_model') or ''):20s} {run.get('notes') or ''}")


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
