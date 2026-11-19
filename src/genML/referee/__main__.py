"""Command line for the referee.

    python -m src.genML.referee create --problem ieee-fraud --name ieee-100k --sample 100000
    python -m src.genML.referee enter ieee-100k ag-medium --seeds 0 1 2 --time-limit 600 -- \\
        ~/.venvs/genml-autogluon/bin/python contestants/autogluon_contestant.py --presets medium_quality
    python -m src.genML.referee summary ieee-100k
    python -m src.genML.referee compare ieee-100k ag-medium ag-medium-plus
    python -m src.genML.referee check ieee-100k
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.genML.referee")
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create", help="make a contest from a declared problem")
    create.add_argument("--problem", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--sample", type=int, default=None)
    create.add_argument("--overwrite", action="store_true")
    enter = sub.add_parser("enter", help="run a contestant command, once per seed")
    enter.add_argument("contest")
    enter.add_argument("entry")
    enter.add_argument("--seeds", type=int, nargs="+", default=[0])
    enter.add_argument("--time-limit", type=float, default=None)
    enter.add_argument("--no-sandbox", action="store_true", help="run unsandboxed (trusted commands only)")
    for name in ("summary", "check"):
        sub.add_parser(name).add_argument("contest")
    comp = sub.add_parser("compare", help="graded verdict on a candidate entry against a baseline")
    comp.add_argument("contest")
    comp.add_argument("base")
    comp.add_argument("candidate")
    comp.add_argument("--prior-sd", type=float, default=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    if args.action == "create":
        from src.genML.referee.contest import contest_from_problem

        print(contest_from_problem(args.problem, args.name, sample=args.sample, overwrite=args.overwrite))
    elif args.action == "enter":
        from src.genML.referee.arena import run_entry

        if not command:
            parser.error("give the contestant command after --")
        for seed in args.seeds:
            result = run_entry(args.contest, args.entry, command, seed=seed, time_limit=args.time_limit,
                               sandbox=not args.no_sandbox)
            print(json.dumps({k: result.get(k) for k in ("entry", "seed", "status", "seconds", "errors",
                                                         "self_report")}, default=str))
    elif args.action == "summary":
        from src.genML.referee.verdict import entry_summary

        print(json.dumps(entry_summary(args.contest), indent=2))
    elif args.action == "compare":
        from src.genML.referee.verdict import compare

        print(json.dumps(compare(args.contest, args.base, args.candidate, prior_sd=args.prior_sd), indent=2))
    else:
        from src.genML.referee.checks import check_contest

        print(json.dumps(check_contest(args.contest), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
