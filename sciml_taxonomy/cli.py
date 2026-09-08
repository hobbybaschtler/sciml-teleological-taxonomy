"""Command-line interface.

    python -m sciml_taxonomy replay        replay all case studies of Section 9.4
    python -m sciml_taxonomy replay case5  replay one case
    python -m sciml_taxonomy structural    reproduce Table 5
    python -m sciml_taxonomy validate      validate taxonomy.yaml
    python -m sciml_taxonomy leaves        list the corpus of Table 4
"""

from __future__ import annotations

import argparse
import sys

from .cases import load_cases
from .schema import Taxonomy
from .selection import Outcome, select
from .structural import analyse

RULE = "=" * 78


def _replay(args: argparse.Namespace) -> int:
    taxonomy = Taxonomy.load(args.schema)
    cases = load_cases(args.cases)
    if args.case:
        cases = [c for c in cases if c.id in args.case]
        if not cases:
            print(f"no such case: {', '.join(args.case)}", file=sys.stderr)
            return 2

    failures = 0
    for case in cases:
        result = select(case.task, taxonomy)
        expected_outcome = case.expected.get("outcome", "resolved")
        expected_candidates = list(case.expected.get("candidates", []))
        ok = result.outcome.value == expected_outcome and result.ids == expected_candidates

        print(RULE)
        print(f"[{case.id}] {'OK' if ok else 'MISMATCH'}")
        print(RULE)
        print(result.render())
        if not ok:
            failures += 1
            print(
                f"  expected outcome {expected_outcome!r} with "
                f"{expected_candidates}, got {result.outcome.value!r} with {result.ids}"
            )
        if case.task.notes:
            print(f"Note: {case.task.notes.strip()}")
        print()

    print(f"{len(cases) - failures}/{len(cases)} case(s) reproduced as reported.")
    return 1 if failures else 0


def _structural(args: argparse.Namespace) -> int:
    taxonomy = Taxonomy.load(args.schema)
    print(analyse(taxonomy).render())
    return 0


def _validate(args: argparse.Namespace) -> int:
    taxonomy = Taxonomy.load(args.schema)
    observable, total = taxonomy.observability_ratio()
    print(f"schema_version {taxonomy.schema_version}")
    print(f"decision nodes: {total} (a priori observable: {observable})")
    print(f"leaves: {len(taxonomy.leaves)}")
    print("validation passed")
    return 0


def _leaves(args: argparse.Namespace) -> int:
    taxonomy = Taxonomy.load(args.schema)
    for macro_class in ("M1", "M2", "M3", "M4"):
        print(f"\n{macro_class}")
        for leaf in taxonomy.leaves_of(macro_class):
            discharges = ", ".join(leaf.discharges) or "-"
            print(f"  {leaf.id:<20} recovers {leaf.recovers}")
            print(f"  {'':<20} discharges {discharges}")
            print(f"  {'':<20} requires {leaf.requires}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sciml_taxonomy",
        description="Teleological taxonomy for SciML architecture selection.",
    )
    parser.add_argument("--schema", default=None, help="path to taxonomy.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="replay the case studies of Section 9.4")
    replay.add_argument("case", nargs="*", help="case ids; default is all")
    replay.add_argument("--cases", default=None, help="path to cases.yaml")
    replay.set_defaults(func=_replay)

    structural = sub.add_parser("structural", help="reproduce Table 5")
    structural.set_defaults(func=_structural)

    validate = sub.add_parser("validate", help="validate the schema")
    validate.set_defaults(func=_validate)

    leaves = sub.add_parser("leaves", help="list the architecture corpus")
    leaves.set_defaults(func=_leaves)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
