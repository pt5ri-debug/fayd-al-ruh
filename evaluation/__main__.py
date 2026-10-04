"""Command line entry point for the scientific evaluation suite.

Usage:
    python -m evaluation                 # run and compare against baseline
    python -m evaluation --json          # machine-readable report
    python -m evaluation --update-baseline
    python -m evaluation --no-check
"""

from __future__ import annotations

import argparse
import json
import sys

from evaluation.baseline import (
    BASELINE_PATH,
    check_against_baseline,
    default_report,
    save_baseline,
)
from evaluation.benchmark import report_to_dict


def _format_report(report) -> str:
    lines = ["Scientific evaluation suite", ""]
    for scenario in report.scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        lines.append(f"  [{status}] {scenario.scenario_id} ({scenario.domain})")
    lines.append("")
    lines.append("Metrics")
    for spec_name, value in report_to_dict(report)["metrics"].items():
        lines.append(f"  {spec_name:<30} {value:.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation", description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--no-check", action="store_true", help="skip the regression comparison")
    parser.add_argument("--update-baseline", action="store_true", help="write the baseline from this run")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="allowed per-metric slack before a change counts as a regression",
    )
    args = parser.parse_args(argv)

    report = default_report()

    if args.update_baseline:
        save_baseline(report)
        print(f"Baseline written to {BASELINE_PATH}")

    if args.json:
        print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_format_report(report))

    if args.no_check or args.update_baseline:
        return 0

    try:
        regression = check_against_baseline(report, tolerance=args.tolerance)
    except FileNotFoundError as error:
        print(f"\n{error}", file=sys.stderr)
        return 0

    print()
    print(regression.summary())
    return 0 if regression.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
