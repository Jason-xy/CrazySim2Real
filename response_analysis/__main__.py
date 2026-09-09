"""Run with python -m response_analysis. All commands are offline."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from . import AnalysisConfig, ValidationError, analyze_run, compare_groups, validate_run
from .protocol import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline closed-loop response and Sim2Real gap analysis")
    parser.add_argument("--version", action="version", version="response_analysis 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate", "Check CSV columns, samples and timestamps"),
        ("analyze", "Estimate each recorded angle/rate closed loop"),
        ("compare", "Compare named groups using a comparison.json manifest"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("input", type=Path, help="CSV file, run directory or comparison manifest")
        command.add_argument("--out", type=Path, required=name != "validate", help="New or empty report directory")
        if name != "validate":
            command.add_argument("--config", type=Path, help="JSON object overriding AnalysisConfig defaults")
            command.add_argument(
                "--interval", type=float, nargs=2, action="append", metavar=("START", "END"),
                help="Analyze this recorded-time interval in seconds; repeat for disjoint intervals",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_run(args.input)
            print(json.dumps(result, indent=2, allow_nan=False))
            if args.out is not None:
                from .report import write_report
                write_report(result, args.out, protected_paths=[args.input])
            return 0 if result["valid"] else 2
        config = AnalysisConfig(**read_json(args.config)) if args.config else AnalysisConfig()
        if args.interval is not None:
            config = replace(config, analysis_intervals_s=args.interval)
        result = analyze_run(args.input, config) if args.command == "analyze" else compare_groups(args.input, config)
        from .report import write_report
        report = write_report(result, args.out)
        entries = result["channels"].values() if args.command == "analyze" else result["cases"]
        usable = sum(entry["status"] == "ok" for entry in entries)
        print(f"Report: {report}\nUsable channels/cases: {usable}")
        return 0 if usable else 3
    except (ValidationError, OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
