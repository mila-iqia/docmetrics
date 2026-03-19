#!/usr/bin/env python3
"""Formats docmetrics results as a GitHub PR comment body (Markdown)."""

import argparse
import json
from pathlib import Path


def fmt_score(score: float, correct: int, total: int) -> str:
    return f"{score:.0%} ({correct}/{total})"


def fmt_delta(current: float, base: float) -> str:
    delta_pp = round((current - base) * 100)
    sign = "+" if delta_pp >= 0 else ""
    return f"{sign}{delta_pp}pp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, help="Path to current branch results JSON.")
    parser.add_argument("--base", help="Path to base branch results JSON.")
    parser.add_argument("--questions-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-ref", default="base")
    parser.add_argument(
        "--base-source",
        choices=["cached", "computed"],
        default=None,
        help="How the base branch results were obtained (for display).",
    )
    args = parser.parse_args()

    with open(args.current) as f:
        current = json.load(f)

    base = None
    if args.base and Path(args.base).exists():
        with open(args.base) as f:
            base = json.load(f)

    cur_no = current["without_docs"]
    cur_wi = current["with_docs"]

    lines = [
        "<!-- docmetrics -->",
        "## DocMetrics Results",
        "",
        "| | Without docs | With docs | Delta (docs − no&nbsp;docs) |",
        "|---|---|---|---|",
    ]

    lines.append(
        f"| **This PR** "
        f"| {fmt_score(cur_no['score'], cur_no['correct_answers'], cur_no['num_questions'])} "
        f"| {fmt_score(cur_wi['score'], cur_wi['correct_answers'], cur_wi['num_questions'])} "
        f"| {fmt_delta(cur_wi['score'], cur_no['score'])} |"
    )

    if base:
        base_no = base["without_docs"]
        base_wi = base["with_docs"]

        label = f"`{args.base_ref}`"
        if args.base_source == "cached":
            label += " *(cached)*"
        elif args.base_source == "computed":
            label += " *(computed)*"

        lines.append(
            f"| **Base ({label})** "
            f"| {fmt_score(base_no['score'], base_no['correct_answers'], base_no['num_questions'])} "
            f"| {fmt_score(base_wi['score'], base_wi['correct_answers'], base_wi['num_questions'])} "
            f"| {fmt_delta(base_wi['score'], base_no['score'])} |"
        )
        lines.append(
            f"| **Change** "
            f"| {fmt_delta(cur_no['score'], base_no['score'])} "
            f"| {fmt_delta(cur_wi['score'], base_wi['score'])} "
            f"| — |"
        )

    lines += [
        "",
        f"*Model: `{args.model}` · Questions: `{args.questions_file}`*",
    ]

    print("\n".join(lines))


if __name__ == "__main__":
    main()
