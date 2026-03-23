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


def result_icon(r: bool | None) -> str:
    if r is True:
        return "✅"
    if r is False:
        return "❌"
    return "❓"


def question_label(questions: list | None, i: int, max_len: int = 80) -> str:
    """Return a display label for question *i* (truncated question text, or Q<n>)."""
    if not questions or i >= len(questions):
        return f"Q{i + 1}"
    text = questions[i].get("question", f"Q{i + 1}")
    text = " ".join(text.split())  # normalise whitespace
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text.replace("|", "\\|")  # escape pipe so it doesn't break the table


def format_comment(
    current: dict,
    base: dict | None,
    questions_file: str,
    model: str,
    base_ref: str = "base",
    base_source: str | None = None,
) -> str:
    """Return the full Markdown body for the PR comment."""
    cur_no = current["without_docs"]
    cur_wi = current["with_docs"]
    questions: list | None = current.get("questions")

    lines: list[str] = ["<!-- docmetrics -->", "## DocMetrics Report", ""]

    # ── Summary line (only when a base comparison is available) ──────────────
    if base:
        base_wi = base["with_docs"]
        delta_pp = round((cur_wi["score"] - base_wi["score"]) * 100)
        base_ref_label = f"`{base_ref}`"
        delta_str = fmt_delta(cur_wi["score"], base_wi["score"])
        if delta_pp > 0:
            lines.append(
                f"> Merging this PR into {base_ref_label} will **increase** the"
                f" with-docs score by `{delta_str}` :arrow_up:."
            )
        elif delta_pp < 0:
            lines.append(
                f"> Merging this PR into {base_ref_label} will **decrease** the"
                f" with-docs score by `{delta_str}` :arrow_down:."
            )
        else:
            lines.append(
                f"> Merging this PR into {base_ref_label} will **not change** the with-docs score."
            )
        lines.append("")

    # ── Scores table ─────────────────────────────────────────────────────────
    lines += [
        "| | Without docs | With docs | Δ&nbsp;(docs&nbsp;−&nbsp;no&nbsp;docs) |",
        "|:---|:---:|:---:|:---:|",
        f"| **This PR**"
        f" | {fmt_score(cur_no['score'], cur_no['correct_answers'], cur_no['num_questions'])}"
        f" | {fmt_score(cur_wi['score'], cur_wi['correct_answers'], cur_wi['num_questions'])}"
        f" | {fmt_delta(cur_wi['score'], cur_no['score'])} |",
    ]

    if base:
        base_no = base["without_docs"]
        base_wi = base["with_docs"]
        label = f"`{base_ref}`"
        if base_source == "cached":
            label += " *(cached)*"
        elif base_source == "computed":
            label += " *(computed)*"
        lines += [
            f"| **Base ({label})**"
            f" | {fmt_score(base_no['score'], base_no['correct_answers'], base_no['num_questions'])}"
            f" | {fmt_score(base_wi['score'], base_wi['correct_answers'], base_wi['num_questions'])}"
            f" | {fmt_delta(base_wi['score'], base_no['score'])} |",
            f"| **Change**"
            f" | {fmt_delta(cur_no['score'], base_no['score'])}"
            f" | {fmt_delta(cur_wi['score'], base_wi['score'])}"
            f" | — |",
        ]

    lines.append("")

    # ── Per-question expandable block ─────────────────────────────────────────
    cur_no_answers: list = cur_no.get("answers", [])
    cur_wi_answers: list = cur_wi.get("answers", [])

    if base:
        base_no = base["without_docs"]
        base_wi = base["with_docs"]

        label = f"`{base_ref}`"
        if base_source == "cached":
            label += " *(cached)*"
        elif base_source == "computed":
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

    elif cur_no_answers or cur_wi_answers:
        n = max(len(cur_no_answers), len(cur_wi_answers))
        lines += [
            "<details>",
            "<summary>📋 Per-question results</summary>",
            "",
            "| # | Question | Without docs | With docs |",
            "|:---|:---|:---:|:---:|",
        ]
        for i in range(n):
            c_no = cur_no_answers[i] if i < len(cur_no_answers) else None
            c_wi = cur_wi_answers[i] if i < len(cur_wi_answers) else None
            q_label = question_label(questions, i)
            lines.append(f"| {i + 1} | {q_label} | {result_icon(c_no)} | {result_icon(c_wi)} |")
        lines += ["", "</details>", ""]

    # ── Footer ───────────────────────────────────────────────────────────────

    lines += [
        "",
        f"*Model: `{model}` · Questions: `{questions_file}`*",
        "*Comparing: PR documentation (current) vs. base documentation*",
    ]

    # print("\n".join(lines))

    # lines += [
    #     "---",
    #     f"*Model: `{model}` · Questions: `{questions_file}`*",
    # ]

    return "\n".join(lines)


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

    print(
        format_comment(
            current=current,
            base=base,
            questions_file=args.questions_file,
            model=args.model,
            base_ref=args.base_ref,
            base_source=args.base_source,
        )
    )


if __name__ == "__main__":
    main()
