#!/usr/bin/env python3
"""Formats docmetrics results as a GitHub PR comment body (Markdown).

Takes three result JSONs (no-docs baseline, with-base-docs, with-PR-docs) and
produces a three-row comparison table. The headline delta compares the two
doc-providing runs: with-PR-docs − with-base-docs.
"""

import argparse
import json


def fmt_score(
    score: float,
    correct: int,
    total: int,
    score_std: float | None = None,
    num_candidates: int = 1,
) -> str:
    s = f"{score:.0%} ({correct}/{total * num_candidates})"
    if score_std is not None:
        s += f" ±{score_std:.0%}"
    return s


def fmt_delta(current: float, base: float) -> str:
    delta_pp = round((current - base) * 100)
    sign = "+" if delta_pp >= 0 else ""
    return f"{sign}{delta_pp}pp"


def result_icon(a: "dict | bool | None") -> str:
    if isinstance(a, dict):
        if "pass_rate" in a:
            selected = a.get("selected", [])
            n = len(selected)
            correct = round(a["pass_rate"] * n)
            if correct == n:
                return "✅"
            if correct == 0:
                return "❌"
            return f"🔄 {correct}/{n}"
        a = a.get("correct")
    if a is True:
        return "✅"
    if a is False:
        return "❌"
    return "❓"


def _answer_score(a: "dict | bool | None") -> "bool | float | None":
    """Extract a comparable correctness score from an answer entry."""
    if isinstance(a, dict):
        if "pass_rate" in a:
            return a["pass_rate"]
        return a.get("correct")
    return a


def question_label(questions: list | None, i: int, max_len: int = 80) -> str:
    """Return a display label for question *i* (truncated question text, or Q<n>)."""
    if not questions or i >= len(questions):
        return f"Q{i + 1}"
    text = questions[i].get("question", f"Q{i + 1}")
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text.replace("|", "\\|")


def _score_cell(result: dict) -> str:
    return fmt_score(
        result["score"],
        result["correct_answers"],
        result["num_questions"],
        result.get("score_std"),
        result.get("num_candidates", 1),
    )


def format_comment(
    without_docs: dict,
    with_base_docs: dict,
    with_docs: dict,
    questions_file: str,
    model: str,
) -> str:
    """Return the full Markdown body for the PR comment."""
    no_res = without_docs
    base_res = with_base_docs
    pr_res = with_docs
    questions: list | None = with_docs.get("questions") or without_docs.get("questions")

    lines: list[str] = ["<!-- docmetrics -->", "## DocMetrics Report", ""]

    # Summary line: PR docs vs base docs.
    delta_pp = round((pr_res["score"] - base_res["score"]) * 100)
    delta_str = fmt_delta(pr_res["score"], base_res["score"])
    if delta_pp > 0:
        lines.append(
            f"> Merging this PR will **increase** the with-docs score by `{delta_str}`"
            f" compared to the base docs :arrow_up:."
        )
    elif delta_pp < 0:
        lines.append(
            f"> Merging this PR will **decrease** the with-docs score by `{delta_str}`"
            f" compared to the base docs :arrow_down:."
        )
    else:
        lines.append(
            "> Merging this PR will **not change** the with-docs score compared to the base docs."
        )
    lines.append("")

    # Three-row comparison table.
    lines += [
        "| | Score | Δ vs. no docs |",
        "|:---|:---:|:---:|",
        f"| **Without docs** | {_score_cell(no_res)} | — |",
        f"| **With base docs** | {_score_cell(base_res)}"
        f" | {fmt_delta(base_res['score'], no_res['score'])} |",
        f"| **With PR docs** | {_score_cell(pr_res)}"
        f" | {fmt_delta(pr_res['score'], no_res['score'])} |",
        "",
    ]

    # Per-question block: questions where base-docs and PR-docs disagree.
    base_answers: list = base_res.get("answers", [])
    pr_answers: list = pr_res.get("answers", [])
    changed = [
        (i, b, p)
        for i, (b, p) in enumerate(zip(base_answers, pr_answers))
        if _answer_score(b) != _answer_score(p)
    ]
    if changed:
        lines += [
            "<details>",
            f"<summary>📋 {len(changed)} question(s) where base docs and PR docs disagree</summary>",
            "",
            "| # | Question | Base docs → PR docs |",
            "|:---|:---|:---:|",
        ]
        for i, b, p in changed:
            q_label = question_label(questions, i)
            lines.append(f"| {i + 1} | {q_label} | {result_icon(b)} → {result_icon(p)} |")
        lines += ["", "</details>", ""]

    lines += [
        "",
        f"*Model: `{model}` · Questions: `{questions_file}`*",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-docs", required=True, help="Path to no-docs result JSON.")
    parser.add_argument(
        "--with-base-docs", required=True, help="Path to with-base-docs result JSON."
    )
    parser.add_argument("--with-docs", required=True, help="Path to with-PR-docs result JSON.")
    parser.add_argument("--questions-file", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with open(args.without_docs) as f:
        without_docs = json.load(f)
    with open(args.with_base_docs) as f:
        with_base_docs = json.load(f)
    with open(args.with_docs) as f:
        with_docs = json.load(f)

    print(
        format_comment(
            without_docs=without_docs,
            with_base_docs=with_base_docs,
            with_docs=with_docs,
            questions_file=args.questions_file,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
