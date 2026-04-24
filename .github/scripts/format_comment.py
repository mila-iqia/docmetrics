#!/usr/bin/env python3
"""Formats docmetrics results as a GitHub PR comment body (Markdown).

Takes three result JSONs (no-docs baseline, with-base-docs, with-PR-docs) and
produces a three-row comparison table. The headline delta compares the two
doc-providing runs: with-PR-docs − with-base-docs.
"""

import argparse
import json
import math
import textwrap

from docmetrics.objects import EvaluationResult, QuestionResult


def format_comment(
    without_docs: EvaluationResult,
    with_base_docs: EvaluationResult,
    with_docs: EvaluationResult,
    questions_file: str,
    model: str,
) -> str:
    """Return the full Markdown body for the PR comment."""
    # NOTE: We only use the quiz in the PR. No need to worry about the questions changing.
    # _questions = [qr.question for qr in with_base_docs.question_results]

    lines: list[str] = ["<!-- docmetrics -->", "## DocMetrics Report", ""]

    # Summary line: PR docs vs base docs.
    delta_pp = with_docs.score - with_base_docs.score
    delta_std = math.sqrt(with_docs.score_std**2 + with_base_docs.score_std**2)
    if delta_pp > 0:
        lines.append(
            f"> Merging this PR will **increase** the with-docs score by {delta_pp:.1%}±{delta_std:.1%}"
            f" compared to the base docs :arrow_up:."
        )
    elif delta_pp < 0:
        lines.append(
            f"> Merging this PR will **decrease** the with-docs score by {delta_pp:.1%}±{delta_std:.1%}"
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
        f"| **Without docs**   | {_score_cell(without_docs)} | — |",
        f"| **With base docs** | {_score_cell(with_base_docs)} | {_fmt_delta(with_base_docs, without_docs)} |",
        f"| **With PR docs**   | {_score_cell(with_docs)} | {_fmt_delta(with_docs, without_docs)} |",
        "",
    ]
    if with_docs.answers != with_base_docs.answers:
        num_changed = sum(
            set(q_a.runs) != set(q_b.runs)
            for q_a, q_b in zip(with_docs.question_results, with_base_docs.question_results)
        )

        lines += [
            "<details>",
            f"<summary>📋 {num_changed} question(s) where base docs and PR docs disagree</summary>",
            "",
            "| # | Question | Base docs → PR docs |",
            "|:---|:---|:---:|",
        ]
        # Per-question block: questions where base-docs and PR-docs disagree.
        for i, (base_qa, pr_qa) in enumerate(
            zip(with_base_docs.question_results, with_docs.question_results)
        ):
            if set(pr_qa.runs) != set(base_qa.runs):
                q_label = question_label(base_qa, i)
                lines.append(
                    f"| {i + 1} | {q_label} | {base_qa.correct_count}/{base_qa.total} → {pr_qa.correct_count}/{pr_qa.total} |"
                )
        lines += ["", "</details>", ""]

    lines += [
        "",
        f"*Model: `{model}` · Questions: `{questions_file}`*",
    ]

    return "\n".join(lines)


def _fmt_score(
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


def _fmt_delta(current: EvaluationResult, base: EvaluationResult) -> str:
    delta_pp = round((current.score - base.score) * 100)
    new_variance = current.score_std**2
    old_variance = base.score_std**2
    new_std = math.sqrt(new_variance + old_variance)
    sign = "+" if delta_pp >= 0 else ""
    return f"{sign}{delta_pp}pp (±{new_std:.0%})"


def _result_icon(a: "dict | bool | None") -> str:
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


def question_label(question: QuestionResult, i: int, max_len: int = 80) -> str:
    """Return a display label for question *i* (truncated question text, or Q<n>)."""
    text = question.question.question
    text = " ".join(text.split())
    text = textwrap.shorten(text, width=max_len, placeholder="…")
    return text.replace("|", "\\|")


def _score_cell(result: EvaluationResult) -> str:
    score = result.score
    correct = result.correct_answers
    total = result.num_questions
    num_candidates = result.num_candidates
    score_std = result.score_std
    s = f"{score:.0%} ({correct}/{total * num_candidates})"
    if score_std is not None:
        s += f" ±{score_std:.0%}"
    return s


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
        without_docs = EvaluationResult.model_validate(json.load(f))
    with open(args.with_base_docs) as f:
        with_base_docs = EvaluationResult.model_validate(json.load(f))
    with open(args.with_docs) as f:
        with_docs = EvaluationResult.model_validate(json.load(f))

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
