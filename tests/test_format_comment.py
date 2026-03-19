"""Tests for .github/scripts/format_comment.py"""

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module (it lives outside the package tree)
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "format_comment.py"
_spec = importlib.util.spec_from_file_location("format_comment", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

fmt_score = _mod.fmt_score
fmt_delta = _mod.fmt_delta
result_icon = _mod.result_icon
question_label = _mod.question_label
format_comment = _mod.format_comment

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_QUESTIONS = [
    {"question": "Which location stores temporary checkpoints on the Mila cluster?"},
    {"question": "Which location stores project code on the Mila cluster?"},
    {"question": "A" * 90},  # deliberately long question text
]

_CURRENT = {
    "questions": _QUESTIONS,
    "without_docs": {
        "num_questions": 3,
        "correct_answers": 1,
        "invalid_answers": 0,
        "score": 1 / 3,
        "answers": [True, False, False],
    },
    "with_docs": {
        "num_questions": 3,
        "correct_answers": 2,
        "invalid_answers": 0,
        "score": 2 / 3,
        "answers": [True, True, False],
    },
}

_BASE = {
    "questions": _QUESTIONS,
    "without_docs": {
        "num_questions": 3,
        "correct_answers": 0,
        "invalid_answers": 0,
        "score": 0.0,
        "answers": [False, False, False],
    },
    "with_docs": {
        "num_questions": 3,
        "correct_answers": 1,
        "invalid_answers": 0,
        "score": 1 / 3,
        "answers": [True, False, False],
    },
}

# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_fmt_score():
    assert fmt_score(0.75, 3, 4) == "75% (3/4)"
    assert fmt_score(0.0, 0, 5) == "0% (0/5)"
    assert fmt_score(1.0, 10, 10) == "100% (10/10)"


def test_fmt_delta_positive():
    assert fmt_delta(0.8, 0.6) == "+20pp"


def test_fmt_delta_negative():
    assert fmt_delta(0.4, 0.6) == "-20pp"


def test_fmt_delta_zero():
    assert fmt_delta(0.5, 0.5) == "+0pp"


def test_result_icon_true():
    assert result_icon(True) == "✅"


def test_result_icon_false():
    assert result_icon(False) == "❌"


def test_result_icon_none():
    assert result_icon(None) == "❓"


def test_question_label_short():
    qs = [{"question": "Short question?"}]
    assert question_label(qs, 0) == "Short question?"


def test_question_label_truncated():
    qs = [{"question": "A" * 90}]
    label = question_label(qs, 0)
    assert len(label) <= 80
    assert label.endswith("…")


def test_question_label_no_questions():
    assert question_label(None, 0) == "Q1"
    assert question_label([], 2) == "Q3"


def test_question_label_out_of_range():
    qs = [{"question": "Only one question"}]
    assert question_label(qs, 5) == "Q6"


def test_question_label_escapes_pipe():
    qs = [{"question": "A | B | C"}]
    label = question_label(qs, 0)
    assert "|" not in label.replace("\\|", "")


# ---------------------------------------------------------------------------
# format_comment – no base
# ---------------------------------------------------------------------------


def test_format_comment_no_base_contains_marker():
    body = format_comment(_CURRENT, None, "sample.yaml", "gemini-2.5-flash")
    assert "<!-- docmetrics -->" in body


def test_format_comment_no_base_heading():
    body = format_comment(_CURRENT, None, "sample.yaml", "gemini-2.5-flash")
    assert "## DocMetrics Report" in body


def test_format_comment_no_base_no_summary_line():
    """Without a base there is no 'Merging this PR…' sentence."""
    body = format_comment(_CURRENT, None, "sample.yaml", "gemini-2.5-flash")
    assert "Merging" not in body


def test_format_comment_no_base_footer():
    body = format_comment(_CURRENT, None, "sample.yaml", "gemini-2.5-flash")
    assert "*Model: `gemini-2.5-flash` · Questions: `sample.yaml`*" in body


def test_format_comment_no_base_per_question_details():
    """All per-question results appear in an expandable block."""
    body = format_comment(_CURRENT, None, "sample.yaml", "gemini-2.5-flash")
    assert "<details>" in body
    assert "Per-question results" in body
    # Three rows – one per question
    assert body.count("✅") + body.count("❌") >= 3


# ---------------------------------------------------------------------------
# format_comment – with base, changes exist
# ---------------------------------------------------------------------------


def test_format_comment_with_base_summary_increase():
    """Score went up → summary says 'increase' with arrow_up."""
    body = format_comment(_CURRENT, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main")
    assert "**increase**" in body
    assert ":arrow_up:" in body


def test_format_comment_with_base_summary_decrease():
    current_lower = {
        **_CURRENT,
        "with_docs": {**_CURRENT["with_docs"], "score": 0.0, "correct_answers": 0},
    }
    body = format_comment(current_lower, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main")
    assert "**decrease**" in body
    assert ":arrow_down:" in body


def test_format_comment_with_base_summary_no_change():
    same = {
        **_CURRENT,
        "with_docs": {**_CURRENT["with_docs"], "score": _BASE["with_docs"]["score"]},
    }
    body = format_comment(same, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main")
    assert "**not change**" in body


def test_format_comment_with_base_table_rows():
    body = format_comment(_CURRENT, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main")
    assert "**This PR**" in body
    assert "**Base" in body
    assert "**Change**" in body


def test_format_comment_base_source_cached():
    body = format_comment(
        _CURRENT, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main", base_source="cached"
    )
    assert "*(cached)*" in body


def test_format_comment_base_source_computed():
    body = format_comment(
        _CURRENT, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main", base_source="computed"
    )
    assert "*(computed)*" in body


# ---------------------------------------------------------------------------
# format_comment – expandable changed-questions block
# ---------------------------------------------------------------------------


def test_format_comment_changed_questions_block():
    """When answers differ, a <details> block listing changed questions appears."""
    body = format_comment(_CURRENT, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main")
    assert "<details>" in body
    assert "question(s) with changed results" in body


def test_format_comment_changed_questions_count():
    """Only questions that actually changed are listed."""
    # Q1 (no-docs): False→True  ✓ changed
    # Q2 (with-docs): False→True ✓ changed
    # Q3: unchanged
    body = format_comment(_CURRENT, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main")
    assert "2 question(s)" in body


def test_format_comment_no_changed_questions_no_block():
    """When nothing changed there is no expanded details block."""
    body = format_comment(_CURRENT, _CURRENT, "q.yaml", "gemini-2.5-flash", base_ref="main")
    assert "<details>" not in body


def test_format_comment_changed_questions_shows_question_text():
    """The question text (or truncated form) appears in the changed-questions table."""
    body = format_comment(_CURRENT, _BASE, "q.yaml", "gemini-2.5-flash", base_ref="main")
    # First question text is short enough to appear in full
    assert "temporary checkpoints" in body


def test_format_comment_no_answers_no_details_block():
    """Older JSON without 'answers' produces no expandable block."""
    current_no_answers = {
        "without_docs": {k: v for k, v in _CURRENT["without_docs"].items() if k != "answers"},
        "with_docs": {k: v for k, v in _CURRENT["with_docs"].items() if k != "answers"},
    }
    body = format_comment(current_no_answers, None, "q.yaml", "gemini-2.5-flash")
    assert "<details>" not in body
