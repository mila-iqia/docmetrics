from typing import Sequence
from unittest.mock import MagicMock, patch

from docmetrics.main import Question, run_quiz

QUESTIONS = [
    Question(question="Q1?", options={"A": "opt1", "B": "opt2", "C": "opt3"}, answer="A"),
    Question(question="Q2?", options={"A": "opt1", "B": "opt2"}, answer="B"),
    Question(
        question="Q3?",
        options={"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
        answer="C",
    ),
]


def _quiz_answers(answers: Sequence[str | None]):
    """Patch questionary.select so .ask() returns successive values from `answers`."""
    return patch(
        "docmetrics.main.questionary.select",
        return_value=MagicMock(ask=MagicMock(side_effect=answers)),
    )


def test_run_quiz_all_correct():
    """Returns a perfect score when all answers are correct."""
    answers = ["A: opt1", "B: opt2", "C: opt3"]
    with _quiz_answers(answers):
        result = run_quiz(QUESTIONS)
    assert result.correct_answers == 3
    assert result.invalid_answers == 0
    assert result.score == 1.0


def test_run_quiz_all_wrong():
    """Returns zero correct answers when all answers are wrong."""
    answers = ["B: opt2", "A: opt1", "A: opt1"]
    with _quiz_answers(answers):
        result = run_quiz(QUESTIONS)
    assert result.correct_answers == 0
    assert result.num_questions == 3


def test_run_quiz_mixed():
    """Counts only the questions answered correctly."""
    # Q1 correct (A), Q2 wrong (A instead of B), Q3 correct (C)
    answers = ["A: opt1", "A: opt1", "C: opt3"]
    with _quiz_answers(answers):
        result = run_quiz(QUESTIONS)
    assert result.correct_answers == 2
    assert result.num_questions == 3


def test_run_quiz_quit_with_q():
    """Quitting with 'q: Quit' stops the quiz and reflects only answered questions."""
    # Answer Q1 correctly, then quit on Q2
    answers = ["A: opt1", "q: Quit"]
    with _quiz_answers(answers):
        result = run_quiz(QUESTIONS)
    assert result.correct_answers == 1
    assert result.num_questions == 3  # total is always the full set


def test_run_quiz_quit_with_ctrl_c():
    """Ctrl+C (questionary returns None) stops the quiz."""
    # Answer Q1 wrong, then Ctrl+C on Q2
    answers = ["B: opt2", None]
    with _quiz_answers(answers):
        result = run_quiz(QUESTIONS)
    assert result.correct_answers == 0
    assert result.num_questions == 3
