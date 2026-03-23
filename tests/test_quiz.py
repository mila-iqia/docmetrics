from unittest.mock import MagicMock, patch


from docmetrics.main import Question
from docmetrics.quiz import run_quiz

QUESTIONS = [
    Question(question="Q1?", options={"A": "opt1", "B": "opt2", "C": "opt3"}, answer="A"),
    Question(question="Q2?", options={"A": "opt1", "B": "opt2"}, answer="B"),
    Question(
        question="Q3?",
        options={"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
        answer="C",
    ),
]


def _mock_select(answers: list[str | None]):
    """Return a patched questionary.select that yields answers in sequence."""
    call_count = 0

    def fake_select(prompt, choices):
        nonlocal call_count
        answer = answers[call_count]
        call_count += 1
        mock = MagicMock()
        if answer is KeyboardInterrupt:
            mock.ask.side_effect = KeyboardInterrupt
        else:
            mock.ask.return_value = answer
        return mock

    return fake_select


def test_quiz_all_correct(capsys):
    answers = [q.answer for q in QUESTIONS]
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    assert "Correct" in out
    assert "Wrong" not in out
    assert f"Score: {len(QUESTIONS)}/{len(QUESTIONS)}" in out


def test_quiz_some_wrong(capsys):
    # Answer correctly only the last question
    answers = ["B", "A", QUESTIONS[2].answer]  # Q1 wrong, Q2 wrong, Q3 correct
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    assert "Score: 1/3" in out


def test_quiz_quit_with_q(capsys):
    # Answer first question correctly, then quit
    answers = [QUESTIONS[0].answer, "q"]
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    assert "Exiting quiz" in out
    assert "Score: 1/1" in out


def test_quiz_quit_none(capsys):
    # questionary returns None (e.g. user pressed Ctrl+C in some versions)
    answers = [QUESTIONS[0].answer, None]
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    assert "Exiting quiz" in out
    assert "Score: 1/1" in out


def test_quiz_keyboard_interrupt(capsys):
    # Ctrl+C on the very first question: score should not be printed (0 answered)
    answers = [KeyboardInterrupt]
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    assert "Exiting quiz" in out
    # No score line since nothing was answered
    assert "Score:" not in out


def test_quiz_keyboard_interrupt_after_one(capsys):
    # Ctrl+C after answering one question correctly
    answers = [QUESTIONS[0].answer, KeyboardInterrupt]
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    assert "Score: 1/1" in out


def test_quiz_all_questions_answered_shows_score(capsys):
    answers = [q.answer for q in QUESTIONS]
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    assert f"Score: {len(QUESTIONS)}/{len(QUESTIONS)}" in out


def test_quiz_empty_questions(capsys):
    with patch("docmetrics.quiz.questionary.select") as mock_select:
        run_quiz([])
    mock_select.assert_not_called()
    out = capsys.readouterr().out
    assert "Score:" not in out


def test_quiz_options_displayed_in_output(capsys):
    """Option text is printed via Rich before the select widget, so long answers wrap properly."""
    answers = [QUESTIONS[0].answer, "q"]
    with patch("docmetrics.quiz.questionary.select", side_effect=_mock_select(answers)):
        run_quiz(QUESTIONS)
    out = capsys.readouterr().out
    for letter, text in QUESTIONS[0].options.items():
        assert f"{letter}: {text}" in out
