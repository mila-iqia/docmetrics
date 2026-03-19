import json
import random
import typing
from unittest.mock import MagicMock, patch

from google import genai
from google.genai import types

from docmetrics.main import (
    DUMMY_MODEL,
    EvaluationResult,
    Letter,
    Question,
    Response,
    ask_question,
    evaluate_llm,
    get_agent_answer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUESTIONS = [
    Question(question="Q1?", options={"A": "opt1", "B": "opt2", "C": "opt3"}, answer="A"),
    Question(question="Q2?", options={"A": "opt1", "B": "opt2"}, answer="B"),
    Question(
        question="Q3?",
        options={"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
        answer="C",
    ),
]


def make_fake_response(
    answer: Letter,
    *,
    justification: str = "Fake justification.",
    use_parsed: bool = True,
    with_url_metadata: bool = False,
) -> MagicMock:
    """Build a fake GenerateContentResponse-like object.

    Uses real google.genai.types objects for `candidates` to match the actual
    API schema, but wraps them in a MagicMock so that the computed `text` and
    `parsed` properties can be freely set.
    """
    response_obj = Response(answer=answer, justification=justification)
    response_json = response_obj.model_dump_json()

    url_context_metadata = None
    if with_url_metadata:
        url_context_metadata = types.UrlContextMetadata(
            url_metadata=[
                types.UrlMetadata(
                    retrieved_url="https://example.com/docs",
                    url_retrieval_status=types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                )
            ]
        )

    candidate = types.Candidate(
        index=0,
        content=types.Content(role="model", parts=[types.Part(text=response_json)]),
        grounding_metadata=None,
        url_context_metadata=url_context_metadata,
    )

    response = MagicMock()
    response.candidates = [candidate]
    response.text = response_json
    response.parsed = response_obj if use_parsed else None
    return response


def mock_client(answer: Letter, **kwargs) -> MagicMock:
    """Return a mock genai.Client whose generate_content returns a fake response."""
    client = MagicMock()
    client.models.generate_content.return_value = make_fake_response(answer, **kwargs)
    return client


# ---------------------------------------------------------------------------
# get_agent_answer tests
# ---------------------------------------------------------------------------


def test_get_agent_answer_via_parsed():
    """Returns the Response directly when response.parsed is already a Response."""
    result = get_agent_answer(mock_client("B"), "fake-model", tools=None, prompt="Q?")
    assert isinstance(result, Response)
    assert result.answer == "B"


def test_get_agent_answer_via_text_json():
    """Falls back to JSON-parsing response.text when response.parsed is None."""
    result = get_agent_answer(
        mock_client("C", use_parsed=False), "fake-model", tools=None, prompt="Q?"
    )
    assert isinstance(result, Response)
    assert result.answer == "C"


def test_get_agent_answer_via_fallback():
    """Falls back to parse_response_fallback() for plain text containing a letter."""
    client = MagicMock()
    response = MagicMock()
    response.candidates = [types.Candidate(index=0)]
    response.text = "The best answer is clearly option D. Answer: D"
    response.parsed = None
    client.models.generate_content.return_value = response

    result = get_agent_answer(client, "fake-model", tools=None, prompt="Q?")
    assert isinstance(result, Response)
    assert result.answer == "D"


def test_get_agent_answer_returns_none_for_garbage():
    """Returns None when the response cannot be parsed at all."""
    client = MagicMock()
    response = MagicMock()
    response.candidates = [types.Candidate(index=0)]
    response.text = "xyz 123 !!! no letter here at all $$"
    response.parsed = None
    client.models.generate_content.return_value = response

    result = get_agent_answer(client, "fake-model", tools=None, prompt="Q?")
    assert result is None


def test_get_agent_answer_with_url_metadata():
    """Handles a response with url_context_metadata without errors."""
    result = get_agent_answer(
        mock_client("A", with_url_metadata=True), "fake-model", tools=None, prompt="Q?"
    )
    assert isinstance(result, Response)
    assert result.answer == "A"


# ---------------------------------------------------------------------------
# ask_question tests
# ---------------------------------------------------------------------------


def test_ask_question_correct():
    """Returns True when the LLM answers correctly."""
    q = QUESTIONS[0]  # answer is "A"
    assert (
        ask_question(mock_client("A"), q, with_docs=False, model="fake-model", tools=None) is True
    )


def test_ask_question_incorrect():
    """Returns False when the LLM answers incorrectly."""
    q = QUESTIONS[0]  # answer is "A"
    assert (
        ask_question(mock_client("B"), q, with_docs=False, model="fake-model", tools=None) is False
    )


def test_ask_question_invalid():
    """Returns None when the LLM response cannot be parsed."""
    q = QUESTIONS[0]
    client = MagicMock()
    response = MagicMock()
    response.candidates = [types.Candidate(index=0)]
    response.text = "xyz 123 !!! $$"
    response.parsed = None
    client.models.generate_content.return_value = response

    assert ask_question(client, q, with_docs=False, model="fake-model", tools=None) is None


# ---------------------------------------------------------------------------
# evaluate_llm tests
# ---------------------------------------------------------------------------


def test_evaluate_llm_score():
    """Score matches the number of random answers that happen to be correct."""
    rng = random.Random(42)
    answers: list[Letter] = [rng.choice(list(q.options.keys())) for q in QUESTIONS]
    expected_correct = sum(a == q.answer for a, q in zip(answers, QUESTIONS))

    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [make_fake_response(a) for a in answers]

        result = evaluate_llm(QUESTIONS, with_docs=False, model="fake-model")
        mock_factory.assert_called()

    assert isinstance(result, EvaluationResult)
    assert result.num_questions == len(QUESTIONS)
    assert result.correct_answers == expected_correct
    assert result.invalid_answers == 0


def test_evaluate_llm_with_docs_adds_url_tool():
    """When with_docs=True, each generate_content call includes a UrlContext tool."""
    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = typing.cast(genai.Client, MagicMock())
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [  # type: ignore
            make_fake_response(q.answer) for q in QUESTIONS
        ]

        evaluate_llm(QUESTIONS, with_docs=True, model="fake-model")

    for call in client.models.generate_content.call_args_list:  # type: ignore
        config: types.GenerateContentConfig = call.kwargs["config"]
        assert config.tools is not None
        assert any(
            isinstance(t, types.Tool) and t.url_context is not None for t in config.tools
        ), "Expected a UrlContext tool in the generate_content call"


def test_evaluate_llm_dummy_model():
    """test:dummy model returns results without any API calls."""
    result = evaluate_llm(QUESTIONS, with_docs=False, model=DUMMY_MODEL)
    assert result.num_questions == len(QUESTIONS)
    assert result.invalid_answers == 0
    assert result.correct_answers <= result.num_questions


def test_evaluate_llm_dummy_model_with_docs():
    """test:dummy model ignores with_docs=True and makes no API calls."""
    result = evaluate_llm(QUESTIONS, with_docs=True, model=DUMMY_MODEL)
    assert result.num_questions == len(QUESTIONS)
    assert result.invalid_answers == 0


def test_evaluate_llm_without_docs_no_url_tool():
    """When with_docs=False, no UrlContext tool is passed to generate_content."""
    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = typing.cast(genai.Client, MagicMock())
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [  # type: ignore
            make_fake_response(q.answer) for q in QUESTIONS
        ]

        evaluate_llm(QUESTIONS, with_docs=False, model="fake-model")

    for call in client.models.generate_content.call_args_list:  # type: ignore
        config: types.GenerateContentConfig = call.kwargs["config"]
        tools = config.tools or []
        assert not any(isinstance(t, types.Tool) and t.url_context is not None for t in tools), (
            "Did not expect a UrlContext tool when with_docs=False"
        )



# ---------------------------------------------------------------------------
# main() CLI tests
# ---------------------------------------------------------------------------


def _make_all_correct_side_effects(questions, n_passes):
    """Return side_effect list for n_passes over questions, all answers correct."""
    return [make_fake_response(q.answer) for q in questions] * n_passes


def test_main_no_docs_urls_runs_baseline_only(tmp_path, capsys):
    """Without --docs-urls, only the without-docs pass runs."""
    import yaml

    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        yaml.dump([{"question": "Q?", "options": {"A": "opt1", "B": "opt2"}, "answer": "A"}])
    )

    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [make_fake_response("A")]

        with patch("sys.argv", ["docmetrics", "evaluate", "--questions", str(questions_file), "--output", "json"]):
            from docmetrics.main import main
            main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "without_docs" in data
    assert "with_docs" not in data
    assert client.models.generate_content.call_count == 1


def test_main_with_docs_urls_runs_both_passes(tmp_path, capsys):
    """With --docs-urls and no --with-docs-only, both passes run."""
    import yaml

    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        yaml.dump([{"question": "Q?", "options": {"A": "opt1", "B": "opt2"}, "answer": "A"}])
    )

    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [make_fake_response("A"), make_fake_response("A")]

        with patch("sys.argv", ["docmetrics", "evaluate", "--questions", str(questions_file), "--output", "json", "--docs-urls", "https://docs.example.com"]):
            from docmetrics.main import main
            main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "without_docs" in data
    assert "with_docs" in data


def test_main_with_docs_only(tmp_path, capsys):
    """--with-docs-only skips the without-docs pass."""
    import yaml

    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        yaml.dump([{"question": "Q?", "options": {"A": "opt1", "B": "opt2"}, "answer": "A"}])
    )

    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [make_fake_response("A")]

        with patch("sys.argv", ["docmetrics", "evaluate", "--questions", str(questions_file), "--output", "json", "--docs-urls", "https://docs.example.com", "--with-docs-only"]):
            from docmetrics.main import main
            main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "with_docs" in data
    assert "without_docs" not in data
    assert client.models.generate_content.call_count == 1


def test_main_docs_urls_appear_in_prompt(tmp_path, capsys):
    """--docs-urls passes URLs to the prompt."""
    import yaml

    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        yaml.dump([{"question": "Q?", "options": {"A": "opt1", "B": "opt2"}, "answer": "A"}])
    )

    captured_prompts = []

    def fake_generate(model, contents, config):
        captured_prompts.append(contents)
        return make_fake_response("A")

    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.side_effect = fake_generate

        with patch(
            "sys.argv",
            ["docmetrics", "evaluate", "--questions", str(questions_file), "--with-docs-only", "--docs-urls", "https://docs.example.com/page1"],
        ):
            from docmetrics.main import main
            main()

    assert any("https://docs.example.com/page1" in p for p in captured_prompts)
