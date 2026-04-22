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
    QuestionResult,
    Response,
    ask_question,
    evaluate_llm,
    get_agent_answer,
    shuffle_options,
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
    """Returns the correct letter when the LLM answers correctly."""
    q = QUESTIONS[0]  # answer is "A"
    result = ask_question(mock_client("A"), q, with_docs=False, model="fake-model", docs_urls=None, tools=None)
    assert result == "A"


def test_ask_question_incorrect():
    """Returns the selected letter (not the correct one) when the LLM answers incorrectly."""
    q = QUESTIONS[0]  # answer is "A"
    result = ask_question(mock_client("B"), q, with_docs=False, model="fake-model", docs_urls=None, tools=None)
    assert result == "B"


def test_ask_question_invalid():
    """Returns None when the LLM response cannot be parsed."""
    q = QUESTIONS[0]
    client = MagicMock()
    response = MagicMock()
    response.candidates = [types.Candidate(index=0)]
    response.text = "xyz 123 !!! $$"
    response.parsed = None
    client.models.generate_content.return_value = response

    assert ask_question(client, q, with_docs=False, model="fake-model", docs_urls=None, tools=None) is None


# ---------------------------------------------------------------------------
# evaluate_llm tests
# ---------------------------------------------------------------------------


def test_evaluate_llm_score():
    """Score matches the number of random answers that happen to be correct."""
    rng = random.Random(42)
    answers: list[Letter] = [rng.choice(list(q.options.keys())) for q in QUESTIONS]
    expected_correct = sum(a == q.answer for a, q in zip(answers, QUESTIONS))

    with (
        patch("docmetrics.main.get_google_genai_client") as mock_factory,
        patch("docmetrics.main.shuffle_options", side_effect=lambda q: q),
    ):
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [make_fake_response(a) for a in answers]

        result = evaluate_llm(QUESTIONS, with_docs=False, model="fake-model")
        mock_factory.assert_called()

    assert isinstance(result, EvaluationResult)
    assert result.num_questions == len(QUESTIONS)
    assert result.correct_answers == expected_correct
    assert result.invalid_answers == 0
    assert all(isinstance(qr, QuestionResult) for qr in result.answers)
    assert all(len(qr.runs) == 1 for qr in result.answers)


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
    assert len(result.answers) == len(QUESTIONS)
    assert all(isinstance(qr, QuestionResult) for qr in result.answers)
    assert all(qr.runs[0] in qr.expected or qr.runs[0] != qr.expected for qr in result.answers)


def test_evaluate_llm_num_candidates():
    """With num_candidates=N, each QuestionResult has N runs."""
    n = 3
    answers: list[Letter] = ["A", "B", "A", "C", "B", "B", "A", "C", "B"]  # 3 per question
    with (
        patch("docmetrics.main.get_google_genai_client") as mock_factory,
        patch("docmetrics.main.shuffle_options", side_effect=lambda q: q),
    ):
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.side_effect = [make_fake_response(a) for a in answers]

        result = evaluate_llm(QUESTIONS, with_docs=False, model="fake-model", num_candidates=n)

    assert result.num_candidates == n
    assert all(len(qr.runs) == n for qr in result.answers)
    assert result.correct_answers == sum(
        sum(1 for run in qr.runs if run == qr.expected) for qr in result.answers
    )


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
# shuffle_options tests
# ---------------------------------------------------------------------------


def test_shuffle_options_preserves_texts_and_updates_answer():
    """Shuffled question has same option texts and correct answer text is unchanged."""
    random.seed(0)
    q = Question(question="Q?", options={"A": "correct", "B": "wrong1", "C": "wrong2"}, answer="A")
    shuffled = shuffle_options(q)
    assert set(shuffled.options.values()) == set(q.options.values())
    assert shuffled.options[shuffled.answer] == q.options[q.answer]


def test_shuffle_options_single_option_unchanged():
    """A question with one option is returned unchanged."""
    q = Question(question="Q?", options={"A": "only"}, answer="A")
    assert shuffle_options(q) is q


def test_shuffle_options_answer_always_valid():
    """Shuffled answer letter is always one of the option keys."""
    for seed in range(20):
        random.seed(seed)
        q = Question(
            question="Q?",
            options={"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
            answer="C",
        )
        shuffled = shuffle_options(q)
        assert shuffled.answer in shuffled.options
        assert shuffled.options[shuffled.answer] == q.options[q.answer]
