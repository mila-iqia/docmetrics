import random
import typing
from unittest.mock import MagicMock, patch

import pytest
from google import genai
from google.genai import types

from docmetrics.main import (
    DUMMY_MODEL,
    EvaluationResult,
    Letter,
    MultiResponse,
    Question,
    Response,
    ask_question,
    evaluate_llm,
    get_agent_answer,
    get_agent_answers_for_group,
    make_multi_prompt,
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


def make_fake_multi_response(
    answers: list[Letter],
    *,
    justification: str = "Fake justification.",
    use_parsed: bool = True,
) -> MagicMock:
    """Build a fake GenerateContentResponse-like object for multiple questions."""
    response_objs = [Response(answer=a, justification=justification) for a in answers]
    multi_response = MultiResponse(answers=response_objs)
    response_json = multi_response.model_dump_json()

    candidate = types.Candidate(
        index=0,
        content=types.Content(role="model", parts=[types.Part(text=response_json)]),
        grounding_metadata=None,
        url_context_metadata=None,
    )

    response = MagicMock()
    response.candidates = [candidate]
    response.text = response_json
    response.parsed = multi_response if use_parsed else None
    return response


def mock_client_multi(answers: list[Letter], **kwargs) -> MagicMock:
    """Return a mock genai.Client whose generate_content returns a multi-question response."""
    client = MagicMock()
    client.models.generate_content.return_value = make_fake_multi_response(answers, **kwargs)
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
        ask_question(mock_client("A"), q, with_docs=False, model="fake-model", docs_urls=None, tools=None) is True
    )


def test_ask_question_incorrect():
    """Returns False when the LLM answers incorrectly."""
    q = QUESTIONS[0]  # answer is "A"
    assert (
        ask_question(mock_client("B"), q, with_docs=False, model="fake-model", docs_urls=None, tools=None) is False
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

    assert ask_question(client, q, with_docs=False, model="fake-model", docs_urls=None, tools=None) is None


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
        # All QUESTIONS have docs_urls=None → grouped into a single API call.
        client.models.generate_content.return_value = make_fake_multi_response(answers)

        result = evaluate_llm(QUESTIONS, with_docs=False, model="fake-model")
        mock_factory.assert_called()

    assert isinstance(result, EvaluationResult)
    assert result.num_questions == len(QUESTIONS)
    assert result.correct_answers == expected_correct
    assert result.invalid_answers == 0
    assert len(result.per_question_scores) == len(QUESTIONS)
    # One API call for the whole group
    assert client.models.generate_content.call_count == 1


def test_evaluate_llm_with_docs_adds_url_tool():
    """When with_docs=True, the generate_content call includes a UrlContext tool."""
    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = typing.cast(genai.Client, MagicMock())
        mock_factory.return_value = client
        # All QUESTIONS have docs_urls=None → one grouped API call.
        client.models.generate_content.return_value = make_fake_multi_response(  # type: ignore
            [q.answer for q in QUESTIONS]
        )

        evaluate_llm(QUESTIONS, with_docs=True, model="fake-model")

    # Should have been called exactly once (all questions grouped together)
    assert client.models.generate_content.call_count == 1  # type: ignore
    call = client.models.generate_content.call_args_list[0]  # type: ignore
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
        client.models.generate_content.return_value = make_fake_multi_response(  # type: ignore
            [q.answer for q in QUESTIONS]
        )

        evaluate_llm(QUESTIONS, with_docs=False, model="fake-model")

    call = client.models.generate_content.call_args_list[0]  # type: ignore
    config: types.GenerateContentConfig = call.kwargs["config"]
    tools = config.tools or []
    assert not any(isinstance(t, types.Tool) and t.url_context is not None for t in tools), (
        "Did not expect a UrlContext tool when with_docs=False"
    )


# ---------------------------------------------------------------------------
# make_multi_prompt tests
# ---------------------------------------------------------------------------


def test_make_multi_prompt_single_question():
    """For a single question without docs, make_multi_prompt returns a valid prompt."""
    q = QUESTIONS[0]
    result = make_multi_prompt([q], docs_urls=None)
    assert q.question in result
    assert "Select the correct answer" in result


def test_make_multi_prompt_multiple_questions():
    """For multiple questions, each question is numbered in the prompt."""
    result = make_multi_prompt(QUESTIONS[:2], docs_urls=None)
    assert "Question 1:" in result
    assert "Question 2:" in result
    assert QUESTIONS[0].question in result
    assert QUESTIONS[1].question in result


def test_make_multi_prompt_with_docs():
    """Documentation URLs appear at the top of the prompt when provided."""
    url = "https://docs.example.com"
    result = make_multi_prompt(QUESTIONS[:2], docs_urls=[url])
    assert url in result
    # URL should only appear once
    assert result.count(url) == 1


# ---------------------------------------------------------------------------
# get_agent_answers_for_group tests
# ---------------------------------------------------------------------------


def test_get_agent_answers_for_group_single_question():
    """Single-question group returns one response per candidate (default: 1)."""
    q = QUESTIONS[0]
    result = get_agent_answers_for_group(
        mock_client("A"), "fake-model", tools=None, questions=[q], docs_urls=None
    )
    assert len(result) == 1  # one candidate
    assert len(result[0]) == 1  # one question
    assert isinstance(result[0][0], Response)
    assert result[0][0].answer == "A"


def test_get_agent_answers_for_group_multi_question():
    """Multi-question group returns one MultiResponse parsed into separate Responses."""
    answers: list[Letter] = ["A", "B", "C"]
    result = get_agent_answers_for_group(
        mock_client_multi(answers),
        "fake-model",
        tools=None,
        questions=QUESTIONS,
        docs_urls=None,
    )
    assert len(result) == 1  # one candidate
    assert len(result[0]) == 3  # three questions
    for i, letter in enumerate(answers):
        assert result[0][i] is not None
        assert result[0][i].answer == letter  # type: ignore[union-attr]


def test_get_agent_answers_for_group_multiple_candidates():
    """When multiple candidates are returned, all are parsed."""
    # Build a response with two candidates
    answers_c1: list[Letter] = ["A", "B", "C"]
    answers_c2: list[Letter] = ["B", "B", "C"]

    resp_c1 = MultiResponse(
        answers=[Response(answer=a, justification="j") for a in answers_c1]
    )
    resp_c2 = MultiResponse(
        answers=[Response(answer=a, justification="j") for a in answers_c2]
    )

    def make_candidate(resp: MultiResponse, idx: int) -> types.Candidate:
        return types.Candidate(
            index=idx,
            content=types.Content(
                role="model", parts=[types.Part(text=resp.model_dump_json())]
            ),
            grounding_metadata=None,
            url_context_metadata=None,
        )

    api_response = MagicMock()
    api_response.candidates = [make_candidate(resp_c1, 0), make_candidate(resp_c2, 1)]
    api_response.parsed = None
    api_response.text = resp_c1.model_dump_json()

    client = MagicMock()
    client.models.generate_content.return_value = api_response

    result = get_agent_answers_for_group(
        client,
        "fake-model",
        tools=None,
        questions=QUESTIONS,
        docs_urls=None,
        num_candidates=2,
    )
    assert len(result) == 2  # two candidates
    for cand in result:
        assert len(cand) == 3  # three questions per candidate
    # Candidate 1 answers
    assert result[0][0].answer == "A"  # type: ignore[union-attr]
    assert result[0][1].answer == "B"  # type: ignore[union-attr]
    # Candidate 2 answers
    assert result[1][0].answer == "B"  # type: ignore[union-attr]
    assert result[1][1].answer == "B"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# EvaluationResult stats tests
# ---------------------------------------------------------------------------


def test_evaluation_result_score_with_per_question_scores():
    """score property uses mean of per_question_scores when available."""
    result = EvaluationResult(
        num_questions=3,
        correct_answers=2,
        invalid_answers=0,
        per_question_scores=(1.0, 1.0, 0.0),
        num_candidates=1,
    )
    assert result.score == pytest.approx(2 / 3, rel=1e-6)


def test_evaluation_result_score_std():
    """score_std returns stdev of per_question_scores."""
    import statistics as _stats

    scores = (1.0, 0.0, 1.0)
    result = EvaluationResult(
        num_questions=3,
        correct_answers=2,
        invalid_answers=0,
        per_question_scores=scores,
        num_candidates=1,
    )
    assert result.score_std == pytest.approx(_stats.stdev(scores), rel=1e-6)


def test_evaluation_result_score_std_single():
    """score_std is 0 when only one question."""
    result = EvaluationResult(
        num_questions=1,
        correct_answers=1,
        invalid_answers=0,
        per_question_scores=(1.0,),
        num_candidates=1,
    )
    assert result.score_std == 0.0


def test_evaluate_llm_batches_all_questions_in_one_call():
    """All questions are batched into a single API call."""
    answers: list[Letter] = ["A", "B", "A"]

    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.return_value = make_fake_multi_response(answers)

        result = evaluate_llm(
            QUESTIONS[:3], with_docs=False, model="fake-model", docs_urls=None
        )

    # All 3 questions batched → 1 API call
    assert client.models.generate_content.call_count == 1
    assert result.num_questions == 3
    assert len(result.per_question_scores) == 3


def test_evaluate_llm_passes_docs_urls_to_api():
    """docs_urls passed to evaluate_llm are forwarded to the API prompt."""
    url = "https://docs.example.com"

    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.return_value = make_fake_multi_response(
            [q.answer for q in QUESTIONS]
        )

        evaluate_llm(QUESTIONS, with_docs=True, model="fake-model", docs_urls=[url])

    call = client.models.generate_content.call_args_list[0]
    prompt = call.kwargs["contents"]
    assert url in prompt


def test_evaluate_llm_num_candidates_dummy():
    """num_candidates > 1 works with the dummy model (no API calls)."""
    result = evaluate_llm(QUESTIONS, with_docs=False, model=DUMMY_MODEL, num_candidates=3)
    assert result.num_questions == len(QUESTIONS)
    assert result.num_candidates == 3
    assert len(result.per_question_scores) == len(QUESTIONS)
    # Each score should be between 0 and 1
    for score in result.per_question_scores:
        assert 0.0 <= score <= 1.0


def test_evaluate_llm_per_question_scores_populated():
    """evaluate_llm populates per_question_scores for the real model path."""
    with patch("docmetrics.main.get_google_genai_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client
        client.models.generate_content.return_value = make_fake_multi_response(
            [q.answer for q in QUESTIONS]
        )

        result = evaluate_llm(QUESTIONS, with_docs=False, model="fake-model")

    assert len(result.per_question_scores) == len(QUESTIONS)
    # All answers are correct, so each score should be 1.0
    assert all(s == 1.0 for s in result.per_question_scores)
    assert result.score == pytest.approx(1.0)
    assert result.score_std == pytest.approx(0.0)
