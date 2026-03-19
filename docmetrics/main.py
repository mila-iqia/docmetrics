import argparse
import dataclasses
import functools
import json
import logging
import random
import re
import statistics
import warnings
from pathlib import Path
from typing import Callable, Literal

import pydantic
import rich.logging
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

Letter = Literal["A", "B", "C", "D", "E"]

DUMMY_MODEL = "test:dummy"
"""A model name that returns a random answer without calling any external API."""
__all__ = [
    "evaluate_llm",
    "ask_question",
    "load_questions",
    "Question",
    "Response",
    "EvaluationResult",
]


@dataclass(frozen=True)
class Question:
    question: str
    """The question to ask the LLM."""

    options: dict[Letter, str]
    """A list of possible answers to the question."""

    answer: Letter
    """The correct answer to the question (must be one of the letters in `options`)."""

    docs_urls: list[str] | None = None
    """A list of URLs to relevant documentation pages.

    The hope is that by reading these pages, the LLM can answer the question more accurately. Note
    that the LLMs are not going to look at links within these pages, only the content of these
    pages itself. For example, giving a link to docs.mila.quebec wouldn't be very helpful.
    """

    def __postinit__(self):
        assert self.answer in self.options, "correct answer isn't in the options!"


class Response(pydantic.BaseModel):
    answer: Letter
    """The selected answer."""

    # TODO: Check if adding this 'justification' is actually helpful, and whether it increases costs.
    # Seems like it might just be extra tokens to generate, for our purposes.
    justification: str = ""
    """A brief justification for the selected answer."""


class MultiResponse(pydantic.BaseModel):
    answers: list[Response]
    """Answers to each question in the prompt, in the same order."""


@functools.cache
def get_google_genai_client():
    load_dotenv()  # reads variables from a .env file and sets them in os.environ
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    return genai.Client()


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    num_questions: int
    correct_answers: int
    invalid_answers: int
    per_question_scores: tuple[float, ...] = ()
    """Per-question correctness fraction averaged over candidates (0.0–1.0 per question)."""
    num_candidates: int = 1
    """Number of response candidates generated per question group."""

    @property
    def score(self) -> float:
        """The mean fraction of questions the LLM answered correctly (averaged over candidates)."""
        if self.per_question_scores:
            return statistics.mean(self.per_question_scores)
        return self.correct_answers / self.num_questions if self.num_questions > 0 else 0.0

    @property
    def score_std(self) -> float:
        """Standard deviation of per-question correctness scores across questions and candidates."""
        if len(self.per_question_scores) > 1:
            return statistics.stdev(self.per_question_scores)
        return 0.0


def evaluate_llm(
    questions: list[Question],
    with_docs: bool,
    model: str,
    num_candidates: int = 1,
    # tools: Sequence[types.Tool | Callable] | None = None,
) -> EvaluationResult:
    """Evaluates an LLM on some questions with/without documentation as context.

    Questions are grouped by their ``docs_urls`` so that questions sharing the same
    documentation context are asked together in a single API call, reducing the total
    number of requests.  When *num_candidates* > 1 the API is asked to return multiple
    independent answers per group, enabling mean/std statistics.

    Parameters
    ----------
    questions: The list of questions to ask the LLM.
    with_docs: Whether to provide documentation URLs as context to the LLM.
    model: The name of the LLM model to use.
    num_candidates: Number of independent response candidates to request per API call.
        Values > 1 enable mean/std statistics but are not compatible with structured
        output; the fallback text parser is used instead.

    Returns
    -------
    The evaluation results, including per-question scores and standard deviation.


    Notes
    - Could also input a list of tools to give to the agent, in addition to URL search.
    """
    client = None if model == DUMMY_MODEL else get_google_genai_client()
    num_questions = len(questions)

    if num_questions == 0:
        return EvaluationResult(
            num_questions=0,
            correct_answers=0,
            invalid_answers=0,
            per_question_scores=(),
            num_candidates=num_candidates,
        )

    # per_question_scores[i] = fraction of candidates that correctly answered question i
    per_question_scores = [0.0] * num_questions
    invalid_answers = 0

    if model == DUMMY_MODEL:
        for i, question in enumerate(questions):
            candidate_scores = [
                float(random.choice(list(question.options.keys())) == question.answer)
                for _ in range(num_candidates)
            ]
            per_question_scores[i] = statistics.mean(candidate_scores)
        correct_answers = sum(1 for s in per_question_scores if s >= 0.5)
        return EvaluationResult(
            num_questions=num_questions,
            correct_answers=correct_answers,
            invalid_answers=0,
            per_question_scores=tuple(per_question_scores),
            num_candidates=num_candidates,
        )

    assert client is not None
    tools = [types.Tool(url_context=types.UrlContext())] if with_docs else None

    # Group questions by their docs URLs to reduce the number of API calls.
    # Questions sharing the same docs context are asked together in one request.
    for docs_key, indexed_questions in _group_questions_by_docs(questions):
        qs = [q for _, q in indexed_questions]
        logger.info(
            f"Asking {len(qs)} question(s) in one API call "
            f"(docs context: {list(docs_key) or 'none'}, candidates: {num_candidates})."
        )

        # One API call for all questions in this group, with num_candidates candidates
        candidate_responses = get_agent_answers_for_group(
            client=client,
            model=model,
            tools=tools,
            questions=qs,
            with_docs=with_docs,
            num_candidates=num_candidates,
        )
        # candidate_responses shape: [num_actual_candidates][len(qs)] -> Response | None

        for q_pos, (orig_idx, question) in enumerate(indexed_questions):
            scores_for_q: list[float] = []
            all_invalid = True

            for cand_responses in candidate_responses:
                resp = cand_responses[q_pos] if q_pos < len(cand_responses) else None
                if resp is not None:
                    all_invalid = False
                    scores_for_q.append(float(resp.answer == question.answer))
                    logger.info(
                        f"Question {orig_idx + 1}: correct={question.answer}, LLM={resp.answer}"
                    )
                    if resp.justification:
                        logger.debug(f"Justification: {resp.justification}")

            if all_invalid:
                logger.error(
                    f"Question {orig_idx + 1}: all candidate answers couldn't be parsed!"
                )
                invalid_answers += 1
                per_question_scores[orig_idx] = 0.0
            else:
                per_question_scores[orig_idx] = statistics.mean(scores_for_q)

    # A question is counted as "correct" when at least half the candidates answered it correctly.
    correct_answers = sum(1 for s in per_question_scores if s >= 0.5)

    return EvaluationResult(
        num_questions=num_questions,
        correct_answers=correct_answers,
        invalid_answers=invalid_answers,
        per_question_scores=tuple(per_question_scores),
        num_candidates=num_candidates,
    )


def load_questions(questions_path: Path) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(questions_path.read_text())]


def _group_questions_by_docs(
    questions: list[Question],
) -> list[tuple[tuple[str, ...], list[tuple[int, Question]]]]:
    """Group questions by their ``docs_urls`` to minimise the number of API calls.

    Questions with the same documentation context can be asked together in a single
    request.  The original question indices are preserved so that results can be
    mapped back to the right positions in the output.

    Returns a list of ``(docs_key, [(original_index, question), ...])`` pairs,
    maintaining insertion order.
    """
    groups: dict[tuple[str, ...], list[tuple[int, Question]]] = {}
    for idx, question in enumerate(questions):
        key = tuple(question.docs_urls or [])
        if key not in groups:
            groups[key] = []
        groups[key].append((idx, question))
    return list(groups.items())


def make_multi_prompt(questions: list[Question], with_docs: bool) -> str:
    """Build a prompt for answering one or more questions at once.

    For a single question the output is identical to ``make_prompt``.  For multiple
    questions each question is numbered and the LLM is asked to answer all of them.
    The shared documentation URLs (if any) appear once at the top of the prompt.
    """
    if len(questions) == 1:
        return make_prompt(questions[0], with_docs=with_docs)

    parts: list[str] = []

    if with_docs:
        # Collect all unique URLs from the group (they share the same key, but union
        # is safer in case the list was constructed from heterogeneous sources).
        seen: list[str] = []
        for q in questions:
            for url in q.docs_urls or []:
                if url not in seen:
                    seen.append(url)
        if seen:
            parts.append("Based on this documentation: " + ", ".join(seen) + ",")

    parts.append(
        f"Answer ALL {len(questions)} of the following multiple-choice questions. "
        "Provide your answers in order."
    )

    for i, q in enumerate(questions, 1):
        parts.append(f"\nQuestion {i}: {q.question}")
        for letter, answer in q.options.items():
            parts.append(f"- {letter}: {answer}")

    return "\n".join(parts) + "\n"


def get_agent_answers_for_group(
    client: genai.Client,
    model: str,
    tools: list[types.Tool | Callable] | None,
    questions: list[Question],
    with_docs: bool,
    num_candidates: int = 1,
) -> list[list[Response | None]]:
    """Ask a group of questions in a single API call, optionally with multiple candidates.

    Batching questions that share the same documentation context reduces the total
    number of API calls.  Requesting *num_candidates* > 1 returns independent answers
    from the same call and enables mean/std statistics.

    Note: structured output (``response_json_schema``) is only used when
    *num_candidates* == 1, because the Gemini API does not support both at the same
    time.  For *num_candidates* > 1 the text output is parsed with the normal JSON
    fallback logic.

    Returns
    -------
    A list of shape ``[num_actual_candidates][num_questions]`` → ``Response | None``.
    """
    is_single = len(questions) == 1
    prompt = make_multi_prompt(questions, with_docs=with_docs)
    logger.debug(f"Prompt sent to LLM: [magenta]{prompt}")

    response_schema = (
        Response.model_json_schema() if is_single else MultiResponse.model_json_schema()
    )

    # Structured output is only reliable for gemini-2.5-flash without tools, and is
    # incompatible with candidate_count > 1.
    use_structured = not tools and model == "gemini-2.5-flash"
    use_schema_in_request = use_structured and num_candidates == 1

    api_response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            candidate_count=num_candidates if num_candidates > 1 else None,
            response_mime_type="application/json" if use_schema_in_request else None,
            response_json_schema=response_schema if use_schema_in_request else None,
            tools=tools,
        ),
    )
    assert api_response.candidates

    results: list[list[Response | None]] = []

    for candidate_idx, candidate in enumerate(api_response.candidates):
        # Log URL consultation metadata
        if (consulted_urls := candidate.url_context_metadata) and consulted_urls.url_metadata:
            logger.info(
                f"The LLM consulted {len(consulted_urls.url_metadata)} web pages to answer "
                f"the question(s) (candidate {candidate_idx})."
            )
            logger.debug(
                "Consulted URLs: "
                + "\n".join(
                    "- "
                    + (url_meta.retrieved_url or "n/a")
                    + " "
                    + (
                        "(success)"
                        if url_meta.url_retrieval_status
                        == types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS
                        else "(failure)"
                    )
                    for url_meta in consulted_urls.url_metadata
                )
            )

        # Extract candidate text from content parts
        candidate_text: str | None = None
        if candidate.content and candidate.content.parts:
            texts = [p.text for p in candidate.content.parts if p.text]
            if texts:
                candidate_text = "".join(texts)

        if is_single:
            # --- single-question response ---
            parsed: Response | None = None
            # For the first candidate with structured output, try api_response.parsed first
            if candidate_idx == 0 and use_schema_in_request and isinstance(
                api_response.parsed, Response
            ):
                logger.debug("LLM output was correctly parsed by the client library.")
                parsed = api_response.parsed
            elif candidate_text:
                try:
                    parsed = Response.model_validate_json(candidate_text)
                except pydantic.ValidationError:
                    parsed = parse_response_fallback(candidate_text)
            if parsed is None:
                logger.error(f"LLM answer for candidate {candidate_idx} couldn't be parsed!")
            results.append([parsed])

        else:
            # --- multi-question response ---
            parsed_responses: list[Response | None] = [None] * len(questions)
            if candidate_text:
                try:
                    multi = MultiResponse.model_validate_json(candidate_text)
                    n = min(len(multi.answers), len(questions))
                    if len(multi.answers) != len(questions):
                        logger.warning(
                            f"Expected {len(questions)} answers from LLM, "
                            f"got {len(multi.answers)} (candidate {candidate_idx})."
                        )
                    for i in range(n):
                        parsed_responses[i] = multi.answers[i]
                except pydantic.ValidationError:
                    logger.error(
                        f"Failed to parse multi-question response for candidate "
                        f"{candidate_idx}: {candidate_text[:200]}"
                    )
            results.append(parsed_responses)

    return results


def ask_question(
    client: genai.Client | None,
    question: Question,
    with_docs: bool,
    model: str,
    tools: list[types.Tool | Callable] | None,
) -> bool | None:
    """Asks a question to the LLM and returns whether the LLM answered correctly.

    Returns None if the LLM's answer was invalid.
    """
    if model == DUMMY_MODEL:
        dummy_answer = random.choice(list(question.options.keys()))
        logger.info(f"Correct answer: {question.answer}, dummy answer: {dummy_answer}")
        return dummy_answer == question.answer

    # TODO: For a lot of the models available through Google AI Studio, they can't
    # use tools to fetch the docs content. It *might* be worthwhile to actually fetch,
    # parse, and embed the page into the prompt ourselves for those models to get results with mode models.
    # OR, we could switch to something like VertexAI and see if we have access to more models with tool use
    # there.

    assert client is not None
    prompt = make_prompt(question, with_docs=with_docs)
    logger.debug(f"Prompt sent to LLM: [magenta]{prompt}")
    # TODO: use https://ai.google.dev/api/batch-api instead of single requests.
    agent_answer = get_agent_answer(client, model, tools, prompt)
    if not agent_answer:
        logger.error("LLM's answer couldn't be parsed!")
        return None
    # correct_answer = question.options[question.answer]
    logger.info(f"Correct answer: {question.answer}, LLM's answer: {agent_answer.answer}")
    if agent_answer.justification:
        logger.debug(f"LLM's justification: {agent_answer.justification}")
    return agent_answer.answer == question.answer


def get_agent_answer(
    client: genai.Client,
    model: str,
    tools: list[types.Tool | Callable] | None,
    prompt: str,
) -> Response | None:
    api_response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            # This response_mime_type can only be set if using the gemini-3-pro model.
            # https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools
            response_mime_type=(
                "application/json" if not tools and model == "gemini-2.5-flash" else None
            ),
            response_json_schema=Response.model_json_schema(),
            tools=tools,
        ),
    )
    assert api_response.candidates

    if len(api_response.candidates) > 1:
        warnings.warn(
            f"Response contained {len(api_response.candidates)} candidates, but only the first one will be used!"
        )
    for candidate in api_response.candidates:
        # todo: look into this maybe?
        # logger.debug(
        #     f"Grounding metadata for candidate {candidate.index}: {candidate.grounding_metadata}"
        # )
        if (consulted_urls := candidate.url_context_metadata) and consulted_urls.url_metadata:
            logger.info(
                f"The LLM consulted {len(consulted_urls.url_metadata)} web pages to answer the question."
            )
            logger.debug(
                "Consulted URLs: "
                + "\n".join(
                    "- "
                    + (url_meta.retrieved_url or "n/a")
                    + " "
                    + (
                        "(success)"
                        if url_meta.url_retrieval_status
                        == types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS
                        else "(failure)"
                    )
                    for url_meta in consulted_urls.url_metadata
                )
            )

    if isinstance(api_response.parsed, Response):
        logger.debug("LLM output was correctly parsed by the client library.")
        return api_response.parsed

    assert api_response.text is not None
    try:
        return Response.model_validate_json(api_response.text)
    except pydantic.ValidationError:
        return parse_response_fallback(api_response.text)


def parse_response_fallback(response_str: str) -> Response | None:
    """Parses a `Response` object from the API output when the LLM doesn't follow the requested response schema.

    >>> parse_response_fallback('A')
    Response(answer='A', justification='')

    >>> parse_response_fallback('This description perfectly matches the requirement for storing "temporary model checkpoints". Answer: A')
    Response(answer='A', justification='This description perfectly matches the requirement for storing "temporary model checkpoints". Answer:')
    """
    try:
        last_line = response_str.strip().splitlines()[-1].strip().removesuffix(".")
        # todo: if the last word is a single letter (capitalised or not), use this as the LLM's guess.
        last_char = re.search(r"\b([A-Ea-e])\b", last_line)
        if not last_char:
            logger.error(f"Invalid response: {response_str}")
            return None
        last_char = last_char.group(0)
        last_char = last_char.upper()
        assert last_char in ("A", "B", "C", "D", "E")
        return Response(
            answer=last_char,
            justification=response_str.strip().removesuffix(last_char).strip(),
        )
    except ValueError:
        logger.error(f"Invalid response: {response_str}")
        return None


def make_prompt(question: Question, with_docs: bool) -> str:
    return (
        (
            ("Based on this documentation: " + ", ".join(question.docs_urls) + ",\n")
            if with_docs and question.docs_urls
            else ""
        )
        # + ((context + "\n\n") if context else "")
        + "Select the correct answer to the following question:\n"
        + f"{question.question}\n"
        + "\n".join(f"- {letter}: {answer}" for letter, answer in question.options.items())
        + "\n"
    )


def _add_evaluate_args(p: argparse.ArgumentParser, questions_required: bool = True) -> None:
    p.add_argument("--questions", type=Path, required=questions_required)
    p.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help='LLM model to use (e.g. "gemini-2.5-flash"). Use "test:dummy" for random answers without any API calls.',
    )
    p.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format. 'json' emits a machine-readable JSON object suitable for CI pipelines.",
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help=(
            "Number of independent response candidates to generate per question group. "
            "Values > 1 enable mean/std statistics across candidates. "
            "Note: candidate_count > 1 is not compatible with structured output; "
            "the fallback text parser is used instead."
        ),
    )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument("-v", "--verbose", action="count")
    # Evaluate args on the top-level parser so `docmetrics` (no subcommand) works.
    _add_evaluate_args(parser, questions_required=False)

    subparsers = parser.add_subparsers(dest="subcommand")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        formatter_class=argparse.MetavarTypeHelpFormatter,
        help="Evaluate an LLM on documentation questions (default when no subcommand is given).",
    )
    evaluate_parser.add_argument("-v", "--verbose", action="count")
    _add_evaluate_args(evaluate_parser)

    quiz_parser = subparsers.add_parser(
        "quiz",
        formatter_class=argparse.MetavarTypeHelpFormatter,
        help="Take the quiz interactively in the terminal.",
    )
    quiz_parser.add_argument("-v", "--verbose", action="count")
    quiz_parser.add_argument("--questions", type=Path, required=True)

    args = parser.parse_args()
    verbose: int = args.verbose or 0

    logging.basicConfig(
        level=logging.DEBUG if verbose >= 3 else logging.INFO if verbose == 2 else logging.WARNING,
        handlers=[rich.logging.RichHandler(markup=True)],
        format="%(message)s",
    )
    logger.setLevel(  # this logger specifically.
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    )

    if args.subcommand == "quiz":
        from docmetrics.quiz import run_quiz

        questions = load_questions(questions_path=args.questions)
        run_quiz(questions)
        return

    # Default: evaluate (subcommand is None or "evaluate")
    if args.questions is None:
        parser.error("the following arguments are required: --questions")

    questions = load_questions(questions_path=args.questions)
    model: str = args.model
    num_candidates: int = args.num_candidates
    score_with_no_context = evaluate_llm(
        questions, with_docs=False, model=model, num_candidates=num_candidates
    )
    score_with_mila_docs_urls = evaluate_llm(
        questions, with_docs=True, model=model, num_candidates=num_candidates
    )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "without_docs": {
                        "num_questions": score_with_no_context.num_questions,
                        "correct_answers": score_with_no_context.correct_answers,
                        "invalid_answers": score_with_no_context.invalid_answers,
                        "score": score_with_no_context.score,
                        "score_std": score_with_no_context.score_std,
                        "num_candidates": score_with_no_context.num_candidates,
                    },
                    "with_docs": {
                        "num_questions": score_with_mila_docs_urls.num_questions,
                        "correct_answers": score_with_mila_docs_urls.correct_answers,
                        "invalid_answers": score_with_mila_docs_urls.invalid_answers,
                        "score": score_with_mila_docs_urls.score,
                        "score_std": score_with_mila_docs_urls.score_std,
                        "num_candidates": score_with_mila_docs_urls.num_candidates,
                    },
                }
            )
        )
    else:
        print(f"{score_with_no_context=}")
        print(f"{score_with_mila_docs_urls=}")


if __name__ == "__main__":
    main()
