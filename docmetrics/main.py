import argparse
import dataclasses
import functools
import json
import logging
import math
import os
import random
import re
import warnings
from pathlib import Path
from typing import Callable, Literal

import httpx
import pydantic
import rich.console
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
OLLAMA_PREFIX = "ollama:"
"""Prefix for Ollama model names, e.g. 'ollama:qwen3-coder-next:latest'."""
MAX_TOOL_CALLS = 5
"""Maximum number of tool calls allowed per question before giving up."""
OLLAMA_DEFAULT_URL = "http://localhost:11434"
"""Default base URL for the Ollama server."""
__all__ = [
    "evaluate_llm",
    "ask_question",
    "load_questions",
    "Question",
    "Response",
    "QuestionResult",
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

    def __postinit__(self):
        assert self.answer in self.options, "correct answer isn't in the options!"


class Response(pydantic.BaseModel):
    answer: Letter
    """The selected answer."""

    # TODO: Check if adding this 'justification' is actually helpful, and whether it increases costs.
    # Seems like it might just be extra tokens to generate, for our purposes.
    justification: str = ""
    """A brief justification for the selected answer."""


@dataclasses.dataclass(frozen=True)
class QuestionResult:
    expected: Letter
    """The correct answer letter."""

    runs: tuple[Letter | None, ...]
    """One selected letter per candidate run (None = unparsable response)."""

    @property
    def correct_count(self) -> int:
        return sum(1 for r in self.runs if r == self.expected)

    @property
    def pass_rate(self) -> float:
        return self.correct_count / len(self.runs) if self.runs else 0.0


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    answers: tuple[QuestionResult, ...]
    """Per-question results."""

    num_candidates: int = 1
    """Number of candidate answers requested per question."""

    @property
    def num_questions(self) -> int:
        return len(self.answers)

    @property
    def correct_answers(self) -> int:
        return sum(r.correct_count for r in self.answers)

    @property
    def invalid_answers(self) -> int:
        return sum(1 for r in self.answers for run in r.runs if run is None)

    @property
    def score(self) -> float:
        """Mean per-question pass rate."""
        if not self.answers:
            return 0.0
        return sum(r.pass_rate for r in self.answers) / len(self.answers)

    @property
    def score_std(self) -> float:
        """Population std-dev of per-question pass rates."""
        if not self.answers:
            return 0.0
        rates = [r.pass_rate for r in self.answers]
        mean = sum(rates) / len(rates)
        variance = sum((r - mean) ** 2 for r in rates) / len(rates)
        return math.sqrt(variance)


@functools.cache
def get_google_genai_client():
    load_dotenv()  # reads variables from a .env file and sets them in os.environ
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    return genai.Client()


def _is_ollama_model(model: str) -> bool:
    return model.startswith(OLLAMA_PREFIX)


def _ollama_model_name(model: str) -> str:
    """Strips the 'ollama:' prefix to get the bare model name.

    >>> _ollama_model_name("ollama:qwen3-coder-next:latest")
    'qwen3-coder-next:latest'
    """
    return model.removeprefix(OLLAMA_PREFIX)


@functools.cache
def _get_ollama_client(base_url: str = OLLAMA_DEFAULT_URL):
    load_dotenv()  # read the OLLAMA_API_KEY from the .env file if present.
    import ollama

    return ollama.Client(
        host=base_url,
        headers={"Authorization": "Bearer " + OLLAMA_API_KEY}
        if (OLLAMA_API_KEY := os.environ.get("OLLAMA_API_KEY"))
        else None,
    )


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse

    return urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")


def _is_allowed_docs_url(url: str, docs_urls: list[str]) -> bool:
    """Returns True if `url` is under at least one of the given docs base URLs."""
    for base in docs_urls:
        base = base.rstrip("/")
        if url == base or url.startswith(base + "/"):
            return True
    return False


def _fetch_url(url: str) -> str:
    """Fetch URL content via httpx, stripping HTML tags if the response is HTML."""
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()
    if "html" not in response.headers.get("content-type", ""):
        return response.text
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip and (stripped := data.strip()):
                self._parts.append(stripped)

    ex = _Extractor()
    ex.feed(response.text)
    return "\n".join(ex._parts)


def _get_agent_answer_ollama(
    model: str,
    prompt: str,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    use_web_fetch: bool = False,
    docs_urls: list[str] | None = None,
) -> "Response | None":
    client = _get_ollama_client(ollama_url)
    ollama_model = _ollama_model_name(model)
    schema = Response.model_json_schema()
    full_prompt = (
        prompt
        + f"\nRespond with a JSON object matching this schema: {json.dumps(schema)}\n"
        + 'Example: {"answer": "A", "justification": "Because..."}'
    )

    messages: list = [{"role": "user", "content": full_prompt}]
    tools = [client.web_fetch] if use_web_fetch else None

    import ollama as _ollama

    tool_call_count = 0
    while True:
        try:
            response = client.chat(
                model=ollama_model,
                messages=messages,
                tools=tools,
                format="json" if not tools else None,
            )
        except _ollama.ResponseError as e:
            logger.warning(f"Ollama returned an error (invalid tool call?): {e}")
            return None
        messages.append(response.message)

        if not response.message.tool_calls:
            content = response.message.content
            if not content:
                return None
            try:
                return Response.model_validate_json(content)
            except pydantic.ValidationError:
                return parse_response_fallback(content)

        tool_call_count += len(response.message.tool_calls)
        if tool_call_count > MAX_TOOL_CALLS:
            logger.warning(
                f"Exceeded maximum tool calls ({MAX_TOOL_CALLS}), asking for a final answer."
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of tool calls. "
                        "Based on the information gathered so far, please provide your final answer now. "
                        f"Respond with a JSON object matching this schema: {json.dumps(schema)}\n"
                        'Example: {"answer": "A", "justification": "Because..."}'
                    ),
                }
            )
            try:
                final_response = client.chat(
                    model=ollama_model,
                    messages=messages,
                    format="json",
                )
            except _ollama.ResponseError as e:
                logger.warning(f"Ollama returned an error on final answer request: {e}")
                return None
            content = final_response.message.content
            if not content:
                return None
            try:
                return Response.model_validate_json(content)
            except pydantic.ValidationError:
                return parse_response_fallback(content)

        for tool_call in response.message.tool_calls:
            if tool_call.function.name == "web_fetch":
                url = tool_call.function.arguments.get("url", "")
                if docs_urls and not _is_allowed_docs_url(url, docs_urls):
                    logger.warning(f"LLM requested disallowed URL (blocked): {url}")
                    content = (
                        f"Access denied: {url!r} is not under the allowed documentation URLs."
                    )
                else:
                    logger.info(f"LLM fetching documentation URL: {url}")
                    use_ollama_fetch = bool(
                        os.environ.get("OLLAMA_API_KEY")
                    ) and not _is_local_url(url)
                    try:
                        if use_ollama_fetch:
                            content = str(client.web_fetch(**tool_call.function.arguments))
                        else:
                            content = _fetch_url(url)
                    except Exception as e:
                        logger.warning(f"web_fetch failed for {url}: {e}")
                        content = f"Error fetching URL: {e}"
            else:
                logger.warning(f"LLM requested unknown tool: {tool_call.function.name}")
                content = f"Tool {tool_call.function.name} not found"
            messages.append(
                {"role": "tool", "content": content, "tool_name": tool_call.function.name}
            )


def evaluate_llm(
    questions: list[Question],
    with_docs: bool,
    model: str,
    docs_urls: list[str] | None = None,
    docs_files: list[Path] | None = None,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    num_candidates: int = 1,
    # tools: Sequence[types.Tool | Callable] | None = None,
) -> EvaluationResult:
    """Evaluates an LLM on some questions with/without documentation as context.

    Parameters
    ----------
    questions: The list of questions to ask the LLM.
    with_docs: Whether to provide documentation URLs as context to the LLM.
    model: The name of the LLM model to use.
    docs_urls: A list of URLs to relevant documentation pages to provide as context.
    docs_files: A list of local file paths whose content will be inlined into the prompt.
    ollama_url: Base URL of the Ollama server (only used when model has the 'ollama:' prefix).
    num_candidates: Number of times to ask each question. Scores are averaged across candidates.

    Returns
    -------
    The evaluation results.


    Notes
    - Could also input a list of tools to give to the agent, in addition to URL search.
    """
    client = None if model == DUMMY_MODEL or _is_ollama_model(model) else get_google_genai_client()
    num_questions = len(questions)

    docs_content: str | None = None
    if docs_files:
        docs_content = "\n---\n".join(f.read_text() for f in docs_files)
    elif (
        with_docs
        and docs_urls
        and not _is_ollama_model(model)
        and any(_is_local_url(u) for u in docs_urls)
    ):
        # Google's url_context tool runs server-side and can't reach localhost URLs; pre-fetch them.
        logger.info(f"Pre-fetching {len(docs_urls)} documentation URL(s) (localhost detected)...")
        docs_content = "\n---\n".join(_fetch_url(u) for u in docs_urls)

    # TODO: Group questions based on the docs pages they require and use the batch API
    # to ask multiple questions at once with the same context.

    answer_results: list[QuestionResult] = []
    for question_index, question in enumerate(questions, 1):
        runs: list[Letter | None] = []
        for candidate_index in range(num_candidates):
            selected = ask_question(
                client=client,
                question=question,
                question_index=question_index,
                num_questions=num_questions,
                candidate_index=candidate_index if num_candidates > 1 else None,
                num_candidates=num_candidates if num_candidates > 1 else None,
                with_docs=with_docs,
                model=model,
                docs_urls=docs_urls,
                docs_content=docs_content,
                ollama_url=ollama_url,
                # Adding this gives the LLM the ability to consult URLs given in the prompt.
                # Not needed (or wanted) when docs are already inlined or for Ollama models.
                tools=(
                    [types.Tool(url_context=types.UrlContext())]
                    if with_docs and docs_content is None and not _is_ollama_model(model)
                    else None
                ),
            )
            runs.append(selected)
        answer_results.append(QuestionResult(expected=question.answer, runs=tuple(runs)))

    return EvaluationResult(answers=tuple(answer_results), num_candidates=num_candidates)


def load_questions(questions_path: Path) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(questions_path.read_text())]


def ask_question(
    client: genai.Client | None,
    question: Question,
    with_docs: bool,
    model: str,
    docs_urls: list[str] | None,
    tools: list[types.Tool | Callable] | None,
    docs_content: str | None = None,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    question_index: int | None = None,
    num_questions: int | None = None,
    candidate_index: int | None = None,
    num_candidates: int | None = None,
) -> Letter | None:
    """Asks a question to the LLM and returns the selected answer letter, or None if unparsable."""
    q_prefix = (
        f"Question {question_index}/{num_questions}: "
        if question_index is not None and num_questions is not None
        else ""
    )
    if candidate_index is not None and num_candidates is not None:
        q_prefix += f"[candidate {candidate_index + 1}/{num_candidates}] "

    if model == DUMMY_MODEL:
        dummy_answer = random.choice(list(question.options.keys()))
        logger.info(f"{q_prefix}Correct answer: {question.answer}, dummy answer: {dummy_answer}")
        return dummy_answer

    if _is_ollama_model(model):
        prompt = make_prompt(
            question, with_docs=with_docs, docs_urls=docs_urls, docs_content=docs_content
        )
        logger.debug(f"{q_prefix}Prompt sent to LLM: {prompt}")
        # Use web_fetch tool when docs are given as URLs (not already inlined via --docs-file).
        use_web_fetch = with_docs and bool(docs_urls) and docs_content is None
        agent_answer = _get_agent_answer_ollama(
            model, prompt, ollama_url, use_web_fetch=use_web_fetch, docs_urls=docs_urls
        )
        if not agent_answer:
            logger.error(f"{q_prefix}LLM's answer couldn't be parsed!")
            return None
        logger.info(
            f"{q_prefix}Correct answer: {question.answer}, LLM's answer: {agent_answer.answer}"
        )
        if agent_answer.justification:
            logger.debug(f"{q_prefix}LLM's justification: {agent_answer.justification}")
        return agent_answer.answer

    # TODO: For a lot of the models available through Google AI Studio, they can't
    # use tools to fetch the docs content. It *might* be worthwhile to actually fetch,
    # parse, and embed the page into the prompt ourselves for those models to get results with mode models.
    # OR, we could switch to something like VertexAI and see if we have access to more models with tool use
    # there.

    assert client is not None
    prompt = make_prompt(
        question, with_docs=with_docs, docs_urls=docs_urls, docs_content=docs_content
    )
    logger.debug(f"{q_prefix}Prompt sent to LLM: {prompt}")
    # TODO: use https://ai.google.dev/api/batch-api instead of single requests.
    agent_answer = get_agent_answer(client, model, tools, prompt)
    if not agent_answer:
        logger.error(f"{q_prefix}LLM's answer couldn't be parsed!")
        return None
    logger.info(
        f"{q_prefix}Correct answer: {question.answer}, LLM's answer: {agent_answer.answer}"
    )
    if agent_answer.justification:
        logger.debug(f"{q_prefix}LLM's justification: {agent_answer.justification}")
    return agent_answer.answer


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


def make_prompt(
    question: Question,
    with_docs: bool,
    docs_urls: list[str] | None = None,
    docs_content: str | None = None,
) -> str:
    if with_docs and docs_content:
        preamble = f"Based on the following documentation:\n\n{docs_content}\n\n"
    elif with_docs and docs_urls:
        preamble = "Based on this documentation: " + ", ".join(docs_urls) + ",\n"
    else:
        preamble = ""
    return (
        preamble
        + "Select the correct answer to the following question:\n"
        + f"{question.question}\n"
        + "\n".join(f"- {letter}: {answer}" for letter, answer in question.options.items())
        + "\n"
    )


def _serialize_evaluation_result(result: EvaluationResult) -> dict:
    answers_out = []
    for qr in result.answers:
        if result.num_candidates == 1:
            selected = qr.runs[0] if qr.runs else None
            answers_out.append(
                {
                    "expected": qr.expected,
                    "selected": selected,
                    "correct": (selected == qr.expected) if selected is not None else None,
                }
            )
        else:
            answers_out.append(
                {
                    "expected": qr.expected,
                    "pass_rate": qr.pass_rate,
                    "selected": list(qr.runs),
                }
            )
    out: dict = {
        "num_questions": result.num_questions,
        "correct_answers": result.correct_answers,
        "invalid_answers": result.invalid_answers,
        "score": result.score,
        "answers": answers_out,
    }
    if result.num_candidates > 1:
        out["num_candidates"] = result.num_candidates
        out["score_std"] = result.score_std
    return out


def _add_evaluate_args(p: argparse.ArgumentParser, questions_required: bool = True) -> None:
    p.add_argument("--questions", type=Path, required=questions_required)
    p.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help='LLM model to use (e.g. "gemini-2.5-flash", "ollama:qwen3-coder-next:latest"). Use "test:dummy" for random answers without any API calls.',
    )
    p.add_argument(
        "--docs-url",
        nargs="*",
        default=None,
        help="URLs to documentation pages that will be used as context for all questions.",
    )
    p.add_argument(
        "--docs-file",
        nargs="*",
        type=Path,
        default=None,
        help="Local file paths whose content will be inlined into the prompt as documentation context.",
    )
    p.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_DEFAULT_URL,
        help=f"Base URL of the Ollama server (default: {OLLAMA_DEFAULT_URL}). Only used with 'ollama:' models.",
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to ask each question. Scores are averaged across candidates. (default: 1)",
    )
    p.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format. 'json' emits a machine-readable JSON object suitable for CI pipelines.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count")
    # Evaluate args on the top-level parser so `docmetrics` (no subcommand) works.
    _add_evaluate_args(parser, questions_required=False)

    subparsers = parser.add_subparsers(dest="subcommand")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate an LLM on documentation questions (default when no subcommand is given).",
    )
    evaluate_parser.add_argument("-v", "--verbose", action="count")
    _add_evaluate_args(evaluate_parser)

    quiz_parser = subparsers.add_parser(
        "quiz",
        help="Take the quiz interactively in the terminal.",
    )
    quiz_parser.add_argument("-v", "--verbose", action="count")
    quiz_parser.add_argument("--questions", type=Path, required=True)

    args = parser.parse_args()
    verbose: int = args.verbose or 0

    logging.basicConfig(
        level=logging.DEBUG if verbose >= 3 else logging.INFO if verbose == 2 else logging.WARNING,
        handlers=[
            rich.logging.RichHandler(markup=True, console=rich.console.Console(stderr=True))
        ],
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
    docs_urls: list[str] | None = args.docs_url if args.docs_url else None
    docs_files: list[Path] | None = args.docs_file if args.docs_file else None
    ollama_url: str = args.ollama_url
    num_candidates: int = args.num_candidates
    if docs_urls and docs_files:
        parser.error("--docs-url and --docs-file are mutually exclusive.")
    score_with_no_context = evaluate_llm(
        questions,
        with_docs=False,
        model=model,
        ollama_url=ollama_url,
        num_candidates=num_candidates,
    )
    score_with_docs = evaluate_llm(
        questions,
        with_docs=True,
        model=model,
        docs_urls=docs_urls,
        docs_files=docs_files,
        ollama_url=ollama_url,
        num_candidates=num_candidates,
    )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "questions": [{"question": q.question} for q in questions],
                    "without_docs": _serialize_evaluation_result(score_with_no_context),
                    "with_docs": _serialize_evaluation_result(score_with_docs),
                }
            )
        )
    else:
        print(f"{score_with_no_context=}")
        print(f"{score_with_docs=}")


if __name__ == "__main__":
    main()
