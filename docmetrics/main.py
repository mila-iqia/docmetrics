import argparse
import dataclasses
import functools
import json
import logging
import os
import random
import re
import warnings
from pathlib import Path
from typing import Callable, Literal

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
OLLAMA_DEFAULT_URL = "http://localhost:11434"
"""Default base URL for the Ollama server."""
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

    def __postinit__(self):
        assert self.answer in self.options, "correct answer isn't in the options!"


class Response(pydantic.BaseModel):
    answer: Letter
    """The selected answer."""

    # TODO: Check if adding this 'justification' is actually helpful, and whether it increases costs.
    # Seems like it might just be extra tokens to generate, for our purposes.
    justification: str = ""
    """A brief justification for the selected answer."""


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


OLLAMA_WEB_FETCH_SUPPORTED = "OLLAMA_API_KEY" in os.environ


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


def _get_agent_answer_ollama(
    model: str,
    prompt: str,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    use_web_fetch: bool = False,
) -> "Response | None":
    import ollama as ollama_lib

    client = _get_ollama_client(ollama_url)
    ollama_model = _ollama_model_name(model)
    schema = Response.model_json_schema()
    full_prompt = (
        prompt
        + f"\nRespond with a JSON object matching this schema: {json.dumps(schema)}\n"
        + 'Example: {"answer": "A", "justification": "Because..."}'
    )

    messages: list = [{"role": "user", "content": full_prompt}]
    tools = [ollama_lib.web_fetch] if use_web_fetch else None

    while True:
        response = client.chat(
            model=ollama_model,
            messages=messages,
            tools=tools,
            format="json" if not tools else None,
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            content = response.message.content
            if not content:
                return None
            try:
                return Response.model_validate_json(content)
            except pydantic.ValidationError:
                return parse_response_fallback(content)

        for tool_call in response.message.tool_calls:
            if tool_call.function.name == "web_fetch":
                url = tool_call.function.arguments.get("url", "")
                logger.info(f"LLM fetching documentation URL: {url}")
                if not OLLAMA_WEB_FETCH_SUPPORTED:
                    raise NotImplementedError(
                        "TODO: For now, you need to set OLLAMA_API_KEY for web search to work. "
                    )
                    # TODO: Fallback option, use something like httpx to fetch the page contents ourselves.

                assert "localhost" not in url, (
                    "web_fetch is happening on the server-side. Can't fetch a locally-hosted page."
                )
                try:
                    result = client.web_fetch(**tool_call.function.arguments)
                    content = str(result)
                except Exception as e:
                    logger.warning(f"web_fetch failed for {url}: {e}")
                    content = f"Error fetching URL: {e}"
            else:
                logger.warning(f"LLM requested unknown tool: {tool_call.function.name}")
                content = f"Tool {tool_call.function.name} not found"
            messages.append(
                {"role": "tool", "content": content, "tool_name": tool_call.function.name}
            )


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    num_questions: int
    correct_answers: int
    invalid_answers: int
    answers: tuple[bool | None, ...] = ()
    """Per-question results: True = correct, False = incorrect, None = invalid."""

    @property
    def score(self) -> float:
        """The percentage of questions that the LLM answered correctly."""
        return self.correct_answers / self.num_questions if self.num_questions > 0 else 0.0


def evaluate_llm(
    questions: list[Question],
    with_docs: bool,
    model: str,
    docs_urls: list[str] | None = None,
    docs_files: list[Path] | None = None,
    ollama_url: str = OLLAMA_DEFAULT_URL,
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

    Returns
    -------
    The evaluation results.


    Notes
    - Could also input a list of tools to give to the agent, in addition to URL search.
    """
    client = None if model == DUMMY_MODEL or _is_ollama_model(model) else get_google_genai_client()
    # for model_config in client.models.list().page:
    #     if model_config.supported_actions and "generateContent" in model_config.supported_actions:
    #         logger.info(f"Available model: {model_config.name}")
    #         logger.debug(f"Model config: {model_config}")
    # exit()
    num_questions = len(questions)
    correct_answers = 0
    invalid_answers = 0
    answers: list[bool | None] = []

    docs_content: str | None = None
    if docs_files:
        docs_content = "\n---\n".join(f.read_text() for f in docs_files)

    # TODO: Group questions based on the docs pages they require and use the batch API
    # to ask multiple questions at once with the same context.

    for question in questions:
        result = ask_question(
            client=client,
            question=question,
            with_docs=with_docs,
            model=model,
            docs_urls=docs_urls,
            docs_content=docs_content,
            ollama_url=ollama_url,
            # Adding this gives the LLM the ability to consult URLs given in the prompt.
            # Not needed (or wanted) when docs are inlined via docs_files or for Ollama models.
            tools=(
                [types.Tool(url_context=types.UrlContext())]
                if with_docs and not docs_files and not _is_ollama_model(model)
                else None
            ),
        )
        answers.append(result)
        if result is None:
            invalid_answers += 1
        else:
            correct_answers += int(result)
    return EvaluationResult(
        num_questions=num_questions,
        correct_answers=correct_answers,
        invalid_answers=invalid_answers,
        answers=tuple(answers),
    )


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
) -> bool | None:
    """Asks a question to the LLM and returns whether the LLM answered correctly.

    Returns None if the LLM's answer was invalid.
    """
    if model == DUMMY_MODEL:
        dummy_answer = random.choice(list(question.options.keys()))
        logger.info(f"Correct answer: {question.answer}, dummy answer: {dummy_answer}")
        return dummy_answer == question.answer

    if _is_ollama_model(model):
        prompt = make_prompt(
            question, with_docs=with_docs, docs_urls=docs_urls, docs_content=docs_content
        )
        logger.debug(f"Prompt sent to LLM: [magenta]{prompt}")
        # Use web_fetch tool when docs are given as URLs (not already inlined via --docs-file).
        use_web_fetch = with_docs and bool(docs_urls) and docs_content is None
        agent_answer = _get_agent_answer_ollama(
            model, prompt, ollama_url, use_web_fetch=use_web_fetch
        )
        if not agent_answer:
            logger.error("LLM's answer couldn't be parsed!")
            return None
        logger.info(f"Correct answer: {question.answer}, LLM's answer: {agent_answer.answer}")
        if agent_answer.justification:
            logger.debug(f"LLM's justification: {agent_answer.justification}")
        return agent_answer.answer == question.answer

    # TODO: For a lot of the models available through Google AI Studio, they can't
    # use tools to fetch the docs content. It *might* be worthwhile to actually fetch,
    # parse, and embed the page into the prompt ourselves for those models to get results with mode models.
    # OR, we could switch to something like VertexAI and see if we have access to more models with tool use
    # there.

    assert client is not None
    prompt = make_prompt(
        question, with_docs=with_docs, docs_urls=docs_urls, docs_content=docs_content
    )
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
    if docs_urls and docs_files:
        parser.error("--docs-url and --docs-file are mutually exclusive.")
    score_with_no_context = evaluate_llm(
        questions, with_docs=False, model=model, ollama_url=ollama_url
    )
    score_with_docs = evaluate_llm(
        questions,
        with_docs=True,
        model=model,
        docs_urls=docs_urls,
        docs_files=docs_files,
        ollama_url=ollama_url,
    )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "questions": [{"question": q.question} for q in questions],
                    "without_docs": {
                        "num_questions": score_with_no_context.num_questions,
                        "correct_answers": score_with_no_context.correct_answers,
                        "invalid_answers": score_with_no_context.invalid_answers,
                        "score": score_with_no_context.score,
                        "answers": list(score_with_no_context.answers),
                    },
                    "with_docs": {
                        "num_questions": score_with_docs.num_questions,
                        "correct_answers": score_with_docs.correct_answers,
                        "invalid_answers": score_with_docs.invalid_answers,
                        "score": score_with_docs.score,
                        "answers": list(score_with_docs.answers),
                    },
                }
            )
        )
    else:
        print(f"{score_with_no_context=}")
        print(f"{score_with_docs=}")


if __name__ == "__main__":
    main()
