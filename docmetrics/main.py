import argparse
import dataclasses
import functools
import logging
import random
import re
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

    @property
    def score(self) -> float:
        """The percentage of questions that the LLM answered correctly."""
        return self.correct_answers / self.num_questions if self.num_questions > 0 else 0.0


def evaluate_llm(
    questions: list[Question],
    with_docs: bool,
    model: str,
    # tools: Sequence[types.Tool | Callable] | None = None,
) -> EvaluationResult:
    """Evaluates an LLM on some questions with/without documentation as context.

    Parameters
    ----------
    questions: The list of questions to ask the LLM.
    with_docs: Whether to provide documentation URLs as context to the LLM.
    model: The name of the LLM model to use.

    Returns
    -------
    The evaluation results.


    Notes
    - Could also input a list of tools to give to the agent, in addition to URL search.
    """
    client = None if model == DUMMY_MODEL else get_google_genai_client()
    # for model_config in client.models.list().page:
    #     if model_config.supported_actions and "generateContent" in model_config.supported_actions:
    #         logger.info(f"Available model: {model_config.name}")
    #         logger.debug(f"Model config: {model_config}")
    # exit()
    num_questions = len(questions)
    correct_answers = 0
    invalid_answers = 0

    # TODO: Group questions based on the docs pages they require and use the batch API
    # to ask multiple questions at once with the same context.

    for question in questions:
        result = ask_question(
            client=client,
            question=question,
            with_docs=with_docs,
            model=model,
            # Adding this gives the LLM the ability to consult URLs given in the prompt.
            tools=[types.Tool(url_context=types.UrlContext())] if with_docs else None,
        )
        if result is None:
            invalid_answers += 1
        else:
            correct_answers += int(result)
    return EvaluationResult(
        num_questions=num_questions,
        correct_answers=correct_answers,
        invalid_answers=invalid_answers,
    )


def load_questions(questions_path: Path) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(questions_path.read_text())]


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

    # TODO: For a lot of the models available through Google AI Studio, they cant
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="count")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help='LLM model to use (e.g. "gemini-2.5-flash"). Use "test:dummy" for random answers without any API calls.',
    )
    args = parser.parse_args()
    questions_path: Path = args.questions
    verbose: int = args.verbose or 0
    model: str = args.model

    logging.basicConfig(
        level=logging.DEBUG if verbose >= 3 else logging.INFO if verbose == 2 else logging.WARNING,
        handlers=[rich.logging.RichHandler(markup=True)],
        format="%(message)s",
    )
    logger.setLevel(  # this logger specifically.
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    )

    questions = load_questions(questions_path=questions_path)
    score_with_no_context = evaluate_llm(questions, with_docs=False, model=model)
    score_with_mila_docs_urls = evaluate_llm(questions, with_docs=True, model=model)

    print(f"{score_with_no_context=}")
    print(f"{score_with_mila_docs_urls=}")


if __name__ == "__main__":
    main()
