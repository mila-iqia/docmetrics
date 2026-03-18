import argparse
import logging
import re
from pathlib import Path
from typing import Callable, Literal, Sequence

import pydantic
import rich.logging
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

Letter = Literal["A", "B", "C", "D", "E"]


@dataclass
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


def evaluate_llm(
    client: genai.Client,
    questions: list[Question],
    with_docs: bool,
    model: str,
    tools: Sequence[types.Tool | Callable] | None = None,
) -> int:
    """Evaluates an LLM on some questions with/without documentation as context.

    Parameters
    ----------
    questions: The list of questions to ask the LLM.
    with_docs: Whether to provide documentation URLs as context to the LLM.
    model: The name of the LLM model to use.
    tools: Additional tools to provide to the LLM.

    Returns
    -------
    The number of correct answers.
    """
    tools = list(tools) if tools else []
    if with_docs:
        # Adding this gives the LLM the ability to consult URLs given in the prompt.
        tools.append(types.Tool(url_context=types.UrlContext()))

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
            tools=tools,
        )
        if result is None:
            invalid_answers += 1
        else:
            correct_answers += int(result)
    return correct_answers


def load_questions(questions_path: Path) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(questions_path.read_text())]


def ask_question(
    client: genai.Client,
    question: Question,
    with_docs: bool,
    model: str,
    tools: list[types.Tool | Callable] | None,
) -> bool | None:
    """Asks a question to the LLM and returns whether the LLM answered correctly.

    Returns None if the LLM's answer was invalid.
    """
    prompt = make_prompt(question, with_docs=with_docs)
    logger.debug(f"Prompt sent to LLM: [magenta]{prompt}")
    # TODO: use https://ai.google.dev/api/batch-api instead of single requests.
    response = client.models.generate_content(
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
    assert response.candidates

    logger.debug(f"There were {len(response.candidates)} response candidates from the LLM.")
    for candidate in response.candidates:
        # todo: look into this maybe?
        logger.debug(
            f"Grounding metadata for candidate {candidate.index}: {candidate.grounding_metadata}"
        )
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

    assert response.text

    agent_answer: Response | None = None

    if isinstance(response.parsed, Response):
        logger.debug("LLM output was correctly parsed by the client library.")
        agent_answer = response.parsed
    else:
        try:
            agent_answer = Response.model_validate_json(response.text)
        except pydantic.ValidationError:
            agent_answer = parse_response(response.text)

    if agent_answer is None:
        logger.error(f"Invalid response: {response.text}")
        return None

    # correct_answer = question.options[question.answer]
    logger.info(f"Correct answer: {question.answer}, LLM's answer: {agent_answer.answer}")
    if agent_answer.justification:
        logger.debug(f"LLM's justification: {agent_answer.justification}")
    return agent_answer.answer == question.answer


def parse_response(response_str: str) -> Response | None:
    """Parses a `Response` object from the API output when the LLM doesn't follow the requested response schema.

    >>> get_llm_answer_from_response('A')
    Response(answer='A', justification='')

    >>> get_llm_answer_from_response('This description perfectly matches the requirement for storing "temporary model checkpoints". Answer: A')
    Response(answer='A', justification='This description perfectly matches the requirement for storing "temporary model checkpoints".')
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
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
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

    load_dotenv()  # reads variables from a .env file and sets them in os.environ
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()
    questions = load_questions(questions_path=questions_path)
    score_with_no_context = evaluate_llm(client, questions, with_docs=False, model=model)
    score_with_mila_docs_urls = evaluate_llm(client, questions, with_docs=True, model=model)

    print(f"{score_with_no_context=}")
    print(f"{score_with_mila_docs_urls=}")


if __name__ == "__main__":
    main()
