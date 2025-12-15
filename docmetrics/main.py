import argparse
from pathlib import Path
import re
from typing import Callable, Sequence
import pydantic
from pydantic.dataclasses import dataclass
import yaml
from google import genai
from google.genai import types
import rich.logging
import logging

logger = logging.getLogger(__name__)


@dataclass
class Question:
    question: str
    """The question to ask the LLM."""

    answers: list[str]
    """A list of possible answers to the question."""

    correct_answer: str
    """The correct answer to the question (must be one of the options in `answers`)."""

    docs_urls: list[str] | None = None
    """A list of URLs to relevant documentation pages.

    The hope is that by reading these pages, the LLM can answer the question more accurately. Note
    that the LLMs are not going to look at links within these pages, only the content of these
    pages itself. For example, giving a link to docs.mila.quebec wouldn't be very helpful.
    """

    def __postinit__(self):
        assert self.correct_answer in self.answers, "correct answer isn't in the options!"


class Response(pydantic.BaseModel):
    answer: int = pydantic.Field(description="The selected answer (integer).", ge=1)
    """The selected answer index (1-indexed).

    1: first answer, 2: second answer, etc.
    """

    justification: str = ""
    """A brief justification for the selected answer."""


def evaluate_llm(
    client: genai.Client,
    questions: list[Question],
    with_docs: bool,
    model: str = "gemini-2.5-flash",
    tools: Sequence[types.Tool | Callable] | None = None,
) -> int:
    """Evaluates an LLM on some questions with/without documentation as context.

    Parameters
    ----------
    questions : list[Question]
        The list of questions to ask the LLM.
    with_docs : bool
        Whether to provide documentation URLs as context to the LLM.
    model : str
        The name of the LLM model to use.
    tools : Sequence[types.Tool | Callable] | None
        Additional tools to provide to the LLM.

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
    logger.debug(f"Prompt sent to LLM: {prompt}")
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
    if response.candidates[0].url_context_metadata:
        logger.debug(f"Consulted URLs: {response.candidates[0].url_context_metadata.url_metadata}")
    assert response.text

    agent_answer: Response | None = None
    try:
        agent_answer = Response.model_validate_json(response.text)
    except pydantic.ValidationError:
        agent_answer = get_llm_answer_from_response(response.text)

    if agent_answer is None:
        logger.error(f"Invalid response: {response.text}")
        return None

    correct_answer = question.answers.index(question.correct_answer) + 1
    logger.info(f"Correct answer: {correct_answer}, LLM's answer: {agent_answer.answer}")
    logger.debug(f"LLM's justification: {agent_answer.justification}")
    return agent_answer.answer == correct_answer


def get_llm_answer_from_response(response_str: str) -> Response | None:
    """Extracts the LLM's answer from the response object when the LLM doesn't follow the requested
    response schema.

    >>> example_response = 'This description perfectly matches the requirement for storing "temporary model checkpoints".2'
    >>> get_llm_answer_from_response(example_response)
    Response(answer=2, justification='This description perfectly matches the requirement for storing "temporary model checkpoints".')
    """
    try:
        last_line = response_str.strip().splitlines()[-1].strip().removesuffix(".")
        # todo: if the last character is a digit, use this as the LLM's guess.
        last_char = re.search(r"\d+$", last_line)
        if not last_char:
            logger.error(f"Invalid response: {response_str}")
            return None
        last_word = last_char.group(0)
        return Response(
            answer=int(last_word),
            justification=response_str.strip().removesuffix(last_word).strip(),
        )
    except ValueError:
        logger.error(f"Invalid response: {response_str}")
        return None


def make_prompt(question: Question, with_docs: bool) -> str:
    answers_block = (
        "\n" + "\n".join(f"{i + 1}. {answer}" for i, answer in enumerate(question.answers)) + "\n"
    )
    possible_answers = (
        ", ".join(str(i + 1) for i in range(len(question.answers) - 1))
        + f" or {len(question.answers)}"
    )
    contents = (
        (
            ("Based on these pages of documentation: " + ", ".join(question.docs_urls) + "\n\n")
            if with_docs and question.docs_urls
            else ""
        )
        # + ((context + "\n\n") if context else "")
        + "Select the correct answer to the following question:\n"
        + f"- {question.question}\n"
        + answers_block
        + f"Provide the answer as an integer: ({possible_answers}):\n"
    )

    return contents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="count")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()
    questions_path: Path = args.questions
    verbose: int = args.verbose or 0
    model: str = args.model

    logging.basicConfig(
        level=logging.DEBUG if verbose >= 3 else logging.INFO if verbose == 2 else logging.WARNING,
        handlers=[rich.logging.RichHandler()],
        format="%(message)s",
    )
    logger.setLevel(
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    )

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()
    questions = load_questions(questions_path=questions_path)
    score_with_no_context = evaluate_llm(client, questions, with_docs=False, model=model)
    score_with_mila_docs_urls = evaluate_llm(client, questions, with_docs=True, model=model)

    print(f"{score_with_no_context=}")
    print(f"{score_with_mila_docs_urls=}")


if __name__ == "__main__":
    main()
