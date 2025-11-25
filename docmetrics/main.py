import argparse
from pathlib import Path
from typing import Callable, Sequence
import pydantic
from pydantic.dataclasses import dataclass
import yaml
from google import genai
from google.genai import types
import rich.logging
import logging

logger = logging.getLogger(__name__)
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()


@dataclass
class Question:
    question: str
    answers: list[str]
    correct_answer: str

    docs_urls: list[str] | None = None

    def __postinit__(self):
        assert self.correct_answer in self.answers, "correct answer isn't in the options!"


def load_questions(questions_path: Path) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(questions_path.read_text())]


doc_url = "https://docs.mila.quebec"

# Retrieve and encode the PDF byte
# doc_data = httpx.get(doc_url).content


class Response(pydantic.BaseModel):
    answer: int = pydantic.Field(description="The selected answer (integer).", ge=1)


def evaluate_llm(
    questions: list[Question],
    with_docs: bool,
    model: str = "gemini-2.5-flash",
    tools: Sequence[types.Tool | Callable] | None = None,
) -> int:
    """Evaluates a given LLM with the given questions and context.

    Returns the number of correct answers.
    """
    tools = list(tools) if tools else []
    if with_docs:
        tools.append(types.Tool(url_context=types.UrlContext()))

    correct_answers = 0
    invalid_answers = 0
    for question in questions:
        contents = make_content(question, with_docs=with_docs)
        logger.debug(f"Prompt sent to LLM: {contents}")

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                # This response_mime_type can only be set if using the gemini-3-pro model.
                # https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools
                response_mime_type="application/json"
                if not tools and model == "gemini-2.5-flash"
                else None,
                response_json_schema=Response.model_json_schema(),
                tools=tools,
            ),
        )
        assert response.candidates
        if response.candidates[0].url_context_metadata:
            logger.debug(
                f"Consulted URLs: {response.candidates[0].url_context_metadata.url_metadata}"
            )
        assert response.text
        logger.info(
            f"Correct answer: {question.correct_answer!r}, Agent response: {response.text}"
        )

        agent_answer: int | None = None
        try:
            agent_answer = Response.model_validate_json(response.text).answer
        except pydantic.ValidationError:
            pass
        if agent_answer is None:
            try:
                last_line = response.text.strip().splitlines()[-1].strip().removesuffix(".")
                last_word = last_line.split()[-1]
                agent_answer = int(last_word)
            except ValueError:
                logger.error(f"Invalid guess: {response.text}")
                invalid_answers += 1
                continue

        correct_answer = question.answers.index(question.correct_answer) + 1
        logger.debug(f"LLM answer: {agent_answer}, correct answer: {correct_answer}")
        if agent_answer == correct_answer:
            correct_answers += 1
    return correct_answers


def make_content(question: Question, with_docs: bool) -> str:
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
            if question.docs_urls and with_docs
            else ""
        )
        # + ((context + "\n\n") if context else "")
        + "Select the correct answer to the following question:\n"
        + f"- {question.question}\n"
        + answers_block
        + f"Only provide the answer as a single integer with no explanation: ({possible_answers}):\n"
    )

    return contents


def _ok[T](v: T | None) -> T:
    assert v is not None
    return v


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
    )
    logger.setLevel(
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    )

    questions = load_questions(questions_path=questions_path)
    score_with_no_context = evaluate_llm(questions, with_docs=False, model=model)

    score_with_mila_docs_urls = evaluate_llm(questions, with_docs=True, model=model)

    print(f"{score_with_no_context=}")
    print(f"{score_with_mila_docs_urls=}")


if __name__ == "__main__":
    main()
