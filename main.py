import argparse
from pathlib import Path
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

    def __postinit__(self):
        assert self.correct_answer in self.answers, (
            "correct answer isn't in the options!"
        )


def load_questions(questions_path: Path) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(questions_path.read_text())]


doc_url = "https://docs.mila.quebec"

# Retrieve and encode the PDF byte
# doc_data = httpx.get(doc_url).content


class Response(pydantic.BaseModel):
    answer: int = pydantic.Field(description="The selected answer (integer).", ge=1)


def evaluate_llm(
    questions: list[Question], context: str | None = None, llm: str = "gemini-2.5-flash"
) -> int:
    """Evaluates a given LLM with the given questions and context. Returns the number of correct answers."""
    correct_answers = 0
    invalid_answers = 0
    for question in questions:
        answers_block = (
            "\n"
            + "\n".join(
                f"{i + 1}. {answer}" for i, answer in enumerate(question.answers)
            )
            + "\n"
        )
        possible_answers = (
            ", ".join(str(i) for i in range(len(question.answers) - 1))
            + f" or {len(question.answers)}"
        )
        contents = (
            "Select the correct answer to the following question:\n"
            + f"- {question.question}\n"
            + answers_block
            + f"Only provide the answer as a single integer with no explanation: ({possible_answers}):\n"
        )
        logger.debug(f"Prompt sent to LLM: {contents}")
        response = client.models.generate_content(
            model=llm,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": Response.model_json_schema(),
            },
        )
        logger.info(f"Agent response: {response.text}")
        assert response.text
        response = Response.model_validate_json(response.text)
        agent_answer = response.answer
        # if (
        #     response
        # ):
        #     logger.error(f"Invalid guess: {response.text}")
        #     invalid_answers += 1
        #     continue
        correct_answer = question.answers.index(question.correct_answer) + 1
        logger.debug(f"LLM answer: {agent_answer}, correct answer: {correct_answer}")
        if agent_answer == correct_answer:
            correct_answers += 1
    return correct_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="count")
    args = parser.parse_args()
    questions_path: Path = args.questions
    verbose: int = args.verbose

    logging.basicConfig(
        level=logging.DEBUG
        if verbose >= 3
        else logging.INFO
        if verbose == 2
        else logging.WARNING,
        handlers=[rich.logging.RichHandler()],
    )
    logger.setLevel(
        logging.DEBUG
        if verbose >= 2
        else logging.INFO
        if verbose == 1
        else logging.WARNING
    )

    questions = load_questions(questions_path=questions_path)
    score_with_no_context = evaluate_llm(questions, context=None)
    print(score_with_no_context)


if __name__ == "__main__":
    main()
