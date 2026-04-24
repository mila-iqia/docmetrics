import math
from typing import Literal

import pydantic
from pydantic.dataclasses import dataclass

# Can use up to 10 choices per question (to make the quiz harder perhaps).
Letter = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


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


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    """The question that was asked."""

    runs: tuple[Letter | None, ...]
    """One selected letter per candidate run (None = unparsable response)."""

    @property
    def correct_count(self) -> int:
        return sum(r == self.question.answer for r in self.runs)

    @property
    def incorrect_count(self) -> int:
        return sum(r != self.question.answer for r in self.runs)

    @property
    def pass_rate(self) -> float:
        return self.correct_count / len(self.runs) if self.runs else 0.0


# @dataclass(frozen=True)
class EvaluationResult(pydantic.BaseModel):
    question_results: tuple[QuestionResult, ...]
    """Per-question results."""

    num_candidates: int = 1
    """Number of candidate answers requested per question."""

    @pydantic.computed_field
    @property
    def num_questions(self) -> int:
        return len(self.question_results)

    @pydantic.computed_field
    @property
    def correct_answers(self) -> int:
        return sum(r.correct_count for r in self.question_results)

    @pydantic.computed_field
    @property
    def incorrect_answers(self) -> int:
        return sum(r.incorrect_count for r in self.question_results)

    @pydantic.computed_field
    @property
    def invalid_answers(self) -> int:
        return sum(1 for r in self.question_results for run in r.runs if run is None)

    @pydantic.computed_field
    @property
    def score(self) -> float:
        """Mean per-question pass rate."""
        if not self.question_results:
            return 0.0
        return sum(r.pass_rate for r in self.question_results) / len(self.question_results)

    @pydantic.computed_field
    @property
    def score_std(self) -> float:
        """Population std-dev of per-question pass rates."""
        if not self.question_results:
            return 0.0
        rates = [r.pass_rate for r in self.question_results]
        mean = sum(rates) / len(rates)
        variance = sum((r - mean) ** 2 for r in rates) / len(rates)
        return math.sqrt(variance)
