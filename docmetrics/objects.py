import math
from typing import Literal

import pydantic
from pydantic.dataclasses import dataclass

Letter = Literal["A", "B", "C", "D", "E"]


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


@dataclass(frozen=True)
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
