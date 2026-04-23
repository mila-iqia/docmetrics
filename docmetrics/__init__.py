"""A new project/tool called docmetrics."""

from docmetrics.main import (
    EvaluationResult,
    Response,
    ask_question,
    evaluate_llm,  # noqa: F401
    load_questions,
)
from docmetrics.objects import Question

__all__ = [
    "evaluate_llm",
    "ask_question",
    "load_questions",
    "Question",
    "Response",
    "EvaluationResult",
]
