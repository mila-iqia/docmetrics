"""A new project/tool called docmetrics."""

from docmetrics.main import (
    evaluate_llm,  # noqa: F401
    EvaluationResult,
    Question,
    Response,
    ask_question,
    load_questions,
)

__all__ = [
    "evaluate_llm",
    "ask_question",
    "load_questions",
    "Question",
    "Response",
    "EvaluationResult",
]
