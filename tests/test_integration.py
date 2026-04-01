"""Integration tests requiring a live Ollama instance.

Run with::

    DOCMETRICS_INTEGRATION_TEST=1 OLLAMA_URL=http://dgx:11434 uv run pytest tests/test_integration.py -v
"""

import os
import random
import time
from pathlib import Path

import pytest

from docmetrics.main import OLLAMA_DEFAULT_URL, evaluate_llm, load_questions

OLLAMA_URL = os.environ.get("OLLAMA_URL", OLLAMA_DEFAULT_URL)
MILA_QUESTIONS_FILE = Path(__file__).parent.parent / "mila_docs_questions.yaml"
MILA_DOCS_URL = "https://docs.mila.quebec"
MODEL = "ollama:gpt-oss:120b"

integration = pytest.mark.skipif(
    not os.environ.get("DOCMETRICS_INTEGRATION_TEST"),
    reason="Set DOCMETRICS_INTEGRATION_TEST=1 to run integration tests",
)


@integration
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("num_questions", [10])
@pytest.mark.parametrize("num_candidates", [1, 3])
@pytest.mark.parametrize("model", ["ollama:gpt-oss:20b", "ollama:gpt-oss:120b"])
def test_mila_docs(seed: int, num_questions: int, num_candidates: int, model: str):
    """Mila docs quiz: with-docs score should exceed without-docs score."""
    questions = load_questions(MILA_QUESTIONS_FILE)
    if num_questions == -1:  # If num_questions is -1, use all questions.
        num_questions = len(questions)
    else:
        # Use a subset of questions for speed.
        random.seed(seed)
        random.shuffle(questions)
        questions = questions[:num_questions]

    t_start = time.time()
    result_no_docs = evaluate_llm(
        questions,
        with_docs=False,
        model=model,
        ollama_url=OLLAMA_URL,
        num_candidates=num_candidates,
    )
    time_taken_no_docs = time.time() - t_start
    t_start = time.time()
    result_with_docs = evaluate_llm(
        questions,
        with_docs=True,
        model=model,
        docs_urls=[MILA_DOCS_URL],
        ollama_url=OLLAMA_URL,
        num_candidates=num_candidates,
    )
    time_taken_docs = time.time() - t_start

    print(f"Time taken without docs: {time_taken_no_docs:.1f} seconds")
    print(f"Time taken with docs:    {time_taken_docs:.1f} seconds")
    print(f"\nWithout docs: {result_no_docs.score:.1%} (std: {result_no_docs.score_std:.1%})")
    print(f"With docs:    {result_with_docs.score:.1%} (std: {result_with_docs.score_std:.1%})")

    print(
        f"Time per answer without docs: {time_taken_no_docs / (num_questions * num_candidates):.1f} seconds"
    )
    print(
        f"Time per answer without docs: {time_taken_no_docs / (num_questions * num_candidates):.1f} seconds"
    )
    # IDEALLY.
    assert result_with_docs.score > result_no_docs.score, (
        f"Expected with-docs ({result_with_docs.score:.1%}) > without-docs ({result_no_docs.score:.1%})"
    )
