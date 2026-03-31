"""Integration tests requiring a live Ollama instance.

Run with::

    DOCMETRICS_INTEGRATION_TEST=1 OLLAMA_URL=http://dgx:11434 uv run pytest tests/test_integration.py -v
"""

import os
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
def test_mila_docs_quiz_with_ollama():
    """Mila docs quiz: with-docs score should exceed without-docs score."""
    questions = load_questions(MILA_QUESTIONS_FILE)
    t_start = time.time()
    result_no_docs = evaluate_llm(
        questions,
        with_docs=False,
        model=MODEL,
        ollama_url=OLLAMA_URL,
        num_candidates=3,
    )
    time_taken_no_docs = time.time() - t_start
    t_start = time.time()
    result_with_docs = evaluate_llm(
        questions,
        with_docs=True,
        model=MODEL,
        docs_urls=[MILA_DOCS_URL],
        ollama_url=OLLAMA_URL,
        num_candidates=3,
    )
    time_taken_docs = time.time() - t_start

    print(f"Time taken without docs: {time_taken_no_docs:.1f} seconds")
    print(f"Time taken with docs:    {time_taken_docs:.1f} seconds")
    print(f"\nWithout docs: {result_no_docs.score:.1%} (std: {result_no_docs.score_std:.1%})")
    print(f"With docs:    {result_with_docs.score:.1%} (std: {result_with_docs.score_std:.1%})")

    assert result_with_docs.score > result_no_docs.score, (
        f"Expected with-docs ({result_with_docs.score:.1%}) > without-docs ({result_no_docs.score:.1%})"
    )
