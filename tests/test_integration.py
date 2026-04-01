"""Integration tests requiring a live Ollama instance.

Run with::

    DOCMETRICS_INTEGRATION_TEST=1 OLLAMA_URL=http://dgx:11434 uv run pytest tests/test_integration.py -v
"""

import os
import random
import subprocess
import time
from pathlib import Path

import pytest

from docmetrics.main import OLLAMA_DEFAULT_URL, evaluate_llm, load_questions

OLLAMA_URL = os.environ.get("OLLAMA_URL", OLLAMA_DEFAULT_URL)
MILA_QUESTIONS_FILE = Path(__file__).parent.parent / "mila_docs_questions.yaml"
MILA_DOCS_URL = "https://docs.mila.quebec"
MODEL = "ollama:gpt-oss:120b"

in_github_CI = "GITHUB_ACTIONS" in os.environ
integration = pytest.mark.skipif(
    in_github_CI and os.environ.get("DOCMETRICS_INTEGRATION_TEST", "0") != "1",
    reason="Runs only on dev machines or in GitHub CI when DOCMETRICS_INTEGRATION_TEST=1 is set.",
)


# Cool idea, but this actually affects the results!
@pytest.fixture(scope="session")
def mila_docs_url(tmp_path_factory: pytest.TempPathFactory):
    # IDEA: Spawn a subprocess that just self-hosts the mila-docs locally, and yield the URL to it.
    # This way we can test with a local copy of the docs, and avoid hitting the real docs repeatedly during testing.
    if _SELF_HOST_DOCS := os.environ.get("DOCMETRICS_SELF_HOST_DOCS", "0") == "1":
        docs_repo = "https://www.github.com/mila-iqia/mila-docs"
        tmp_dir = tmp_path_factory.mktemp("mila-docs", numbered=False)
        if not (tmp_dir / "mkdocs.yml").exists():
            subprocess.check_call(f"git clone {docs_repo} {tmp_dir}", shell=True)
        addr = "localhost:8122"
        proc = subprocess.Popen(
            f"cd {tmp_dir} && uv run mkdocs serve --dev-addr={addr}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        yield addr
        proc.terminate()
    else:
        yield MILA_DOCS_URL


@integration
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("num_questions", [5])
@pytest.mark.parametrize("num_candidates", [1, 3])
@pytest.mark.parametrize(
    "model", ["ollama:gpt-oss:20b", "ollama:gpt-oss:120b", "ollama:qwen3-coder-next"]
)
def test_mila_docs(
    mila_docs_url: str, seed: int, num_questions: int, num_candidates: int, model: str
):
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
        docs_urls=[mila_docs_url],
        ollama_url=OLLAMA_URL,
        num_candidates=num_candidates,
    )
    time_taken_with_docs = time.time() - t_start

    print(f"Time taken without docs: {time_taken_no_docs:.1f} seconds")
    print(f"Time taken with docs:    {time_taken_with_docs:.1f} seconds")
    print(f"\nWithout docs: {result_no_docs.score:.1%} (std: {result_no_docs.score_std:.1%})")
    print(f"With docs:    {result_with_docs.score:.1%} (std: {result_with_docs.score_std:.1%})")

    print(
        f"Time per answer without docs: {time_taken_no_docs / (num_questions * num_candidates):.1f} seconds"
    )
    print(
        f"Time per answer with docs: {time_taken_with_docs / (num_questions * num_candidates):.1f} seconds"
    )
    # IDEALLY.
    assert result_with_docs.score >= result_no_docs.score, (
        f"Expected with-docs ({result_with_docs.score:.1%}) > without-docs ({result_no_docs.score:.1%})"
    )
