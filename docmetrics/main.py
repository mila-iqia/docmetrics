import argparse
import dataclasses
import functools
import json
import logging
import math
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Callable, Literal

import httpx
import pydantic
import rich.console
import rich.logging
import tqdm
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

Letter = Literal["A", "B", "C", "D", "E"]

DUMMY_MODEL = "test:dummy"
"""A model name that returns a random answer without calling any external API."""
CLAUDE_CLI_PREFIX = "claude:"
"""Prefix for Claude CLI model names, e.g. 'claude:claude-opus-4-6'."""
OLLAMA_PREFIX = "ollama:"
"""Prefix for Ollama model names, e.g. 'ollama:qwen3-coder-next:latest'."""
MAX_TOOL_CALLS = 5
"""Maximum number of tool calls allowed per question before giving up."""
OLLAMA_DEFAULT_URL = "http://localhost:11434"
"""Default base URL for the Ollama server."""
__all__ = [
    "evaluate_llm",
    "ask_question",
    "load_questions",
    "Question",
    "Response",
    "QuestionResult",
    "EvaluationResult",
]


@dataclass
class Question:
    question: str
    """The question to ask the LLM."""

    options: dict[Letter, str]
    """A list of possible answers to the question."""

    answer: Letter
    """The correct answer to the question (must be one of the letters in `options`)."""

    skills: set[str] = dataclasses.field(default_factory=set)
    """Tool/skill name that should be invoked when answering (e.g. 'Bash', 'Read')."""

    skills_dir: Path | None = None

    def __postinit__(self):
        assert self.answer in self.options, "correct answer isn't in the options!"


class Response(pydantic.BaseModel):
    answer: Letter
    """The selected answer."""

    # TODO: Check if adding this 'justification' is actually helpful, and whether it increases costs.
    # Seems like it might just be extra tokens to generate, for our purposes.
    justification: str = ""
    """A brief justification for the selected answer."""


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
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


@functools.cache
def get_google_genai_client():
    load_dotenv()  # reads variables from a .env file and sets them in os.environ
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    return genai.Client()


def _is_ollama_model(model: str) -> bool:
    return model.startswith(OLLAMA_PREFIX)


def _is_claude_cli_model(model: str) -> bool:
    return model.startswith(CLAUDE_CLI_PREFIX)


def _claude_cli_model_name(model: str) -> str:
    """Strips the 'claude:' prefix to get the bare model name.

    >>> _claude_cli_model_name("claude:claude-opus-4-6")
    'claude-opus-4-6'
    """
    return model.removeprefix(CLAUDE_CLI_PREFIX)


def _ollama_model_name(model: str) -> str:
    """Strips the 'ollama:' prefix to get the bare model name.

    >>> _ollama_model_name("ollama:qwen3-coder-next:latest")
    'qwen3-coder-next:latest'
    """
    return model.removeprefix(OLLAMA_PREFIX)


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse

    return urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")


def _fetch_url(url: str) -> str:
    """Fetch URL content via httpx, stripping HTML tags if the response is HTML."""
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()
    if "html" not in response.headers.get("content-type", ""):
        return response.text
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip and (stripped := data.strip()):
                self._parts.append(stripped)

    ex = _Extractor()
    ex.feed(response.text)
    return "\n".join(ex._parts)


def _get_agent_answer_claude_cli(
    model: str,
    system_prompt: str,
    prompt: str,
    with_docs: bool,
    use_web_fetch: bool = False,
    docs_urls: list[str] | None = None,
    expected_skills: set[str] | None = None,
    skills_dir: Path | None = None,
) -> "tuple[Response | None, bool]":
    """Runs Claude CLI with stream-json output and parses the response.

    Returns (response, skill_invoked) where skill_invoked is True if the
    expected_skill tool name appeared in a tool_use event."""
    import contextlib
    import subprocess
    import tempfile

    expected_skills = expected_skills or set()

    if _is_ollama_model(model):
        ollama_model = _ollama_model_name(model)
        cmd_prefix = ["ollama", "launch", "claude", "--model", ollama_model, "--"]
    else:
        claude_model = _claude_cli_model_name(model)
        cmd_prefix = ["claude", "--model", claude_model]

    cmd = cmd_prefix + [
        "--output-format",
        "stream-json",
        "--print",
        "--verbose",
        "--allowedTools",  # https://code.claude.com/docs/en/tools-reference
        " ".join(
            [
                "Glob",
                "Grep",
                "Monitor",
                "Read",
                "Skill",
                "ToolSearch",
                *(  # https://code.claude.com/docs/en/settings#permission-rule-syntax
                    [f"WebFetch(domain:{d})" for d in docs_urls]
                    if use_web_fetch and docs_urls
                    else (["WebFetch"] if use_web_fetch else [])
                ),
                "WebSearch",
            ]
        ),
        "--permission-mode",
        "dontAsk",  # https://code.claude.com/docs/en/agent-sdk/permissions#available-modes
        "--system-prompt",
        system_prompt,
        "--json-schema",
        json.dumps(Response.model_json_schema()),
    ]

    ctx = tempfile.TemporaryDirectory() if with_docs and skills_dir else contextlib.nullcontext()
    with ctx as tmpdir:
        cwd = None
        # There seams to be no other wait than to have .claude/skills in the CWD
        # for claude to find the required skills
        if skills_dir and tmpdir:
            claude_dir = Path(tmpdir) / ".claude"
            claude_dir.mkdir()
            skills_dir.copy(claude_dir / "skills")
            # --add-dir seams to make claude code hang
            # cmd.extend(["--add-dir", str(skills_dir)])
            # cmd.extend(["--add-dir", tmpdir])
            cwd = tmpdir
        cmd.append(prompt)

        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"Claude CLI exited with code {proc.returncode}: {proc.stderr}")
            return None, False

    skills_invoked = set()
    final_text: str | None = None

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "assistant":
            content = event.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "tool_use":
                    tool_name = block.get("name", "")
                    skill_name = (
                        block.get("input", {}).get("skill")
                        if tool_name.lower() == "skill"
                        else None
                    )
                    logger.info(f"Claude CLI invoked tool: {tool_name}:{skill_name}")
                    skills_invoked.add(skill_name or tool_name)
                elif block.get("type") == "text":
                    final_text = block.get("text", "")
        elif event.get("type") == "result":
            result_text = event.get("structured_output", event.get("result", ""))
            if result_text:
                final_text = result_text

    check_skills = False

    if with_docs:
        check_skills = not (expected_skills - skills_invoked)
    else:
        check_skills = not (expected_skills & skills_invoked)

    if not final_text:
        return None, check_skills

    for validate in [Response.model_validate, Response.model_validate_json]:
        try:
            return validate(final_text), check_skills
        except pydantic.ValidationError:
            pass
    return parse_response_fallback(final_text), check_skills


def evaluate_llm(
    questions: list[Question],
    with_docs: bool,
    model: str,
    docs_urls: list[str] | None = None,
    docs_files: list[Path] | None = None,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    num_candidates: int = 1,
    # tools: Sequence[types.Tool | Callable] | None = None,
) -> EvaluationResult:
    """Evaluates an LLM on some questions with/without documentation as context.

    Parameters
    ----------
    questions: The list of questions to ask the LLM.
    with_docs: Whether to provide documentation URLs as context to the LLM.
    model: The name of the LLM model to use.
    docs_urls: A list of URLs to relevant documentation pages to provide as context.
    docs_files: A list of local file paths whose content will be inlined into the prompt.
    ollama_url: Base URL of the Ollama server (only used when model has the 'ollama:' prefix).
    num_candidates: Number of times to ask each question. Scores are averaged across candidates.

    Returns
    -------
    The evaluation results.


    Notes
    - Could also input a list of tools to give to the agent, in addition to URL search.
    """
    client = (
        None
        if model == DUMMY_MODEL or _is_ollama_model(model) or _is_claude_cli_model(model)
        else get_google_genai_client()
    )
    num_questions = len(questions)

    docs_content: str | None = None
    if docs_files:
        docs_content = "\n---\n".join(f.read_text() for f in docs_files)
    elif (
        with_docs
        and docs_urls
        and not _is_ollama_model(model)
        and any(_is_local_url(u) for u in docs_urls)
    ):
        # Google's url_context tool runs server-side and can't reach localhost URLs; pre-fetch them.
        logger.info(f"Pre-fetching {len(docs_urls)} documentation URL(s) (localhost detected)...")
        docs_content = "\n---\n".join(_fetch_url(u) for u in docs_urls)

    # TODO: Group questions based on the docs pages they require and use the batch API
    # to ask multiple questions at once with the same context.

    answer_results: list[QuestionResult] = []
    progress_bar = tqdm.tqdm(
        total=num_questions * num_candidates,
        disable=not sys.stdout.isatty(),
        desc=f"Evaluating {model} ({'with' if with_docs else 'without'} docs)",
        unit="question",
    )
    score_so_far = 0
    total = 0
    for question_index, question in enumerate(questions, 1):
        runs: list[Letter | None] = []
        for candidate_index in range(num_candidates):
            selected = ask_question(
                client=client,
                question=question,
                question_index=question_index,
                num_questions=num_questions,
                candidate_index=candidate_index if num_candidates > 1 else None,
                num_candidates=num_candidates if num_candidates > 1 else None,
                with_docs=with_docs,
                model=model,
                docs_urls=docs_urls,
                docs_content=docs_content,
                ollama_url=ollama_url,
                # Adding this gives the LLM the ability to consult URLs given in
                # the prompt.
                # Not needed (or wanted) when docs are already inlined or for
                # Ollama/Claude CLI models.
                tools=(
                    [types.Tool(url_context=types.UrlContext())]
                    if with_docs
                    and docs_content is None
                    and not _is_ollama_model(model)
                    and not _is_claude_cli_model(model)
                    else None
                ),
            )
            runs.append(selected)
            progress_bar.update(1)
            score_so_far += 1 if selected == question.answer else 0
            total += 1
            progress_bar.set_postfix(score=f"{score_so_far / total:.2%}")
        answer_results.append(QuestionResult(expected=question.answer, runs=tuple(runs)))
    logger.info(
        f"Final score for {model} ({'with' if with_docs else 'without'} docs): {score_so_far}/{total} = {score_so_far / total:.2%}"
    )
    return EvaluationResult(answers=tuple(answer_results), num_candidates=num_candidates)


def load_questions(questions_path: Path) -> list[Question]:
    questions = []

    for q in yaml.safe_load(questions_path.read_text()):
        q = Question(**q)
        if q.skills:
            q.skills_dir = questions_path.parent.parent
        questions.append(q)

    return questions


def ask_question(
    client: genai.Client | None,
    question: Question,
    with_docs: bool,
    model: str,
    docs_urls: list[str] | None,
    tools: list[types.Tool | Callable] | None,
    docs_content: str | None = None,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    question_index: int | None = None,
    num_questions: int | None = None,
    candidate_index: int | None = None,
    num_candidates: int | None = None,
) -> Letter | None:
    """Asks a question to the LLM and returns the selected answer letter, or None if unparsable."""
    q_prefix = (
        f"Question {question_index}/{num_questions}: "
        if question_index is not None and num_questions is not None
        else ""
    )
    if candidate_index is not None and num_candidates is not None:
        q_prefix += f"[candidate {candidate_index + 1}/{num_candidates}] "

    if model == DUMMY_MODEL:
        dummy_answer = random.choice(list(question.options.keys()))
        logger.info(f"{q_prefix}Correct answer: {question.answer}, dummy answer: {dummy_answer}")
        return dummy_answer

    if _is_ollama_model(model) or _is_claude_cli_model(model):
        system_prompt, prompt = make_prompt(
            question, with_docs=with_docs, docs_urls=docs_urls, docs_content=docs_content
        )
        logger.debug(f"{q_prefix}Prompt sent to LLM: [magenta]{prompt}")
        # Use web_fetch tool when docs are given as URLs (not already inlined via --docs-file).
        use_web_fetch = with_docs and bool(docs_urls) and docs_content is None
        agent_answer, skill_invoked = _get_agent_answer_claude_cli(
            model,
            system_prompt,
            prompt,
            with_docs=with_docs,
            use_web_fetch=use_web_fetch,
            docs_urls=docs_urls,
            expected_skills=question.skills,
            skills_dir=question.skills_dir,
        )
        if not agent_answer:
            logger.error(f"{q_prefix}LLM's answer couldn't be parsed!")
            return None
        if question.skills and with_docs and not skill_invoked:
            logger.warning(f"{q_prefix}Expected skill '{question.skills}' was NOT invoked!")
        logger.info(
            f"{q_prefix}Correct answer: {question.answer}, LLM's answer: {agent_answer.answer}"
            + (f", skill '{question.skills}' invoked: {skill_invoked}" if question.skills else "")
        )
        if agent_answer.justification:
            logger.debug(f"{q_prefix}LLM's justification: {agent_answer.justification}")
        return agent_answer.answer

    # TODO: For a lot of the models available through Google AI Studio, they can't
    # use tools to fetch the docs content. It *might* be worthwhile to actually fetch,
    # parse, and embed the page into the prompt ourselves for those models to get results with mode models.
    # OR, we could switch to something like VertexAI and see if we have access to more models with tool use
    # there.

    assert client is not None
    _, prompt = make_prompt(
        question, with_docs=with_docs, docs_urls=docs_urls, docs_content=docs_content
    )
    logger.debug(f"{q_prefix}Prompt sent to LLM: {prompt}")
    # TODO: use https://ai.google.dev/api/batch-api instead of single requests.
    agent_answer = get_agent_answer(client, model, tools, prompt)
    if not agent_answer:
        logger.error(f"{q_prefix}LLM's answer couldn't be parsed!")
        return None
    logger.info(
        f"{q_prefix}Correct answer: {question.answer}, LLM's answer: {agent_answer.answer}"
    )
    if agent_answer.justification:
        logger.debug(f"{q_prefix}LLM's justification: {agent_answer.justification}")
    return agent_answer.answer


def get_agent_answer(
    client: genai.Client,
    model: str,
    tools: list[types.Tool | Callable] | None,
    prompt: str,
) -> Response | None:
    api_response = client.models.generate_content(
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
    assert api_response.candidates

    if len(api_response.candidates) > 1:
        warnings.warn(
            f"Response contained {len(api_response.candidates)} candidates, but only the first one will be used!"
        )
    for candidate in api_response.candidates:
        if (consulted_urls := candidate.url_context_metadata) and consulted_urls.url_metadata:
            logger.info(
                f"The LLM consulted {len(consulted_urls.url_metadata)} web pages to answer the question."
            )
            logger.debug(
                "Consulted URLs: "
                + "\n".join(
                    "- "
                    + (url_meta.retrieved_url or "n/a")
                    + " "
                    + (
                        "(success)"
                        if url_meta.url_retrieval_status
                        == types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS
                        else "(failure)"
                    )
                    for url_meta in consulted_urls.url_metadata
                )
            )

    if isinstance(api_response.parsed, Response):
        logger.debug("LLM output was correctly parsed by the client library.")
        return api_response.parsed

    assert api_response.text is not None
    try:
        return Response.model_validate_json(api_response.text)
    except pydantic.ValidationError:
        return parse_response_fallback(api_response.text)


def parse_response_fallback(response_str: str) -> Response | None:
    """Parses a `Response` object from the API output when the LLM doesn't follow the requested response schema.

    >>> parse_response_fallback('A')
    Response(answer='A', justification='')

    >>> parse_response_fallback('This description perfectly matches the requirement for storing "temporary model checkpoints". Answer: A')
    Response(answer='A', justification='This description perfectly matches the requirement for storing "temporary model checkpoints". Answer:')
    """
    if response_str.startswith("```json") and response_str.endswith("```"):
        response_str = response_str.removeprefix("```json").removesuffix("```").strip()
        try:
            return Response.model_validate_json(response_str)
        except pydantic.ValidationError:
            pass
    try:
        last_line = response_str.strip().splitlines()[-1].strip().removesuffix(".")
        # todo: if the last word is a single letter (capitalised or not), use this as the LLM's guess.
        last_char = re.search(r"\b([A-Ea-e])\b", last_line)
        if not last_char:
            logger.error(f"Invalid response: {response_str}")
            return None
        last_char = last_char.group(0)
        last_char = last_char.upper()
        assert last_char in ("A", "B", "C", "D", "E")
        return Response(
            answer=last_char,
            justification=response_str.strip().removesuffix(last_char).strip(),
        )
    except ValueError:
        logger.error(f"Invalid response: {response_str}")
        return None


def make_prompt(
    question: Question,
    with_docs: bool,
    docs_urls: list[str] | None = None,
    docs_content: str | None = None,
) -> tuple[str, str]:
    if with_docs and docs_content:
        preamble = f"Based on the following documentation:\n\n{docs_content}\n\n"
    elif with_docs and docs_urls:
        preamble = "Based on this documentation: " + ", ".join(docs_urls) + ",\n"
    else:
        preamble = ""
    return (
        preamble + "Select the correct answer's letter to the question asked",
        (
            f"{question.question}\n"
            + "\n".join(f"- {letter}: {answer}" for letter, answer in question.options.items())
            + "\n"
        ),
    )


def _serialize_evaluation_result(result: EvaluationResult) -> dict:
    answers_out = []
    for qr in result.answers:
        if result.num_candidates == 1:
            selected = qr.runs[0] if qr.runs else None
            answers_out.append(
                {
                    "expected": qr.expected,
                    "selected": selected,
                    "correct": (selected == qr.expected) if selected is not None else None,
                }
            )
        else:
            answers_out.append(
                {
                    "expected": qr.expected,
                    "pass_rate": qr.pass_rate,
                    "selected": list(qr.runs),
                }
            )
    out: dict = {
        "num_questions": result.num_questions,
        "correct_answers": result.correct_answers,
        "invalid_answers": result.invalid_answers,
        "score": result.score,
        "answers": answers_out,
    }
    if result.num_candidates > 1:
        out["num_candidates"] = result.num_candidates
        out["score_std"] = result.score_std
    return out


def _add_evaluate_args(p: argparse.ArgumentParser, questions_required: bool = True) -> None:
    p.add_argument("--questions", type=Path, required=questions_required)
    p.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help='LLM model to use (e.g. "gemini-2.5-flash", "ollama:gemma4:31b", "claude:claude-sonnet-4-6"). Use "test:dummy" for random answers without any API calls.',
    )
    p.add_argument(
        "--docs-url",
        nargs="*",
        default=None,
        help="URLs to documentation pages that will be used as context for all questions.",
    )
    p.add_argument(
        "--docs-file",
        nargs="*",
        type=Path,
        default=None,
        help="Local file paths whose content will be inlined into the prompt as documentation context.",
    )
    p.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_DEFAULT_URL,
        help=f"Base URL of the Ollama server (default: {OLLAMA_DEFAULT_URL}). Only used with 'ollama:' models.",
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to ask each question. Scores are averaged across candidates. (default: 1)",
    )
    p.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format. 'json' emits a machine-readable JSON object suitable for CI pipelines.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count")
    # Evaluate args on the top-level parser so `docmetrics` (no subcommand) works.
    _add_evaluate_args(parser, questions_required=False)

    subparsers = parser.add_subparsers(dest="subcommand")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate an LLM on documentation questions (default when no subcommand is given).",
    )
    evaluate_parser.add_argument("-v", "--verbose", action="count")
    _add_evaluate_args(evaluate_parser)

    quiz_parser = subparsers.add_parser(
        "quiz",
        help="Take the quiz interactively in the terminal.",
    )
    quiz_parser.add_argument("-v", "--verbose", action="count")
    quiz_parser.add_argument("--questions", type=Path, required=True)

    args = parser.parse_args(argv)
    verbose: int = args.verbose or 0

    logging.basicConfig(
        level=logging.DEBUG if verbose >= 3 else logging.INFO if verbose == 2 else logging.WARNING,
        handlers=[
            rich.logging.RichHandler(markup=True, console=rich.console.Console(stderr=True))
        ],
        format="%(message)s",
    )
    logger.setLevel(  # this logger specifically.
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    )

    if args.subcommand == "quiz":
        from docmetrics.quiz import run_quiz

        questions = load_questions(questions_path=args.questions)
        run_quiz(questions)
        return

    # Default: evaluate (subcommand is None or "evaluate")
    if args.questions is None:
        parser.error("the following arguments are required: --questions")

    questions = load_questions(questions_path=args.questions)
    model: str = args.model
    docs_urls: list[str] | None = args.docs_url if args.docs_url else None
    docs_files: list[Path] | None = args.docs_file if args.docs_file else None
    ollama_url: str = args.ollama_url
    num_candidates: int = args.num_candidates
    if ollama_url:
        os.environ["OLLAMA_HOST"] = ollama_url
    if docs_urls and docs_files:
        parser.error("--docs-url and --docs-file are mutually exclusive.")

    if docs_urls or docs_files:
        result_with_docs = evaluate_llm(
            questions,
            with_docs=True,
            model=model,
            docs_urls=docs_urls,
            docs_files=docs_files,
            ollama_url=ollama_url,
            num_candidates=num_candidates,
        )
    else:
        result_with_docs = None
    result_without_docs = evaluate_llm(
        questions,
        with_docs=False,
        model=model,
        ollama_url=ollama_url,
        num_candidates=num_candidates,
    )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "questions": [{"question": q.question} for q in questions],
                    "without_docs": _serialize_evaluation_result(result_without_docs),
                    "with_docs": (
                        _serialize_evaluation_result(result_with_docs)
                        if result_with_docs
                        else None
                    ),
                }
            )
        )
    else:
        print(
            f"Without context: {result_without_docs.score:.1%} ± {result_without_docs.score_std:.1%}"
        )
        if result_with_docs:
            print(
                f"With {'docs URL' if docs_urls else 'docs file'}: {result_with_docs.score:.1%} ± {result_with_docs.score_std:.1%}"
            )


if __name__ == "__main__":
    main()
