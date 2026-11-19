# 📐 DocMetrics

**Test-driven development, but for documentation.**

[![PyPI](https://img.shields.io/pypi/v/docmetrics)](https://pypi.org/project/docmetrics/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/)
[![CI](https://github.com/mila-iqia/docmetrics/actions/workflows/build.yaml/badge.svg)](https://github.com/mila-iqia/docmetrics/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/mila-iqia/docmetrics/graph/badge.svg)](https://codecov.io/gh/mila-iqia/docmetrics)

---

DocMetrics quantifies how **useful** your documentation is — not how readable, not how long, but how much it actually helps someone accomplish something.

Code coverage tells you how much of your code is exercised by tests. DocMetrics does the same thing for documentation: you write questions your users *should* be able to answer after reading your docs, and DocMetrics measures whether the docs actually help answer them.

## The idea

Good documentation is hard to measure. Word count, readability scores, and page views tell you almost nothing about whether your docs are actually *useful*. DocMetrics takes a different approach:

1. **You write multiple-choice questions** that capture what your users should learn from the documentation — installation steps, key concepts, configuration options, anything.
2. **An LLM answers those questions twice**: once *without* your docs, and once *with* your docs as context.
3. **The difference in score is your documentation's usefulness** — a concrete number showing how much your docs help beyond what's already common knowledge.

This is the equivalent of writing tests before writing code. You define what your documentation should teach, then measure whether it actually does. If the LLM can answer your questions just as well *without* the docs, that documentation isn't adding value. If the score jumps when docs are provided, you know the content is genuinely useful.

This works for any kind of written information: websites, README files, API references, agent skill descriptions, runbooks, onboarding guides — anything you can point a URL at or pass as a file.

## PR comments

When used as a GitHub Action, DocMetrics posts a comment on every pull request showing how the documentation score changed — just like a code coverage bot.

Here's what a real PR comment looks like ([PR #17](https://github.com/mila-iqia/docmetrics/pull/17#issuecomment-4157788665)):

> ## DocMetrics Report
>
> > Merging this PR into `master` will **not change** the with-docs score.
>
> | | Without docs | With docs | Δ&nbsp;(docs&nbsp;−&nbsp;no&nbsp;docs) |
> |:---|:---:|:---:|:---:|
> | **This PR** | 100% (1/1) | 100% (1/1) | +0pp |
> | **Base (`master` *(cached)*)** | 100% (1/1) | 100% (1/1) | +0pp |
> | **Change** | +0pp | +0pp | — |
>
> *Model: `ollama:gpt-oss:120b` · Questions: `sample_questions.yaml`*
> *Comparing: PR documentation (current) vs. base documentation*

And when something changes, the comment highlights exactly which questions flipped ([PR #11](https://github.com/mila-iqia/docmetrics/pull/11#issuecomment-4113119476)):

> ## DocMetrics Report
>
> > Merging this PR into `master` will **decrease** the with-docs score by `-100pp` :arrow_down:.
>
> | | Without docs | With docs | Δ&nbsp;(docs&nbsp;−&nbsp;no&nbsp;docs) |
> |:---|:---:|:---:|:---:|
> | **This PR** | 100% (1/1) | 0% (0/1) | -100pp |
> | **Base (`master` *(cached)*)** | 100% (1/1) | 100% (1/1) | +0pp |
> | **Change** | +0pp | -100pp | — |
>
> <details>
> <summary>📋 1 question(s) with changed results</summary>
>
> | # | Question | Without docs | With docs |
> |:---|:---|:---:|:---:|
> | 1 | What does DocMetrics measure? | ✅ → ✅ | ✅ → ❓ |
>
> </details>
>
> *Model: `gemini-2.5-flash` · Questions: `sample_questions.yaml`*
> *Comparing: PR documentation (current) vs. base documentation*

The comment updates in place on each push, compares against the base branch (with caching to avoid redundant LLM calls), and expands per-question diffs so you can see exactly which questions were affected.

## Getting started

### Installation

```bash
pip install docmetrics
# or
uv tool install docmetrics
```

### Write your questions

Create a YAML file with multiple-choice questions that your documentation should help answer:

```yaml
- question: "What does DocMetrics measure?"
  options:
    A: Computes the clarity of documentation using NLP techniques
    B: Documentation usefulness by comparing LLM accuracy with and without docs as context
    C: Documentation usefulness by comparing human ratings to LLM ratings
    D: Computes a readability score for documentation using LLMs
  answer: B

- question: "How do you install docmetrics using uv?"
  options:
    A: "uv add docmetrics"
    B: "uv run pip install docmetrics"
    C: "uv tool install docmetrics"
    D: "uv sync docmetrics"
  answer: C
```

### Run it

```bash
# Evaluate with documentation URLs
docmetrics --questions questions.yaml --docs-url https://your-docs-site.com

# Evaluate with local files
docmetrics --questions questions.yaml --docs-file README.md CONTRIBUTING.md

# Use a different model
docmetrics --questions questions.yaml --model ollama:llama3 --docs-url https://your-docs-site.com

# Output as JSON (for CI pipelines)
docmetrics --questions questions.yaml --docs-url https://your-docs-site.com --output-format json
```

### GitHub Action

Add DocMetrics to your CI pipeline to track documentation quality on every PR:

```yaml
- uses: mila-iqia/docmetrics@v0
  with:
    questions-file: questions.yaml
    docs-url: https://your-docs-site.com
    gemini-api-key: ${{ secrets.GEMINI_API_KEY }}
```

The action will:
- Evaluate your docs on the PR branch and the base branch
- Cache base branch results to minimize LLM API calls
- Post (or update) a PR comment with the score comparison

See [`action.yml`](action.yml) for all available inputs and outputs.

## How it works

```
questions.yaml ──┐
                 ├── evaluate(without docs) ──→ baseline score
                 ├── evaluate(with docs)    ──→ docs score
                 │
                 └── usefulness = docs score − baseline score
```

Each question is sent to the LLM as a multiple-choice prompt. When evaluating *with* docs, the LLM is given access to the documentation URLs (via tool use) or inline file contents. The score is simply the fraction of questions answered correctly, and the **usefulness** of the documentation is how much that score improves when docs are available.

## License

This project is developed at [Mila – Quebec AI Institute](https://mila.quebec/).
