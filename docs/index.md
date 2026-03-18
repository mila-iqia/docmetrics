# docmetrics



DocMetrics is a tool to analyze and measure the complexity and usefulness of documentation.

It does this using a set of multiple choice questions, that you wish your users to be able to answer after reading the documentation.

An LLM is then asked to answer these questions, with and without the documentation as part of its context. The difference in score is then used as a measure of the usefulness of the documentation.

The list of questions need to be provided to the tool by passing a `--questions` option pointing to a yaml file. See the `sample_questions.yaml` file for an example.

The choice of LLM can be configured, but defaults to Google's Gemini model. The set of tools available to the agent can also be configured.

## Installation

1. (optional) Install UV: https://docs.astral.sh/uv/getting-started/installation/

2. Install this package:

```console
uv tool install docmetrics
```

## Usage

```console
docmetrics --help
```

