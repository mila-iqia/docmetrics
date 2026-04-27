#!/bin/bash
MODEL="ollama:gemma4:e2b"
#MODEL="ollama:gpt-oss:20b"
#QUIZ="../mila-docs/quiz.yaml"
QUIZ=$1
set -e
set -x
echo "Using quiz at $QUIZ"
# Results dir should be based on the name of the quiz file, for example quiz_v2.yaml -> results/quiz_v2/{model}
RESULTS_DIR="results/$(basename $QUIZ .yaml)/$(echo $MODEL | tr ':' '_')"
echo "Results will be saved to $RESULTS_DIR"

COMMON_ARGS="--ollama-url=dw-a002:11434 --num-candidates=3 --output-format=json"
mkdir -p $RESULTS_DIR
uv run docmetrics $QUIZ --model=$MODEL $COMMON_ARGS > $RESULTS_DIR/without_docs.json
uv run docmetrics $QUIZ --model=$MODEL $COMMON_ARGS --docs-url https://docs.mila.quebec > $RESULTS_DIR/with_docs_url.json
uv run docmetrics $QUIZ --model=$MODEL $COMMON_ARGS --docs-file compressed_mila_docs.md > $RESULTS_DIR/with_compact_docs.json
#uv run docmetrics $QUIZ --model=$MODEL $COMMON_ARGS --docs-file entire_mila_docs.md > $RESULTS_DIR/with_docs_markdown.json
