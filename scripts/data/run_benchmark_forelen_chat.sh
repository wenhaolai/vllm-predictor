#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Start vllm serve separately on port 8001 before running this client.
# The served model name is discovered from /v1/models unless --model is appended.
# Same 32 random prompts are reused in all 12 thinking/concurrency configurations.
python -u "${SCRIPT_DIR}/benchmark_forelen_chat.py" \
    --base-url http://localhost:8001/v1 \
    --api-key "${VLLM_API_KEY:-EMPTY}" \
    --request-timeout 3600 \
    --input-file "${PROJECT_ROOT}/data/forelen.csv" \
    --prompt-column user_prompt_content \
    --sample-size 32 \
    --sample-seed 42 \
    --batch-sizes 1 2 4 8 16 32 \
    --max-tokens 8192 \
    --temperature 1.0 \
    --top-p 1.0 \
    --top-k -1 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --frequency-penalty 0.0 \
    --repetition-penalty 1.0 \
    --seed 42 \
    --warmup \
    --show-answers \
    "$@"
