#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Same 32 random prompts in each of 12 configurations: 384 measured completions.
# Chunked prefill and prefix caching are disabled by the benchmark.
# max-tokens is a safety ceiling; inspect the capped column, not just mean length.
# Override any option by appending it, e.g. --no-show-answers or --system-prompt "".
python -u "${SCRIPT_DIR}/benchmark_forelen_chat.py" \
    --input-file "${PROJECT_ROOT}/data/forelen.csv" \
    --prompt-column user_prompt_content \
    --sample-size 32 \
    --sample-seed 42 \
    --batch-sizes 1 2 4 8 16 32 \
    --model /home/liuhaiyang/Qwen36_27B_260627/modeldownload \
    --devices "${ASCEND_RT_VISIBLE_DEVICES}" \
    --tensor-parallel-size 4 \
    --distributed-executor-backend auto \
    --block-size 128 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --max-tokens 8192 \
    --temperature 1.0 \
    --top-p 1.0 \
    --top-k -1 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --frequency-penalty 0.0 \
    --repetition-penalty 1.0 \
    --seed 42 \
    --no-trust-remote-code \
    --enforce-eager \
    --warmup \
    --show-answers \
    "$@"
