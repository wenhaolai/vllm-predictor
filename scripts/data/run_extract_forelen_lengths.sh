#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

# All rows are processed by default; add --limit N for a small run.
# Use a fresh output directory so previous 16384-token results are not resumed.
# Sampling matches SamplingParams(max_tokens=2048, temperature=1.0):
# no top-p/top-k filtering, default penalties, and no fixed sampling seed.
# Add --seed 42 if reproducible sampling is needed for a new experiment.
python /home/laiwenhao/vllm-predictor/scripts/data/extract_forelen_lengths.py \
    --input-file /home/laiwenhao/vllm-predictor/data/forelen.csv \
    --prompt-column user_prompt_content \
    --output-dir /home/laiwenhao/vllm-predictor/data/forelen_lengths_2048 \
    --model /home/liuhaiyang/Qwen36_27B_260627/modeldownload \
    --devices 0,1,2,3 \
    --tensor-parallel-size 4 \
    --distributed-executor-backend auto \
    --block-size 128 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --batch-size 8 \
    --max-tokens 2048 \
    --temperature 1.0 \
    --top-p 1.0 \
    --top-k -1 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --frequency-penalty 0.0 \
    --repetition-penalty 1.0 \
    --no-trust-remote-code \
    --enforce-eager
