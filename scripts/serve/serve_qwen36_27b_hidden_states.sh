#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /home/liuhaiyang/Qwen36_27B_260627/modeldownload \
    --served-model-name "qwen3.6-27b" \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mp \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --block-size 128 \
    --enforce-eager \
    --trust-remote-code \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --generation-config vllm \
    --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[64]}}}' \
    --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/home/laiwenhao/vllm-predictor/outputs"}}'
