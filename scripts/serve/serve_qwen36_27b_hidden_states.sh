#!/usr/bin/env bash
set -euo pipefail

# Four-card Qwen3.6-27B service for prompt/prefill hidden-state extraction.
# Defaults are deliberately conservative for four 32 GiB Ascend 910B4 NPUs.
# Every request writes one safetensors file under OUTPUT_DIR.

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6}"
SERVICE_HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PROJECT_DIR="${PROJECT_DIR:-/home/laiwenhao/vllm-predictor}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/hidden_states}"

TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"

# Qwen3.6-27B has 64 text layers. Layer id 64 is the output of the final
# transformer layer before the model's output normalization.
EXTRACT_LAYER_IDS="${EXTRACT_LAYER_IDS:-[64]}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH must point to the Qwen3.6-27B weights inside the container." >&2
    echo "Example: MODEL_PATH=/path/to/Qwen3.6-27B bash $0" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
if [[ ! -w "${OUTPUT_DIR}" ]]; then
    echo "ERROR: hidden-state output directory is not writable: ${OUTPUT_DIR}" >&2
    exit 1
fi

if (( MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
    echo "ERROR: with chunked prefill disabled, MAX_NUM_BATCHED_TOKENS must be" >&2
    echo "       greater than or equal to MAX_MODEL_LEN." >&2
    exit 1
fi

SPECULATIVE_CONFIG=$(printf \
    '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":%s}}}' \
    "${EXTRACT_LAYER_IDS}")

KV_TRANSFER_CONFIG=$(printf \
    '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"%s"}}' \
    "${OUTPUT_DIR}")

serve_args=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${SERVICE_HOST}"
    --port "${PORT}"
    --data-parallel-size 1
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend mp
    --block-size "${BLOCK_SIZE}"
    --no-enable-chunked-prefill
    --no-enable-prefix-caching
    --enforce-eager
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --trust-remote-code
    --generation-config vllm
    --speculative-config "${SPECULATIVE_CONFIG}"
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
)

# Set QUANTIZATION=ascend only when MODEL_PATH points to an Ascend W8A8 model.
if [[ -n "${QUANTIZATION:-}" ]]; then
    serve_args+=(--quantization "${QUANTIZATION}")
fi

echo "Starting Qwen3.6-27B hidden-state extraction service"
echo "  model:  ${MODEL_PATH}"
echo "  NPUs:   ${ASCEND_RT_VISIBLE_DEVICES} (TP=${TP_SIZE})"
echo "  layers: ${EXTRACT_LAYER_IDS}"
echo "  output: ${OUTPUT_DIR}"
echo "  API:    http://${SERVICE_HOST}:${PORT}"

exec vllm serve "${serve_args[@]}"
