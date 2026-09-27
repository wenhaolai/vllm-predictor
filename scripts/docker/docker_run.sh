#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   IMAGE=quay.io/ascend/vllm-ascend:v0.23.0 bash scripts/docker/docker_run.sh
#
# Optional overrides:
#   NAME=vllm-predictor PROJECT_ROOT=/path/to/vllm-predictor \
#   MODEL_CACHE_DIR=/root/.cache HOST_OUTPUT_DIR=/data/hidden_states \
#   bash scripts/docker/docker_run.sh

NAME="${NAME:-vllm-predictor}"
IMAGE="${IMAGE:-}"

if [[ -z "${IMAGE}" ]]; then
    echo "ERROR: please set IMAGE to the vllm-ascend image name or image ID." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/root/.cache}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hidden_states}"

mkdir -p "${HOST_OUTPUT_DIR}"

docker run -d \
    -u 0 \
    --ipc=host \
    --shm-size=64g \
    --net=host \
    --name "${NAME}" \
    -e VLLM_USE_MODELSCOPE=True \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    -e HCCL_BUFFSIZE=512 \
    -e OMP_PROC_BIND=false \
    -e OMP_NUM_THREADS=1 \
    -e TASK_QUEUE_ENABLE=1 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v "${MODEL_CACHE_DIR}:/root/.cache" \
    -v "${HOST_OUTPUT_DIR}:/outputs/hidden_states" \
    -v "${PROJECT_ROOT}:/workspace/vllm-predictor" \
    -w /workspace/vllm-predictor \
    "${IMAGE}" \
    bash -lc "tail -f /dev/null"

echo "Container started: ${NAME}"
echo "Host hidden-state directory: ${HOST_OUTPUT_DIR}"
echo "Start the extraction service with:"
echo "  docker exec -it ${NAME} bash scripts/serve/serve_qwen36_27b_hidden_states.sh"
