#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/data/run_extract_forelen_online.sh \
#   http://localhost:8001 /data/forelen.csv /data/prefill_states /data/forelen_online [--thinking]
# The third argument must be the server's prefill_hidden_states_dir, locally mounted.
if (( $# < 4 )); then
    echo "Usage: $0 URL INPUT_CSV SERVER_HIDDEN_DIR OUTPUT_DIR [extra CLI arguments]" >&2
    exit 2
fi
base_url="$1"
input_file="$2"
hidden_dir="$3"
output_dir="$4"
shift 4
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "$script_dir/extract_forelen_online.py" \
    --base-url "$base_url" \
    --input-file "$input_file" \
    --hidden-states-dir "$hidden_dir" \
    --output-dir "$output_dir" \
    --shard-size 2048 \
    --batch-size 8 \
    --max-tokens 8192 \
    --no-thinking \
    "$@"
