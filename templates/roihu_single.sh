#!/bin/bash
set -euo pipefail

# Single node, all GPUs, 1 rank per GPU.
# Usage:
#   export GPUS_PER_NODE=4   # set to match actual node GPU count
#   ./templates/roihu_single.sh <container.sif> -- bench/run jax-single --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NODES=1
NTASKS_PER_NODE="${GPUS_PER_NODE}"
CPUS_PER_TASK="${CPUS_PER_TASK:-14}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
USE_CUDA_VISIBLE_DEVICES="${USE_CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/roihu_common.sh"

roihu_init
BENCH_CMD=(bench/run jax-single --out "${RESULTS_JSON}")
roihu_override_bench_cmd "$@"
roihu_log_env
roihu_exec
