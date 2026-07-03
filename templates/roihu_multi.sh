#!/bin/bash
set -euo pipefail

# Multi-node, all GPUs per node, 1 rank per GPU.
# Usage:
#   export GPUS_PER_NODE=4   # set to match actual node GPU count
#   export NODES=2
#   ./templates/roihu_multi.sh <container.sif> -- bench/run jax-multi --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NODES="${NODES:-2}"
NTASKS_PER_NODE="${GPUS_PER_NODE}"
CPUS_PER_TASK="${CPUS_PER_TASK:-14}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
USE_CUDA_VISIBLE_DEVICES="${USE_CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/roihu_common.sh"

roihu_init
BENCH_CMD=(bench/run jax-multi --out "${RESULTS_JSON}")
roihu_override_bench_cmd "$@"
roihu_log_env
roihu_exec
