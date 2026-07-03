#!/bin/bash
set -euo pipefail

# Multi-node, all GPUs per node, 1 rank per GPU.
# Usage:
#   export PROJECT_NAME=project_2014553
#   export NODES=2
#   ./templates/roihu_multi.sh -- bench/run jax-multi --out results.json

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NODES="${NODES:-2}"
NTASKS_PER_NODE="${GPUS_PER_NODE}"
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
PARTITION="${PARTITION:-gpumedium}"
USE_CUDA_VISIBLE_DEVICES="${USE_CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/roihu_common.sh"

roihu_init
BENCH_CMD=(bench/run jax-multi --out "${RESULTS_JSON}")
roihu_override_bench_cmd "$@"
roihu_log_env
roihu_exec
