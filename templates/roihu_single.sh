#!/bin/bash
set -euo pipefail

# Single node, all GPUs, 1 rank per GPU.
# Usage:
#   export PROJECT_NAME=project_2014553
#   ./templates/roihu_single.sh -- bench/run jax-single --out results.json

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NODES=1
NTASKS_PER_NODE="${GPUS_PER_NODE}"
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
PARTITION="${PARTITION:-gputest}"
USE_CUDA_VISIBLE_DEVICES="${USE_CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/roihu_common.sh"

roihu_init
BENCH_CMD=(bench/run jax-single --out "${RESULTS_JSON}")
roihu_override_bench_cmd "$@"
roihu_log_env
roihu_exec
