#!/bin/bash
set -euo pipefail

# Single node, 8 GPUs, 16 ranks (2 ranks per GPU).
# Usage:
#   ./templates/single_8g_16r.sh <container.sif> -- bench/run single --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES=1
NTASKS_PER_NODE=16
GPUS_PER_NODE=8
CPUS_PER_TASK="${CPUS_PER_TASK:-3}"
TIME_LIMIT="${TIME_LIMIT:-00:40:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run single --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
