#!/bin/bash
set -euo pipefail

# Single node, 8 GPUs, 8 ranks (1 rank per GPU).
# Usage:
#   ./templates/single_8g_8r.sh <container.sif> -- bench/run single --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES=1
NTASKS_PER_NODE=8
GPUS_PER_NODE=8
CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-1}"
ENABLE_LUMI_HSN="${ENABLE_LUMI_HSN:-0}"
ENABLE_LUMI_CPU_MASKS="${ENABLE_LUMI_CPU_MASKS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run single --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
