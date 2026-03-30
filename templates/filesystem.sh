#!/bin/bash
set -euo pipefail

# Filesystem-focused template (binds, cache, scratch strategy).
# Usage:
#   ./templates/filesystem.sh <container.sif> -- bench/run check --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES=1
NTASKS_PER_NODE=1
GPUS_PER_NODE=0
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
TIME_LIMIT="${TIME_LIMIT:-00:10:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run check --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
