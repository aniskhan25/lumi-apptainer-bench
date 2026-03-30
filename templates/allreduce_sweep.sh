#!/bin/bash
set -euo pipefail

# Multi-node all-reduce sweep.
# Usage:
#   ./templates/allreduce_sweep.sh <container.sif> -- bench/run multi --allreduce --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES="${NODES:-2}"
NTASKS_PER_NODE=8
GPUS_PER_NODE=8
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run multi --allreduce --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
