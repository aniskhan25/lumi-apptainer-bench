#!/bin/bash
set -euo pipefail

# Concurrent communicator count stress (report 4.1).
#
# Creates world-spanning communicators one at a time, exercises each with an all-to-all so
# RCCL actually initialises it, and keeps them all alive. Reports how many coexist before
# anything fails, and how much device memory each one costs outside PyTorch's allocator.
#
# Per-node endpoint demand is (communicators per rank x 8 ranks), which is the quantity a
# finite per-NIC resource like a Portals table entry is consumed by.
#
# Usage:
#   NODES=16 MAX_GROUPS=32 ./templates/comm_count.sh <container.sif>

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES="${NODES:-2}"
NTASKS_PER_NODE=8
GPUS_PER_NODE=8
CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
# Each communicator costs an RCCL init, and those add up at high rank counts.
TIME_LIMIT="${TIME_LIMIT:-00:50:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-1}"
ENABLE_LUMI_HSN="${ENABLE_LUMI_HSN:-0}"
# Off: the mask list cannot be satisfied by a login-node srun on this project.
# See docs/PHASE2_RESULTS.md.
ENABLE_LUMI_CPU_MASKS="${ENABLE_LUMI_CPU_MASKS:-0}"

MAX_GROUPS="${MAX_GROUPS:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run comm-count --max-groups "${MAX_GROUPS}" --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
