#!/bin/bash
set -euo pipefail

# Concurrent JIT cache stress across many ranks (report 4.7).
#
#   LAIF_CACHE_MODE=lustre  shared Lustre cache -- reproduces the corruption
#   LAIF_CACHE_MODE=tmp     per-node /tmp       -- validates the fix (default)
#
# The reported failure needs 64+ ranks, i.e. NODES>=8 at 8 ranks per node.
#
# Usage:
#   NODES=8 LAIF_CACHE_MODE=lustre ./templates/cache_stress.sh <container.sif>

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES="${NODES:-8}"
NTASKS_PER_NODE=8
GPUS_PER_NODE=8
CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
# Compilation is slow on a cold cache: apex alone took ~66 s per rank in Phase 1, and this
# builds several Inductor kernels on top of that.
TIME_LIMIT="${TIME_LIMIT:-00:40:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-1}"
ENABLE_LUMI_HSN="${ENABLE_LUMI_HSN:-0}"
# Off: the mask list cannot be satisfied by a login-node srun on this project.
# See docs/PHASE2_RESULTS.md.
ENABLE_LUMI_CPU_MASKS="${ENABLE_LUMI_CPU_MASKS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run jit-cache --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
