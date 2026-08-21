#!/bin/bash
set -euo pipefail

# All-to-all sweep. GROUP_SIZE selects the communicator topology, which is the axis that
# matters on LUMI-G and corresponds directly to an expert-parallelism degree:
#
#   GROUP_SIZE=8   one node, XGMI            (EP=8,  the reference MoE configuration)
#   GROUP_SIZE=16  two nodes, Slingshot      (EP=16, reported ~60% slower than EP=8)
#   GROUP_SIZE=32  four nodes, Slingshot     (EP=32, reported to fail during init)
#
# NODES must be at least GROUP_SIZE/8.
#
# Usage:
#   NODES=2 GROUP_SIZE=16 ./templates/alltoall_sweep.sh <container.sif>
#   ./templates/alltoall_sweep.sh <container.sif> -- bench/run alltoall --group-size 8 --out r.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES="${NODES:-2}"
NTASKS_PER_NODE=8
GPUS_PER_NODE=8
CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-1}"
# Off by default -- see the note in multi_ng_8rpn.sh. Set to 1 only for the deliberate
# fabric-tuning sweep in Phase 3.
ENABLE_LUMI_HSN="${ENABLE_LUMI_HSN:-0}"
# Off by default, matching every other template in the validation path. The LUMI NUMA
# masks assume an exclusive full node (7 cores per GPU group, 0xfe); a shared or partial
# allocation gives 4 per group (0x1E) and srun aborts the step with "CPU binding outside
# of job step allocation". With this defaulted to 1, scripts/run_validation.sh could not
# get past phase 2 (observed on job 21428142). The recorded EP=8/16/32 baselines were all
# measured with --cpu-bind=cores, so 0 is also what the gate thresholds were calibrated
# against. Set to 1 only under an sbatch allocation holding whole nodes.
ENABLE_LUMI_CPU_MASKS="${ENABLE_LUMI_CPU_MASKS:-0}"

GROUP_SIZE="${GROUP_SIZE:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run alltoall --group-size "${GROUP_SIZE}" --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
