#!/bin/bash
set -euo pipefail

# Capability probe: what the job actually loaded, and what each rank can see.
#
# The three launch arms for the ENTRYPOINT question (report 4.1):
#   USE_ROCR_VISIBLE_DEVICES=1 APPTAINER_MODE=exec  -> launcher binds GPUs (our default)
#   USE_ROCR_VISIBLE_DEVICES=0 APPTAINER_MODE=exec  -> nobody binds; expect 8 devices/rank
#   USE_ROCR_VISIBLE_DEVICES=0 APPTAINER_MODE=run   -> container entrypoint binds
#
# The last two need ROCR_USE_SLURM_LOCALID=1 for the entrypoint to act. Compare
# tests.probe.visibility.torch_device_count across the arms.
#
# Usage:
#   ./templates/probe.sh <container.sif> -- bench/run probe --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-8}"
GPUS_PER_NODE=8
CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-1}"
ENABLE_LUMI_HSN="${ENABLE_LUMI_HSN:-0}"
# Off for the probe. The LUMI GPU/CPU bind masks assume an exclusive full node (7 cores
# per GPU group); a shared or partial allocation -- which is what dev-g hands out -- makes
# srun reject them with "CPU binding outside of job step allocation". A probe measures
# device visibility, not throughput, so NUMA placement is irrelevant here.
ENABLE_LUMI_CPU_MASKS="${ENABLE_LUMI_CPU_MASKS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lumi_common.sh"

lumi_init
BENCH_CMD=(bench/run probe --out "${RESULTS_JSON}")
lumi_override_bench_cmd "$@"
lumi_log_env
lumi_exec
