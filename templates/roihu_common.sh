#!/bin/bash

set -euo pipefail

# Wrapper templates must set these before calling roihu_init:
# NODES, NTASKS_PER_NODE, GPUS_PER_NODE, CPUS_PER_TASK, TIME_LIMIT.
require_template_config() {
  : "${NODES:?set NODES before calling roihu_init}"
  : "${NTASKS_PER_NODE:?set NTASKS_PER_NODE before calling roihu_init}"
  : "${GPUS_PER_NODE:?set GPUS_PER_NODE before calling roihu_init}"
  : "${CPUS_PER_TASK:?set CPUS_PER_TASK before calling roihu_init}"
  : "${TIME_LIMIT:?set TIME_LIMIT before calling roihu_init}"
}

roihu_init() {
  require_template_config
  PROJECT_NAME="${PROJECT_NAME:?set PROJECT_NAME (e.g. project_2014553)}"
  PARTITION="${PARTITION:?set PARTITION before calling roihu_init}"
  ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

  SCRATCH_ROOT="/scratch/${PROJECT_NAME}"
  RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH_ROOT}/${USER}/bench_results}"

  # Lmod is not active in non-login shells; source the init script explicitly.
  source /usr/share/lmod/lmod/init/bash
  export MODULEPATH=/appl/modulefiles/manual/aida/aarch64

  # python-jax sets $SIF and APPTAINER_NV=true.
  module load python-jax

  # csc-common-bind is a standalone binary; no module needed.
  CSC_BIND=$(/appl/soft/manual/general/aarch64/csc-tools/bin/csc-common-bind)

  RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"
  RESULTS_JSON="${RESULTS_DIR}/results.json"
  LOG_DIR="${RESULTS_DIR}/logs"

  mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

  export BENCH_CONTAINER_IMAGE="${SIF}"
  export BENCH_NODES="${NODES}"
  export BENCH_NTASKS_PER_NODE="${NTASKS_PER_NODE}"
  export BENCH_GPUS_PER_NODE="${GPUS_PER_NODE}"
  export BENCH_RESULTS_DIR="${RESULTS_DIR}"

  # Per-rank GPU assignment: each srun task sees exactly one GPU.
  GPU_WRAPPER=()
  if [[ "${USE_CUDA_VISIBLE_DEVICES:-0}" == "1" ]]; then
    GPU_WRAPPER=(bash -c 'export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}; exec "$@"' --)
  fi

  SRUN_BASE=(
    srun
    --partition="${PARTITION}"
    --account="${ACCOUNT}"
    --nodes="${NODES}"
    --ntasks-per-node="${NTASKS_PER_NODE}"
    --gres="gpu:gh200:${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --time="${TIME_LIMIT}"
  )
  if [[ -n "${NODELIST:-}" ]]; then
    SRUN_BASE+=(--nodelist="${NODELIST}")
  fi
  if [[ -n "${EXCLUDE_NODES:-}" ]]; then
    SRUN_BASE+=(--exclude="${EXCLUDE_NODES}")
  fi
}

roihu_override_bench_cmd() {
  if [[ "$#" -eq 0 ]]; then
    return
  fi
  if [[ "$1" == "--" ]]; then
    shift
  fi
  if [[ "$#" -gt 0 ]]; then
    BENCH_CMD=("$@")
  fi
}

roihu_log_env() {
  {
    echo "run_id=${RUN_ID}"
    echo "partition=${PARTITION}"
    echo "account=${ACCOUNT}"
    echo "nodes=${NODES}"
    echo "ntasks_per_node=${NTASKS_PER_NODE}"
    echo "gpus_per_node=${GPUS_PER_NODE}"
    echo "cpus_per_task=${CPUS_PER_TASK}"
    echo "jax_sif=${SIF}"
    echo "bench_cmd=${BENCH_CMD[*]}"
  } | tee "${LOG_DIR}/run_env.txt"
}

roihu_exec() {
  "${SRUN_BASE[@]}" "${GPU_WRAPPER[@]}" \
    apptainer exec --bind="${CSC_BIND}" "${SIF}" \
    "${BENCH_CMD[@]}"
}
