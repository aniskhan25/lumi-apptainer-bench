#!/bin/bash

set -euo pipefail

# Wrapper templates must set these before calling lumi_init:
# CONTAINER_IMAGE, NODES, NTASKS_PER_NODE, GPUS_PER_NODE, CPUS_PER_TASK, TIME_LIMIT.
require_template_config() {
  : "${CONTAINER_IMAGE:?container image path required}"
  : "${NODES:?set NODES before calling lumi_init}"
  : "${NTASKS_PER_NODE:?set NTASKS_PER_NODE before calling lumi_init}"
  : "${GPUS_PER_NODE:?set GPUS_PER_NODE before calling lumi_init}"
  : "${CPUS_PER_TASK:?set CPUS_PER_TASK before calling lumi_init}"
  : "${TIME_LIMIT:?set TIME_LIMIT before calling lumi_init}"
}


LUMI_GPU_CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

resolve_apptainer_cmd() {
  APPTAINER_CMD="${APPTAINER_CMD:-apptainer}"
  if command -v "${APPTAINER_CMD}" >/dev/null 2>&1; then
    return
  fi
  if command -v singularity >/dev/null 2>&1; then
    APPTAINER_CMD="singularity"
    return
  fi
  echo "Apptainer/Singularity not found in PATH." >&2
  exit 1
}

lumi_init() {
  require_template_config
  PROJECT_NAME="${PROJECT_NAME:?set PROJECT_NAME (e.g. project_465000001)}"
  PARTITION="${PARTITION:-standard-g}"
  ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

  SCRATCH_ROOT="/scratch/${PROJECT_NAME}"
  FLASH_ROOT="/flash/${PROJECT_NAME}"
  PROJECT_ROOT="/project/${PROJECT_NAME}"
  HOME_ROOT="/users/${USER}"
  CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/${USER}/bench_cache}"
  RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH_ROOT}/${USER}/bench_results}"

  BIND_ARGS=(
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
    --bind "${FLASH_ROOT}:${FLASH_ROOT}"
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}"
    --bind "${HOME_ROOT}:${HOME_ROOT}"
  )

  resolve_apptainer_cmd

  MPI_MODE="${MPI_MODE:-host}"
  SRUN_MPI_FLAG=()
  if [[ "${MPI_MODE}" == "container" ]]; then
    SRUN_MPI_FLAG=(--mpi=pmi2)
  fi

  DIST="${DIST:-block}"
  CPU_BIND="${CPU_BIND:-cores}"
  ENABLE_LUMI_HSN="${ENABLE_LUMI_HSN:-0}"
  ENABLE_LUMI_CPU_MASKS="${ENABLE_LUMI_CPU_MASKS:-0}"
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"
  RESULTS_JSON="${RESULTS_DIR}/results.json"
  LOG_DIR="${RESULTS_DIR}/logs"

  mkdir -p "${CACHE_ROOT}" "${RESULTS_DIR}" "${LOG_DIR}"

  export BENCH_CONTAINER_IMAGE="${CONTAINER_IMAGE}"
  export BENCH_RESULTS_DIR="${RESULTS_DIR}"
  export BENCH_CACHE_ROOT="${CACHE_ROOT}"
  export BENCH_PARTITION="${PARTITION}"
  export BENCH_ACCOUNT="${ACCOUNT}"
  export BENCH_MPI_MODE="${MPI_MODE}"
  export BENCH_NODES="${NODES}"
  export BENCH_NTASKS_PER_NODE="${NTASKS_PER_NODE}"
  export BENCH_GPUS_PER_NODE="${GPUS_PER_NODE}"
  export BENCH_CPUS_PER_TASK="${CPUS_PER_TASK}"
  export BENCH_DIST="${DIST}"
  export BENCH_CPU_BIND="${CPU_BIND}"

  if [[ "${ENABLE_LUMI_HSN}" == "1" ]]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
  fi

  GPU_WRAPPER=()
  if [[ "${USE_ROCR_VISIBLE_DEVICES:-0}" == "1" ]]; then
    GPU_WRAPPER=(bash -lc 'export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}; exec "$@"' --)
  fi

  SRUN_BASE=(
    srun
    --partition="${PARTITION}"
    --account="${ACCOUNT}"
    --nodes="${NODES}"
    --ntasks-per-node="${NTASKS_PER_NODE}"
  )
  if [[ "${GPUS_PER_NODE}" -gt 0 ]]; then
    SRUN_BASE+=(--gpus-per-node="${GPUS_PER_NODE}")
  fi
  if [[ "${ENABLE_LUMI_CPU_MASKS}" == "1" ]]; then
    CPU_BIND="mask_cpu:${CPU_BIND_MASKS:-${LUMI_GPU_CPU_BIND_MASKS}}"
  fi
  SRUN_BASE+=(
    --cpus-per-task="${CPUS_PER_TASK}"
    --distribution="${DIST}"
    --cpu-bind="${CPU_BIND}"
    "${SRUN_MPI_FLAG[@]}"
    --time="${TIME_LIMIT}"
  )
}

lumi_override_bench_cmd() {
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

lumi_log_env() {
  {
    echo "run_id=${RUN_ID}"
    echo "container_image=${CONTAINER_IMAGE}"
    echo "partition=${PARTITION}"
    echo "account=${ACCOUNT}"
    echo "mpi_mode=${MPI_MODE}"
    echo "nodes=${NODES}"
    echo "ntasks_per_node=${NTASKS_PER_NODE}"
    echo "gpus_per_node=${GPUS_PER_NODE}"
    echo "cpus_per_task=${CPUS_PER_TASK}"
    echo "distribution=${DIST}"
    echo "cpu_bind=${CPU_BIND}"
    echo "time_limit=${TIME_LIMIT}"
    if [[ "${ENABLE_LUMI_HSN}" == "1" ]]; then
      echo "nccl_socket_ifname=${NCCL_SOCKET_IFNAME}"
      echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL}"
    fi
    echo "bench_cmd=${BENCH_CMD[*]}"
    srun --version || true
  } | tee "${LOG_DIR}/run_env.txt"
}

lumi_exec() {
  "${SRUN_BASE[@]}" "${GPU_WRAPPER[@]}" \
    "${APPTAINER_CMD}" exec "${BIND_ARGS[@]}" "${CONTAINER_IMAGE}" \
    "${BENCH_CMD[@]}"
}
