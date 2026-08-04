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

# Cache placement is the difference between a job that works at 8 nodes and one that
# loses a rank at 64. Triton/Inductor cache writes are not atomic on Lustre: at 64+ ranks
# a rank reads a half-written entry, raises JSONDecodeError, exits, and the remaining
# ranks hang at the next collective. Per-node /tmp removes the contention entirely.
#
# LAIF_CACHE_MODE=tmp     per-node /tmp (safe default)
# LAIF_CACHE_MODE=lustre  shared Lustre path -- for deliberately reproducing the failure
lumi_setup_caches() {
  LAIF_CACHE_MODE="${LAIF_CACHE_MODE:-tmp}"

  # Key by container so an incompatible image cannot reuse another's compiled artifacts,
  # and so the compile cost is paid once per container rather than once per job.
  local container_id
  container_id="$(basename "${CONTAINER_IMAGE}" .sif)"

  case "${LAIF_CACHE_MODE}" in
    tmp)
      LAIF_CACHE_ROOT="/tmp/laif-cache-${USER}/${container_id}"
      ;;
    lustre)
      LAIF_CACHE_ROOT="${SCRATCH_ROOT}/${USER}/laif-cache/${container_id}"
      ;;
    *)
      echo "LAIF_CACHE_MODE must be 'tmp' or 'lustre', got '${LAIF_CACHE_MODE}'" >&2
      exit 1
      ;;
  esac

  export LAIF_CACHE_ROOT
  export TRITON_CACHE_DIR="${LAIF_CACHE_ROOT}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${LAIF_CACHE_ROOT}/inductor"
  export TORCH_EXTENSIONS_DIR="${LAIF_CACHE_ROOT}/extensions"
  export MIOPEN_USER_DB_PATH="${LAIF_CACHE_ROOT}/miopen"
  export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}/cache"
  export MIOPEN_USER_DB="${MIOPEN_USER_DB_PATH}/config"

  # Only the Lustre paths can be created here; /tmp is node-local, so each node creates
  # its own inside the srun step (see lumi_cache_mkdir_cmd).
  if [[ "${LAIF_CACHE_MODE}" == "lustre" ]]; then
    mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
             "${TORCH_EXTENSIONS_DIR}" "${MIOPEN_USER_DB_PATH}"
  fi
}

# The digest is published alongside the release, so there is no need to hash 14 GB
# ourselves. Recording it is what makes a result traceable to an exact image.
lumi_container_digest() {
  local resolved dir base sha_file digest
  resolved="$(readlink -f "${CONTAINER_IMAGE}" 2>/dev/null || echo "${CONTAINER_IMAGE}")"
  dir="$(dirname "${resolved}")"
  base="$(basename "${resolved}")"
  for sha_file in "${dir}"/*.sha256; do
    [[ -f "${sha_file}" ]] || continue
    digest="$(awk -v want="${base}" 'index($NF, want) {print $1; exit}' "${sha_file}")"
    if [[ -n "${digest}" ]]; then
      echo "${digest}"
      return
    fi
  done
  echo ""
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

  module purge
  module use /appl/local/laifs/modules
  module load lumi-aif-singularity-bindings
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

  lumi_setup_caches
  export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/${USER}/torch_home}"
  mkdir -p "${TORCH_HOME}"

  export BENCH_CONTAINER_DIGEST="$(lumi_container_digest)"

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

  # Per-task wrapper, run outside the container by srun. Two jobs:
  #   1. Create the cache directories -- with LAIF_CACHE_MODE=tmp these live on each
  #      node's own /tmp, so they cannot be created from the login node.
  #   2. Bind one GCD per rank. The container would do this from its OCI ENTRYPOINT, but
  #      `apptainer exec` never runs an ENTRYPOINT, so under exec we must do it here or
  #      every rank sees all 8 GCDs. Set USE_ROCR_VISIBLE_DEVICES=0 with APPTAINER_MODE=run
  #      to test whether the container's own binding works.
  local wrapper_body='mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$MIOPEN_USER_DB_PATH";'
  if [[ "${USE_ROCR_VISIBLE_DEVICES:-0}" == "1" ]]; then
    wrapper_body+=' export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID};'
  fi
  wrapper_body+=' exec "$@"'
  GPU_WRAPPER=(bash -lc "${wrapper_body}" --)

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
    # The mask list addresses 7 cores in each of 8 GPU groups, so it is only satisfiable
    # on a whole node. A bare srun launched from a login node may be given a subset --
    # observed 4 cores per group -- and srun then aborts the step with "CPU binding
    # outside of job step allocation", which names neither the masks nor the cause.
    # Ask for the whole node explicitly. LUMI-G bills per node anyway.
    CPU_BIND="mask_cpu:${CPU_BIND_MASKS:-${LUMI_GPU_CPU_BIND_MASKS}}"
    SRUN_BASE+=(--exclusive)
  fi
  SRUN_BASE+=(
    --cpus-per-task="${CPUS_PER_TASK}"
    --distribution="${DIST}"
    --cpu-bind="${CPU_BIND}"
    "${SRUN_MPI_FLAG[@]}"
    --time="${TIME_LIMIT}"
  )
  if [[ -n "${NODELIST:-}" ]]; then
    SRUN_BASE+=(--nodelist="${NODELIST}")
  fi
  if [[ -n "${EXCLUDE_NODES:-}" ]]; then
    SRUN_BASE+=(--exclude="${EXCLUDE_NODES}")
  fi
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
    echo "container_digest=${BENCH_CONTAINER_DIGEST}"
    echo "apptainer_mode=${APPTAINER_MODE:-exec}"
    echo "use_rocr_visible_devices=${USE_ROCR_VISIBLE_DEVICES:-0}"
    echo "laif_cache_mode=${LAIF_CACHE_MODE}"
    echo "laif_cache_root=${LAIF_CACHE_ROOT}"
    if [[ -n "${NODELIST:-}" ]]; then
      echo "nodelist=${NODELIST}"
    fi
    if [[ -n "${EXCLUDE_NODES:-}" ]]; then
      echo "exclude_nodes=${EXCLUDE_NODES}"
    fi
    if [[ "${ENABLE_LUMI_HSN}" == "1" ]]; then
      echo "nccl_socket_ifname=${NCCL_SOCKET_IFNAME}"
      echo "nccl_net_gdr_level=${NCCL_NET_GDR_LEVEL}"
    fi
    echo "bench_cmd=${BENCH_CMD[*]}"
    srun --version || true
  } | tee "${LOG_DIR}/run_env.txt"
}

lumi_exec() {
  # APPTAINER_MODE=exec (default) runs the command directly and bypasses the image's
  # ENTRYPOINT. APPTAINER_MODE=run goes through it. The lumi-multitorch entrypoint is
  # where ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES are set, so the two modes do not
  # produce the same environment -- which is the point of having the toggle.
  "${SRUN_BASE[@]}" "${GPU_WRAPPER[@]}" \
    "${APPTAINER_CMD}" "${APPTAINER_MODE:-exec}" "${BIND_ARGS[@]}" "${CONTAINER_IMAGE}" \
    "${BENCH_CMD[@]}"
}
