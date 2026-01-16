#!/bin/bash
set -euo pipefail

# Puhti filesystem-focused template.
# Usage:
#   ./templates/puhti_filesystem.sh <container.sif> -- bench/run check --out results.json

CONTAINER_IMAGE="${1:?container image path required}"
shift

PROJECT_NAME="${PROJECT_NAME:?set PROJECT_NAME (e.g. project_2001234)}"

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
  echo "Container image not found: ${CONTAINER_IMAGE}" >&2
  echo "Tip: use a full path on /scratch or /projappl (e.g. /scratch/${PROJECT_NAME}/containers/my.sif)." >&2
  exit 1
fi
PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_NAME}}"
PROJECT_ROOT="${PROJECT_ROOT:-/projappl/${PROJECT_NAME}}"
HOME_ROOT="${HOME_ROOT:-/users/${USER}}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/${USER}/bench_cache}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH_ROOT}/${USER}/bench_results}"

APPTAINER_GPU_FLAG="${APPTAINER_GPU_FLAG:---nv}"

BIND_ARGS=()
add_bind() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    BIND_ARGS+=(--bind "${path}:${path}")
  fi
}
add_bind "${SCRATCH_ROOT}"
add_bind "${PROJECT_ROOT}"
add_bind "${HOME_ROOT}"

MPI_MODE="${MPI_MODE:-host}" # host|container
SRUN_MPI_FLAG=()
if [[ "${MPI_MODE}" == "container" ]]; then
  SRUN_MPI_FLAG=(--mpi=pmi2)
fi

NODES=1
NTASKS_PER_NODE=1
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
DIST="${DIST:-block}"
CPU_BIND="${CPU_BIND:-cores}"

TIME_LIMIT="${TIME_LIMIT:-}"
if [[ -z "${TIME_LIMIT}" && "${PARTITION}" == "gputest" ]]; then
  TIME_LIMIT="00:15:00"
fi

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
export BENCH_GPUS_PER_NODE=0
export BENCH_CPUS_PER_TASK="${CPUS_PER_TASK}"
export BENCH_DIST="${DIST}"
export BENCH_CPU_BIND="${CPU_BIND}"

BENCH_CMD=(bench/run check --out "${RESULTS_JSON}")
if [[ "$#" -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  if [[ "$#" -gt 0 ]]; then
    BENCH_CMD=("$@")
  fi
fi

{
  echo "run_id=${RUN_ID}"
  echo "container_image=${CONTAINER_IMAGE}"
  echo "partition=${PARTITION}"
  echo "account=${ACCOUNT}"
  echo "mpi_mode=${MPI_MODE}"
  echo "nodes=${NODES}"
  echo "ntasks_per_node=${NTASKS_PER_NODE}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "distribution=${DIST}"
  echo "cpu_bind=${CPU_BIND}"
  echo "bench_cmd=${BENCH_CMD[*]}"
  srun --version || true
} | tee "${LOG_DIR}/run_env.txt"

SRUN_BASE=(
  srun
  --partition="${PARTITION}"
  --account="${ACCOUNT}"
  --nodes="${NODES}"
  --ntasks-per-node="${NTASKS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --distribution="${DIST}"
  --cpu-bind="${CPU_BIND}"
  "${SRUN_MPI_FLAG[@]}"
)

if [[ -n "${TIME_LIMIT}" ]]; then
  SRUN_BASE+=(--time="${TIME_LIMIT}")
fi

"${SRUN_BASE[@]}" \
  apptainer exec "${APPTAINER_GPU_FLAG}" "${BIND_ARGS[@]}" "${CONTAINER_IMAGE}" \
  "${BENCH_CMD[@]}"
