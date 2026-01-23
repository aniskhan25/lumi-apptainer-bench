#!/bin/bash
set -euo pipefail

# Puhti single node, 4 GPUs, 4 ranks (1 rank per GPU).
# Usage:
#   ./templates/puhti_single_4g_4r.sh <container.sif> -- bench/run single --out results.json

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
GPU_VISIBLE_ENV="${GPU_VISIBLE_ENV:-CUDA_VISIBLE_DEVICES}"

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

APPTAINER_CMD="${APPTAINER_CMD:-apptainer}"
if ! command -v "${APPTAINER_CMD}" >/dev/null 2>&1; then
  if command -v singularity >/dev/null 2>&1; then
    APPTAINER_CMD="singularity"
  else
    echo "Apptainer/Singularity not found in PATH." >&2
    exit 1
  fi
fi

MPI_MODE="${MPI_MODE:-host}" # host|container
SRUN_MPI_FLAG=()
if [[ "${MPI_MODE}" == "container" ]]; then
  SRUN_MPI_FLAG=(--mpi=pmi2)
fi

NODES=1
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-10}" # Puhti guideline: <= 10 CPU cores per GPU
DIST="${DIST:-block}"
CPU_BIND="${CPU_BIND:-cores}"
GPU_BIND="${GPU_BIND:-closest}"

TIME_LIMIT="${TIME_LIMIT:-}"
if [[ -z "${TIME_LIMIT}" && "${PARTITION}" == "gputest" ]]; then
  TIME_LIMIT="00:15:00"
fi

USE_GPU_VISIBLE_DEVICES="${USE_GPU_VISIBLE_DEVICES:-1}"
GPU_WRAPPER=()
if [[ "${USE_GPU_VISIBLE_DEVICES}" == "1" ]]; then
  GPU_WRAPPER=(bash -lc "export ${GPU_VISIBLE_ENV}=\${SLURM_LOCALID}; exec \"\$@\"" --)
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
export BENCH_GPUS_PER_NODE="${GPUS_PER_NODE}"
export BENCH_CPUS_PER_TASK="${CPUS_PER_TASK}"
export BENCH_DIST="${DIST}"
export BENCH_CPU_BIND="${CPU_BIND}"

BENCH_CMD=(bench/run single --out "${RESULTS_JSON}")
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
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "distribution=${DIST}"
  echo "cpu_bind=${CPU_BIND}"
  echo "gpu_bind=${GPU_BIND}"
  echo "bench_cmd=${BENCH_CMD[*]}"
  srun --version || true
} | tee "${LOG_DIR}/run_env.txt"

SRUN_BASE=(
  srun
  --partition="${PARTITION}"
  --account="${ACCOUNT}"
  --nodes="${NODES}"
  --ntasks-per-node="${NTASKS_PER_NODE}"
  --gres="gpu:v100:${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --distribution="${DIST}"
  --cpu-bind="${CPU_BIND}"
  --gpu-bind="${GPU_BIND}"
  "${SRUN_MPI_FLAG[@]}"
)

if [[ -n "${TIME_LIMIT}" ]]; then
  SRUN_BASE+=(--time="${TIME_LIMIT}")
fi

"${SRUN_BASE[@]}" "${GPU_WRAPPER[@]}" \
  "${APPTAINER_CMD}" exec "${APPTAINER_GPU_FLAG}" "${BIND_ARGS[@]}" "${CONTAINER_IMAGE}" \
  "${BENCH_CMD[@]}"
