#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bench/compare.sh --old <old.sif> --new <new.sif> --mode <check|single|multi|ddp> --results-dir <dir> [--template <path>] [-- <bench args>]

Either pass --template or set BENCH_TEMPLATE in the environment.
USAGE
}

OLD_IMAGE=""
NEW_IMAGE=""
MODE="single"
RESULTS_DIR=""
TEMPLATE=""
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --old)
      OLD_IMAGE="$2"
      shift 2
      ;;
    --new)
      NEW_IMAGE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --template)
      TEMPLATE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${OLD_IMAGE}" || -z "${NEW_IMAGE}" ]]; then
  echo "--old and --new are required" >&2
  usage
  exit 1
fi

if [[ -z "${TEMPLATE}" ]]; then
  TEMPLATE="${BENCH_TEMPLATE:-}"
fi

if [[ -z "${TEMPLATE}" ]]; then
  echo "Template not set. Use --template or BENCH_TEMPLATE." >&2
  usage
  exit 1
fi

if [[ -z "${RESULTS_DIR}" ]]; then
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  if [[ -n "${SCRATCH:-}" ]]; then
    RESULTS_DIR="${SCRATCH}/${USER}/bench_results/${RUN_ID}"
  elif [[ -n "${PROJECT_NAME:-}" ]]; then
    RESULTS_DIR="/scratch/${PROJECT_NAME}/${USER}/bench_results/${RUN_ID}"
  else
    RESULTS_DIR="/tmp/bench_results/${RUN_ID}"
  fi
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${RESULTS_DIR}"

if [[ ! -x "${TEMPLATE}" ]]; then
  echo "Template not executable: ${TEMPLATE}" >&2
  exit 1
fi

OLD_OUT="${RESULTS_DIR}/results_old.json"
NEW_OUT="${RESULTS_DIR}/results_new.json"
DELTA_OUT="${RESULTS_DIR}/delta.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -f "${OLD_OUT}" "${NEW_OUT}" "${DELTA_OUT}"

"${TEMPLATE}" "${OLD_IMAGE}" -- bench/run "${MODE}" --out "${OLD_OUT}" "${EXTRA_ARGS[@]}"
"${TEMPLATE}" "${NEW_IMAGE}" -- bench/run "${MODE}" --out "${NEW_OUT}" "${EXTRA_ARGS[@]}"

python3 "${SCRIPT_DIR}/compare_results.py" "${OLD_OUT}" "${NEW_OUT}" "${DELTA_OUT}"

echo "Wrote: ${OLD_OUT}"
echo "Wrote: ${NEW_OUT}"
echo "Wrote: ${DELTA_OUT}"
