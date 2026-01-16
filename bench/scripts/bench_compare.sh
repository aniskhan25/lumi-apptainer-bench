#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bench/compare.sh --old <old.sif> --new <new.sif> --mode <single|multi> --results-dir <dir> [--template <path>] [-- <bench args>]

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
    RESULTS_DIR="${SCRATCH}/bench_results/${USER}/${RUN_ID}"
  elif [[ -n "${PROJECT_NAME:-}" ]]; then
    RESULTS_DIR="/scratch/${PROJECT_NAME}/bench_results/${USER}/${RUN_ID}"
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

"${TEMPLATE}" "${OLD_IMAGE}" -- bench/run "${MODE}" --out "${OLD_OUT}" "${EXTRA_ARGS[@]}"
"${TEMPLATE}" "${NEW_IMAGE}" -- bench/run "${MODE}" --out "${NEW_OUT}" "${EXTRA_ARGS[@]}"

python3 - "${OLD_OUT}" "${NEW_OUT}" "${DELTA_OUT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

old_path, new_path, out_path = sys.argv[1:4]

def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def pct_delta(old, new):
    if old is None or new is None:
        return None
    try:
        old_v = float(old)
        new_v = float(new)
    except (TypeError, ValueError):
        return None
    if old_v == 0:
        return None
    return (new_v - old_v) / old_v * 100.0

def avg(values):
    if not values:
        return None
    return sum(values) / len(values)

old = load(old_path)
new = load(new_path)

gemm_old = old.get("tests", {}).get("single", {}).get("gemm", {}).get("tflops")
gemm_new = new.get("tests", {}).get("single", {}).get("gemm", {}).get("tflops")

km_old = old.get("tests", {}).get("single", {}).get("kernel_mix", {}).get("latency_p50_ms")
km_new = new.get("tests", {}).get("single", {}).get("kernel_mix", {}).get("latency_p50_ms")

ar_bw_old = avg(old.get("tests", {}).get("multi", {}).get("allreduce", {}).get("bandwidth_gbps", []))
ar_bw_new = avg(new.get("tests", {}).get("multi", {}).get("allreduce", {}).get("bandwidth_gbps", []))

ar_lat_old = avg(old.get("tests", {}).get("multi", {}).get("allreduce", {}).get("latency_us", []))
ar_lat_new = avg(new.get("tests", {}).get("multi", {}).get("allreduce", {}).get("latency_us", []))

thr_gemm = float(os.environ.get("BENCH_REGRESS_GEMM_PCT", "10"))
thr_bw = float(os.environ.get("BENCH_REGRESS_ALLREDUCE_BW_PCT", "10"))
thr_lat = float(os.environ.get("BENCH_REGRESS_LATENCY_PCT", "15"))

metrics = {}

def add_metric(name, old_v, new_v, mode):
    delta = pct_delta(old_v, new_v)
    regression = None
    if delta is not None:
        if mode == "drop":
            regression = delta < -thr_gemm
        elif mode == "bw_drop":
            regression = delta < -thr_bw
        elif mode == "lat_increase":
            regression = delta > thr_lat
    metrics[name] = {
        "old": old_v,
        "new": new_v,
        "delta_pct": delta,
        "regression": regression,
    }

add_metric("single_gemm_tflops", gemm_old, gemm_new, "drop")
add_metric("single_kernel_mix_p50_ms", km_old, km_new, "lat_increase")
add_metric("multi_allreduce_bw_avg_gbps", ar_bw_old, ar_bw_new, "bw_drop")
add_metric("multi_allreduce_lat_avg_us", ar_lat_old, ar_lat_new, "lat_increase")

regressions = [k for k, v in metrics.items() if v.get("regression")]

payload = {
    "run_id": old.get("run_id", ""),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "old_results": old_path,
    "new_results": new_path,
    "metrics": metrics,
    "regressions": regressions,
    "regression_count": len(regressions),
    "thresholds": {
        "gemm_drop_pct": thr_gemm,
        "allreduce_bw_drop_pct": thr_bw,
        "latency_increase_pct": thr_lat,
    },
}

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY

echo "Wrote: ${OLD_OUT}"
echo "Wrote: ${NEW_OUT}"
echo "Wrote: ${DELTA_OUT}"
