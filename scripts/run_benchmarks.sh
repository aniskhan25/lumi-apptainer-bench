#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  export PROJECT_NAME=project_462000131
  export OLD_CONTAINER=/path/to/old.sif
  export NEW_CONTAINER=/path/to/new.sif
  ./scripts/run_benchmarks.sh

Optional:
  export PARTITION=dev-g
  export ACCOUNT=$PROJECT_NAME
  export RESULTS_ROOT=/scratch/$PROJECT_NAME/$USER/bench_results
EOF
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    usage >&2
    exit 1
  fi
}

require_env PROJECT_NAME

OLD_CONTAINER="${OLD_CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif}"
NEW_CONTAINER="${NEW_CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

export PARTITION="${PARTITION:-dev-g}"
export ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/${PROJECT_NAME}/${USER}/bench_results}"

run_template() {
  local template="$1"
  local output_name="$2"
  shift 2
  "${template}" "${NEW_CONTAINER}" -- "$@" --out "${RESULTS_ROOT}/${output_name}"
}

run_compare() {
  local template="$1"
  local mode="$2"
  local results_dir="$3"
  export BENCH_TEMPLATE="${template}"
  ./bench/compare.sh \
    --old "${OLD_CONTAINER}" \
    --new "${NEW_CONTAINER}" \
    --mode "${mode}" \
    --results-dir "${RESULTS_ROOT}/${results_dir}"
}

# 1. Single-node compute baselines.
run_template "./templates/single_8g_8r.sh" "lumi_single.json" bench/run single
run_template "./templates/single_8g_8r.sh" "lumi_ddp.json" bench/run ddp
run_template "./templates/single_8g_16r.sh" "lumi_single_16r.json" bench/run single

# 2. Multi-node collectives.
export NODES=2

run_template "./templates/allreduce_sweep.sh" "lumi_allreduce.json" bench/run multi --allreduce
run_template "./templates/multi_ng_8rpn.sh" "lumi_multi.json" bench/run multi
run_template "./templates/multi_ng_8rpn.sh" "lumi_ddp_2n.json" bench/run ddp

# 3. Filesystem and environment check.
run_template "./templates/filesystem.sh" "lumi_check.json" bench/run check

# 4. Old-vs-new comparisons.
run_compare "./templates/filesystem.sh" "check" "lumi_check_compare"
run_compare "./templates/single_8g_8r.sh" "ddp" "lumi_ddp_compare"

export NODES=2
run_compare "./templates/multi_ng_8rpn.sh" "multi" "lumi_multi_compare"
run_compare "./templates/multi_ng_8rpn.sh" "ddp" "lumi_ddp_2n_compare"
run_compare "./templates/single_8g_16r.sh" "single" "lumi_single_16r_compare"
