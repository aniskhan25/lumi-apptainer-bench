#!/bin/bash
set -euo pipefail

# LUMI benchmark runbook. Update these two paths when container images change.
OLD_CONTAINER="/appl/local/csc/soft/ai/images/pytorch_2.7.1_lumi.sif"
NEW_CONTAINER="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif"

# Environment configuration.
export PROJECT_NAME="${PROJECT_NAME:-project_462000131}"
export PARTITION="${PARTITION:-standard-g}"
export ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

RESULTS_ROOT="/scratch/${PROJECT_NAME}/${USER}/bench_results"

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

# 1) Single-node compute baselines.
run_template "./templates/single_8g_8r.sh" "lumi_single.json" bench/run single
run_template "./templates/single_8g_8r.sh" "lumi_ddp.json" bench/run ddp
run_template "./templates/single_8g_16r.sh" "lumi_single_16r.json" bench/run single

# 2) Multi-node collectives.
export NODES=2

run_template "./templates/allreduce_sweep.sh" "lumi_allreduce.json" bench/run multi --allreduce
run_template "./templates/multi_ng_8rpn.sh" "lumi_multi.json" bench/run multi
run_template "./templates/multi_ng_8rpn.sh" "lumi_ddp_2n.json" bench/run ddp

# 3) Filesystem / environment check.
run_template "./templates/filesystem.sh" "lumi_check.json" bench/run check

# 4) Comparisons (old vs new).
run_compare "./templates/filesystem.sh" "check" "lumi_check_compare"
run_compare "./templates/single_8g_8r.sh" "ddp" "lumi_ddp_compare"

export NODES=2
run_compare "./templates/multi_ng_8rpn.sh" "multi" "lumi_multi_compare"
run_compare "./templates/multi_ng_8rpn.sh" "ddp" "lumi_ddp_2n_compare"
run_compare "./templates/single_8g_16r.sh" "single" "lumi_single_16r_compare"
