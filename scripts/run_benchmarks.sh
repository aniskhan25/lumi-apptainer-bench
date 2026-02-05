#!/bin/bash
set -euo pipefail

# LUMI benchmark runbook. Update these two paths when container images change.
OLD_CONTAINER="/appl/local/csc/soft/ai/images/pytorch_2.7.1_lumi.sif"
NEW_CONTAINER="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif"

# Environment configuration.
export PROJECT_NAME="${PROJECT_NAME:-project_462000131}"
export PARTITION="${PARTITION:-standard-g}"
export ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

RESULTS_ROOT="/scratch/${PROJECT_NAME}/${USER}/bench_results"

# -----------------------------------------------------------------------------
# 1) Single-node compute baselines.
# -----------------------------------------------------------------------------
export BENCH_TEMPLATE="./templates/single_8g_8r.sh"
./templates/single_8g_8r.sh \
  "${NEW_CONTAINER}" \
  -- bench/run single --out \
  "${RESULTS_ROOT}/lumi_single.json"

./templates/single_8g_8r.sh \
  "${NEW_CONTAINER}" \
  -- bench/run ddp --out \
  "${RESULTS_ROOT}/lumi_ddp.json"

./templates/single_8g_16r.sh \
  "${NEW_CONTAINER}" \
  -- bench/run single \
  --out "${RESULTS_ROOT}/lumi_single_16r.json"

# -----------------------------------------------------------------------------
# 2) Multi-node collectives.
# -----------------------------------------------------------------------------
export NODES=2

./templates/allreduce_sweep.sh \
  "${NEW_CONTAINER}" \
  -- bench/run multi --allreduce \
  --out "${RESULTS_ROOT}/lumi_allreduce.json"

./templates/multi_ng_8rpn.sh \
  "${NEW_CONTAINER}" \
  -- bench/run multi \
  --out "${RESULTS_ROOT}/lumi_multi.json"

./templates/multi_ng_8rpn.sh \
  "${NEW_CONTAINER}" \
  -- bench/run ddp \
  --out "${RESULTS_ROOT}/lumi_ddp_2n.json"

# -----------------------------------------------------------------------------
# 3) Filesystem / environment check.
# -----------------------------------------------------------------------------
export BENCH_TEMPLATE="./templates/filesystem.sh"
./templates/filesystem.sh \
  "${NEW_CONTAINER}" \
  -- bench/run check \
  --out "${RESULTS_ROOT}/lumi_check.json"

# -----------------------------------------------------------------------------
# 4) Comparisons (old vs new).
# -----------------------------------------------------------------------------
./bench/compare.sh \
  --old "${OLD_CONTAINER}" \
  --new "${NEW_CONTAINER}" \
  --mode check \
  --results-dir "${RESULTS_ROOT}/lumi_check_compare"

export NODES=2
export BENCH_TEMPLATE="./templates/multi_ng_8rpn.sh"
./bench/compare.sh \
  --old "${OLD_CONTAINER}" \
  --new "${NEW_CONTAINER}" \
  --mode multi \
  --results-dir "${RESULTS_ROOT}/lumi_multi_compare"

export BENCH_TEMPLATE="./templates/single_8g_16r.sh"
./bench/compare.sh \
  --old "${OLD_CONTAINER}" \
  --new "${NEW_CONTAINER}" \
  --mode single \
  --results-dir "${RESULTS_ROOT}/lumi_single_16r_compare"
