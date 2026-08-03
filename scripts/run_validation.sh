#!/bin/bash
set -euo pipefail

# Validate a single container against the release gates. No old/new pairing: with one
# image under test there is no baseline, so each metric is checked against a declared
# limit in manifests/gates/ rather than against a previous run.
#
# Usage:
#   export PROJECT_NAME=project_462000131
#   ./scripts/run_validation.sh                 # phases 1 and 2 (1 and 2 nodes)
#   PHASE=1 ./scripts/run_validation.sh         # capability probe only
#   PHASE=3 ./scripts/run_validation.sh         # 4-node EP=32 (needs 4 nodes)
#
# Environment:
#   CONTAINER       image under test (default: the -latest symlink)
#   PHASE           1 | 2 | 3 | all  (default: all-but-3, i.e. 1 and 2)
#   EXCLUDE_NODES   nodes to avoid; some hang indefinitely during RCCL init
#   NODELIST        pin to specific nodes

PROJECT_NAME="${PROJECT_NAME:?set PROJECT_NAME (e.g. project_462000131)}"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
PHASE="${PHASE:-default}"

export PROJECT_NAME
export PARTITION="${PARTITION:-standard-g}"
export ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

export RESULTS_ROOT="${RESULTS_ROOT:-/scratch/${PROJECT_NAME}/${USER}/validation_results}"
mkdir -p "${RESULTS_ROOT}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATES_DIR="${REPO_DIR}/manifests/gates"
EVAL="${REPO_DIR}/bench/scripts/eval_gates.py"

echo "container: ${CONTAINER}"
echo "results:   ${RESULTS_ROOT}"
echo

FAILED=0

# Run one template, then check its output against a gate spec. A missing results file is
# a failure, not a skip -- a job that died before writing is exactly what we are hunting.
run_gate() {
  local label="$1" gate_spec="$2" template="$3"
  shift 3
  local out="${RESULTS_ROOT}/${label}.json"
  local gates_out="${RESULTS_ROOT}/${label}.gates.json"

  echo "=== ${label} ==="
  if ! "${REPO_DIR}/templates/${template}" "${CONTAINER}" -- "$@" --out "${out}"; then
    echo "  [FAIL] ${label}: launch or run failed"
    FAILED=$((FAILED + 1))
    return
  fi
  if [[ ! -f "${out}" ]]; then
    echo "  [FAIL] ${label}: no results written to ${out}"
    FAILED=$((FAILED + 1))
    return
  fi
  if ! python3 "${EVAL}" "${out}" "${GATES_DIR}/${gate_spec}" "${gates_out}"; then
    FAILED=$((FAILED + 1))
  fi
  echo
}

# NODES and GROUP_SIZE are exported rather than prefixed onto the run_gate call:
# `VAR=x some_function` has unspecified scoping for shell functions, and wrapping the
# call in a subshell would lose the FAILED counter.

# Phase 1 -- one node, minutes. Capability and environment.
if [[ "${PHASE}" == "1" || "${PHASE}" == "all" || "${PHASE}" == "default" ]]; then
  export NODES=1
  run_gate phase1_probe probe.json probe.sh bench/run probe
fi

# Phase 2 -- two nodes. The intra-node control must pass before a cross-node result
# means anything, so EP=8 is checked first.
if [[ "${PHASE}" == "2" || "${PHASE}" == "all" || "${PHASE}" == "default" ]]; then
  export NODES=2 GROUP_SIZE=8
  run_gate phase2_alltoall_ep8 alltoall_intra.json \
    alltoall_sweep.sh bench/run alltoall --group-size 8
  export GROUP_SIZE=16
  run_gate phase2_alltoall_ep16 alltoall_cross.json \
    alltoall_sweep.sh bench/run alltoall --group-size 16
fi

# Phase 3 -- four nodes. EP=32, the configuration reported to fail during init.
if [[ "${PHASE}" == "3" || "${PHASE}" == "all" ]]; then
  export NODES=4 GROUP_SIZE=32
  run_gate phase3_alltoall_ep32 alltoall_cross.json \
    alltoall_sweep.sh bench/run alltoall --group-size 32
fi

echo "================================"
if [[ "${FAILED}" -eq 0 ]]; then
  echo "All gates passed."
else
  echo "${FAILED} gate group(s) failed. See ${RESULTS_ROOT}/*.gates.json"
fi
exit "${FAILED}"
