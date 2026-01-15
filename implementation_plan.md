# Benchmark implementation plan: "new vs previous" LUMI container

## 1) Objective and non-goals

Objective: build a LUMI-ready benchmark package that:
- Runs via standard CLI: modules, srun, apptainer.
- Provides blessed srun templates with correct CPU/GPU binding and filesystem binds.
- Compares old vs new containers and flags regressions.
- Emits JSON plus a brief human summary.

Non-goals:
- App-specific assumptions.
- Deep tuning research.

## 2) Deliverables

### A) templates/
Small library of shell templates:
1. Single node: 8 GPUs, 8 ranks (1 rank/GPU).
2. Single node: 8 GPUs, 16 ranks (2 ranks/GPU, optional).
3. Multi-node: N nodes, 8 GPUs/node, 8 ranks/node.
4. Multi-node all-reduce sweep (stress).
5. Filesystem template (binds, cache, scratch strategy).

Each template sets:
- Slurm placement/binding flags.
- Container bind paths.
- Cache root on scratch.
- Environment contract (what must be present).

### B) bench/
Single entrypoint `bench/run`:
- `check` (sanity)
- `single` (node-local compute)
- `multi` (multi-node comm + minimal distributed step)
- `compare` (A/B old vs new; thin wrapper around `bench/compare.sh`)

### C) docs/
- How to run on LUMI.
- How to interpret results.
- Template selection guide.

## 3) Benchmark suite (minimal, high-signal)

Tier A: check (30 to 60s)
- GPU visibility.
- Framework import (if present).
- Cache dir writable.
- Report ROCm/HIP version, GPU count, node info.

Tier B: single
- GEMM/matmul throughput (BF16/FP16, FP32 if available).
- Kernel-mix proxy (small operator mix or tiny step).

Tier C: multi
- All-reduce sweep (latency and bandwidth regimes).
- Minimal distributed step (optional; can be collectives plus correctness).

## 4) Template contract

Every template guarantees:
- Rank to GPU mapping (explicit GPU binding).
- Explicit CPU binding and no oversubscription (`--cpus-per-task`).
- Explicit distribution policy (`--distribution=...`).
- Partition and account are explicit (`--partition`, `--account`), or set via env.
- HPC-correct filesystem: scratch cache bind, no container writes, optional project/work bind.
- Results JSON at deterministic scratch path.

## 5) Template design (concept)

User calls:
- `./templates/single_8g_8r.sh <container.sif> -- bench/run single --out results.json`
- `./templates/multi_ng_8rpn.sh <container.sif> -- bench/run multi --out results.json`

Standard variables:
- `CONTAINER_IMAGE`
- `CACHE_ROOT`, `RESULTS_DIR`
- `BIND_PATHS`
- `SRUN_FLAGS`

Template actions:
1. Create cache/results dirs.
2. Print module list, `srun --version`, hostnames, Slurm env.
3. Run `srun` with fixed binding and distribution.
4. Run `apptainer exec` with fixed binds.
5. Save logs and JSON.

## 6) A/B regression workflow

Command:
- `./bench/compare.sh --old <old.sif> --new <new.sif> --mode single --results-dir <dir>`
- `./bench/compare.sh --old <old.sif> --new <new.sif> --mode multi --results-dir <dir>`

Behavior:
- Runs the same template twice (old then new).
- Writes `results_old.json`, `results_new.json`, `delta.json`.
- If `--results-dir` is omitted, default to `$SCRATCH/bench_results/<run_id>`.

Regression policy (initial):
- Throughput drop > 10 to 15%.
- All-reduce bandwidth drop > 10 to 15%.
- Latency increase > 15 to 20%.

## 7) Milestones

1. Templates: single 8g/8r, multi N/8rpn, scratch bind, logging.
2. Tier A + GEMM + JSON writer.
3. All-reduce sweep + correctness, wired into multi template.
4. Compare + docs.

## 8) LUMI-grade checklist

- Templates are explicit and opinionated.
- Logs include exact placement/binding flags.
- Cache paths are stable and on scratch.
- Results are structured JSON.

## 9) Design principles (simple but extensible)

- One entrypoint: `bench/run` only.
- Versioned JSON schema, backward compatible.
- Record the exact `srun` + `apptainer exec` command and key env vars.
- Templates are declarative (env overrides > branching).
- `bench/compare.sh` has sane defaults.

---

# Appendix A: LUMI defaults (from CSC docs)

Filesystem binds:
- Bind full project paths: `/scratch/<project>` and `/project/<project>`.
- Prefer scratch/flash for IO:
  - `/scratch/<project>` (scratch)
  - `/flash/<project>` (flash)
  - `/project/<project>` (project)
  - `/users/<username>` (home)

MPI mode:
- Prefer host MPI (better network). App must be ABI-compatible.
- Container MPI uses `srun --mpi=pmi2` and will be slower.
- Template knob: `MPI_MODE={host|container}`.

GPU binding on LUMI-G:
- Binding assumes full-node allocation; LUMI-G has 56 usable CPU cores.
- If app uses local rank for GPU selection: `--cpu-bind=map_cpu:49,57,17,25,1,9,33,41`.
- If app cannot select GPU: wrap with `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`.

Distribution:
- Use explicit `--distribution=block` (or chosen policy) and log it.

References:
- https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs/
- https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/
- https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/
- https://docs.lumi-supercomputer.eu/storage/

# Appendix B: Reference (schema + template)

## Results JSON schema (minimal)

| Section | Required fields | Notes |
| --- | --- | --- |
| Top-level | `schema_version`, `run_id`, `timestamp_utc` | Strings, `timestamp_utc` is RFC3339. |
| container | `image_path`, `image_digest` | `image_digest` if available. |
| slurm | `job_id`, `nodes`, `ntasks`, `ntasks_per_node`, `gpus_per_node`, `cpus_per_task`, `distribution`, `cpu_bind`, `mpi_mode` | `mpi_mode` is `host` or `container`. |
| system | `hostname_list`, `partition`, `rocm_version`, `gpu_count` | `hostname_list` is array of strings. |
| paths | `cache_root`, `results_dir` | Absolute paths. |
| tests.check | `status`, `details` | Required if test executed. |
| tests.single.gemm | `dtype`, `tflops`, `latency_p50_ms`, `latency_p95_ms` | Required if test executed. |
| tests.single.kernel_mix | `latency_p50_ms`, `latency_p95_ms` | Required if test executed. |
| tests.multi.allreduce | `message_sizes_bytes`, `bandwidth_gbps`, `latency_us`, `checksum` | Required if test executed. |
| optional | `git_rev`, `template_name`, `template_version`, `warnings`, `notes` | Optional metadata. |

## Template header block (canonical defaults)

```bash
#!/bin/bash
set -euo pipefail

# ---- User inputs ----
CONTAINER_IMAGE="${1:?container image path required}"
shift

# ---- Project identifiers ----
PROJECT_NAME="${PROJECT_NAME:?set PROJECT_NAME (e.g. project_465000001)}"
PARTITION="${PARTITION:-standard-g}"
ACCOUNT="${ACCOUNT:-${PROJECT_NAME}}"

# ---- Standard paths ----
SCRATCH_ROOT="/scratch/${PROJECT_NAME}"
FLASH_ROOT="/flash/${PROJECT_NAME}"
PROJECT_ROOT="/project/${PROJECT_NAME}"
HOME_ROOT="/users/${USER}"
CACHE_ROOT="${SCRATCH_ROOT}/bench_cache"
RESULTS_ROOT="${SCRATCH_ROOT}/bench_results"

# ---- Container binds (explicit full paths) ----
BIND_PATHS=(
  "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
  "${FLASH_ROOT}:${FLASH_ROOT}"
  "${PROJECT_ROOT}:${PROJECT_ROOT}"
  "${HOME_ROOT}:${HOME_ROOT}"
)
BIND_ARGS="$(printf ' -B %s' "${BIND_PATHS[@]}")"

# ---- MPI mode ----
MPI_MODE="${MPI_MODE:-host}" # host|container
SRUN_MPI_FLAG=""
if [[ "${MPI_MODE}" == "container" ]]; then
  SRUN_MPI_FLAG="--mpi=pmi2"
fi

# ---- Slurm layout ----
NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-7}" # 56 cores / 8 ranks on LUMI-G
DIST="${DIST:-block}"

# ---- Binding (LUMI-G defaults) ----
CPU_BIND="${CPU_BIND:-map_cpu:49,57,17,25,1,9,33,41}"

# ---- GPU selection wrapper ----
USE_ROCR_VISIBLE_DEVICES="${USE_ROCR_VISIBLE_DEVICES:-1}"
GPU_WRAPPER=""
if [[ "${USE_ROCR_VISIBLE_DEVICES}" == "1" ]]; then
  GPU_WRAPPER='bash -lc "export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}; exec \"$@\""'
fi

# ---- Results paths ----
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"
RESULTS_JSON="${RESULTS_DIR}/results.json"

mkdir -p "${CACHE_ROOT}" "${RESULTS_DIR}"
```

Note: `CPU_BIND` and `CPUS_PER_TASK` are LUMI-G defaults; override for LUMI-C.

Usage snippet:

```bash
SRUN_BASE=(
  srun
  --nodes="${NODES}"
  --ntasks-per-node="${NTASKS_PER_NODE}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --distribution="${DIST}"
  --cpu-bind="${CPU_BIND}"
  ${SRUN_MPI_FLAG}
)

${SRUN_BASE[@]} ${GPU_WRAPPER} \
  apptainer exec ${BIND_ARGS} "${CONTAINER_IMAGE}" \
  bench/run single --out "${RESULTS_JSON}" "$@"
```

Multi-node:

```bash
${SRUN_BASE[@]} ${GPU_WRAPPER} \
  apptainer exec ${BIND_ARGS} "${CONTAINER_IMAGE}" \
  bench/run multi --out "${RESULTS_JSON}" "$@"
```

A/B compare:

```bash
OLD_IMAGE="${OLD_IMAGE:?set OLD_IMAGE}"
NEW_IMAGE="${NEW_IMAGE:?set NEW_IMAGE}"

bench/compare.sh \
  --old "${OLD_IMAGE}" \
  --new "${NEW_IMAGE}" \
  --mode "${MODE:-single}" \
  --results-dir "${RESULTS_DIR}" \
  "$@"
```
