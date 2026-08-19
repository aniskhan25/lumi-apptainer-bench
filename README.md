# LUMI Apptainer Benchmark

Two modes:

**Comparison** — benchmark a new LUMI container against a known stable one under the same
launch setup. Outputs JSON plus `delta.json` percentage deltas. This is what `main` does.

**Validation** — check a single container against absolute release gates. There is no
baseline run, so each metric is checked against a declared limit in `manifests/gates/`
rather than against a previous result. Added on this branch to validate the `-latest`
container against the issues raised in the `project_465003047` experience report.

### Findings

**Start here: [`docs/FINDINGS.md`](docs/FINDINGS.md)** — consolidated, evidence-backed
response to the report, with job IDs and the container digest.

Container under test: `lumi-multitorch-full-u24r70f21m50t210-20260513_121430`
(`sha256 f0de72f4…`), up to 16 nodes / 128 ranks, pure PyTorch (no Megatron-Core).

| # | Report finding | Status |
| --- | --- | --- |
| 4.1 | Inter-node all-to-all regression (`PTLTE_NOT_FOUND`) | Not reproduced — hypotheses exhausted |
| 4.2 | Usable HBM below nameplate | **Confirmed and quantified** |
| 4.3 | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` from `torch.compile` | Not tested |
| 4.4 | 32-rank expert all-to-all fails | Not reproduced in its own topology |
| 4.5 | Very long collective bootstrap | Not reproduced at ≤128 ranks; confound found |
| 4.6 | Two jobs on one Lustre dataset | Not tested |
| 4.7 | Lustre JIT caches corrupt under rank pressure | **Reproduced, root cause identified** |
| 4.8 | `torch_dist` checkpoint hang | Not tested (needs Megatron) |
| 4.9 | QOS limit discoverability | Out of scope (Slurm policy) |
| 4.10 | Minor items | Partly addressed |

Headline results:

- **§4.7 reproduced.** Inductor's cache reader lists the cache directory and `open()`s every
  entry, including other ranks' in-flight `.{pid}.{tid}.tmp` files, which get renamed away
  mid-read (`codecache.py:1040`). On Lustre with 128 ranks sharing one directory the race is
  frequently lost; per-node `/tmp` fixes it completely, and Lustre is also ~1.5× slower to
  compile.
- **§4.2 explained.** ~630–650 MiB of device memory per RCCL communicator, invisible to
  PyTorch (`memory_allocated()` reports 0.0 MiB throughout). At 8–10 communicators that is
  ~6.8 GiB, matching the ~7 GiB gap the report measured.
- **§4.1/§4.4 not reproduced**, including EP=32 with four concurrent meshes at 128 ranks.
- **Eight issues found that the report does not raise**, including `mpi4py` in `full`/`plus`
  silently running on Intel MPI (no `cxi` provider) and Intel's shims shadowing a working
  `fi_info` — both from one transitive `oneccl` → `impi-rt` dependency; the container's
  GPU-binding fix reaching no documented workflow (it needs `apptainer run` *and* two
  undocumented variables); and `MIOPEN_USER_DB` in 17 LUMI-AI-Guide scripts being a variable
  MIOpen does not read.

### Reports

| Document | Contents |
| --- | --- |
| [`FINDINGS.md`](docs/FINDINGS.md) | **Consolidated response and recommendations** |
| [`ESCALATION.md`](docs/ESCALATION.md) | What to file upstream, and what is already reported |
| [`issues/`](docs/issues/) | Ready-to-paste issue drafts (not filed) |
| [`VALIDATION.md`](docs/VALIDATION.md) | How to run the gates |
| [`PHASE0_FINDINGS.md`](docs/PHASE0_FINDINGS.md) | Desk analysis of shipped release artifacts |
| [`PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md) | Capability probe, `exec`-vs-`run` binding, allocator |
| [`PHASE2_RESULTS.md`](docs/PHASE2_RESULTS.md) | All-to-all at EP=8 / EP=16, 2 nodes |
| [`PHASE3_RESULTS.md`](docs/PHASE3_RESULTS.md) | EP=32, 4 nodes |
| [`EP32_16NODE_RESULTS.md`](docs/EP32_16NODE_RESULTS.md) | EP=32, 128 ranks, four concurrent meshes |
| [`PHASE4_RESULTS.md`](docs/PHASE4_RESULTS.md) | JIT cache under rank pressure |
| [`PHASE5_COMM_COUNT_RESULTS.md`](docs/PHASE5_COMM_COUNT_RESULTS.md) | Communicator count and hidden memory |

The comparison scope stays narrow on purpose:
- single-node compute
- single-node DDP step timing
- two-node allreduce
- two-node DDP step timing
- runtime and filesystem sanity checks

Validation adds:
- a startup capability probe (what the job actually loaded, per rank)
- all-to-all correctness and bandwidth at EP=8 / 16 / 32
- safe-by-default JIT cache placement, with a mode to reproduce the unsafe one

## Repo Layout
- `bench/bench.py`: benchmark CLI
- `bench/tests/`: individual benchmark tests
- `bench/compare.sh`: run old and new containers and write `delta.json`
- `bench/scripts/eval_gates.py`: check one results file against a gate spec
- `manifests/gates/`: gate specs (thresholds and correctness invariants)
- `templates/`: LUMI Slurm launch templates
- `scripts/run_benchmarks.sh`: run the standard comparison set
- `scripts/run_validation.sh`: run the validation gates against one container
- `scripts/summarize_results.py`: print Markdown tables from a results directory
- `docs/`: findings records

## Clone
```bash
git clone https://github.com/aniskhan25/lumi-apptainer-bench.git
cd lumi-apptainer-bench
```

## Run The Full Benchmark On LUMI
Set the required environment:
```bash
export PROJECT_NAME=project_462000131
```

Optional overrides (defaults are set in the script):
```bash
export OLD_CONTAINER=/path/to/old.sif
export NEW_CONTAINER=/path/to/new.sif
export PARTITION=standard-g
export ACCOUNT="$PROJECT_NAME"
export RESULTS_ROOT=/scratch/$PROJECT_NAME/$USER/bench_results
```

Run the standard benchmark set:
```bash
./scripts/run_benchmarks.sh
```

This writes results under:
```bash
${RESULTS_ROOT:-/scratch/$PROJECT_NAME/$USER/bench_results}
```

## Run Individual Benchmarks
Single-node compute:
```bash
./templates/single_8g_8r.sh /path/to/container.sif -- bench/run single --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_single.json
```

Single-node DDP:
```bash
./templates/single_8g_8r.sh /path/to/container.sif -- bench/run ddp --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_ddp.json
```

Two-node allreduce:
```bash
export NODES=2
./templates/allreduce_sweep.sh /path/to/container.sif -- bench/run multi --allreduce --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_allreduce.json
```

Two-node DDP:
```bash
export NODES=2
./templates/multi_ng_8rpn.sh /path/to/container.sif -- bench/run ddp --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_ddp_2n.json
```

Sanity check:
```bash
./templates/filesystem.sh /path/to/container.sif -- bench/run check --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_check.json
```

## Compare Two Containers
Use the same template and benchmark mode for both containers.

Example:
```bash
export BENCH_TEMPLATE=./templates/multi_ng_8rpn.sh
./bench/compare.sh \
  --old "$OLD_CONTAINER" \
  --new "$NEW_CONTAINER" \
  --mode ddp \
  --results-dir /scratch/$PROJECT_NAME/$USER/bench_results/lumi_ddp_2n_compare
```

The main verdict is in `delta.json`.

## Summarize Results
Generate Markdown tables from an existing results directory:
```bash
python3 ./scripts/summarize_results.py /scratch/$PROJECT_NAME/$USER/bench_results
```

## Expected Outputs
Standard run files:
- `lumi_single.json`
- `lumi_ddp.json`
- `lumi_single_16r.json`
- `lumi_allreduce.json`
- `lumi_multi.json`
- `lumi_ddp_2n.json`
- `lumi_check.json`

Comparison directories:
- `lumi_check_compare/`
- `lumi_ddp_compare/`
- `lumi_multi_compare/`
- `lumi_ddp_2n_compare/`
- `lumi_single_16r_compare/`

## How To Read The Results
Read `delta.json` first.

Important fields:
- `metrics`: old value, new value, and percent delta for each metric
- `regressions`: metrics that crossed the configured threshold
- `regression_count`: number of flagged regressions

Typical interpretation:
- stable single-node compute with worse multi-node metrics suggests a communication or runtime issue
- stable allreduce with worse DDP step time suggests overhead outside the collective itself
- zero `regression_count` means no threshold-defined regressions were detected

## Metric Glossary
- `DDP`: PyTorch `DistributedDataParallel`, where each rank trains the same model and gradients are synchronized across ranks
- `GEMM TFLOPS`: raw GPU matrix multiplication throughput
- `GEMM p50 ms`: median time for one matrix multiply
- `KernelMix p50 ms`: median time for a small transformer-like mix of GPU operations
- `Allreduce BW (GB/s)`: how fast data is reduced and exchanged across GPUs or nodes
- `Allreduce Lat (us)`: how long one allreduce operation takes
- `DDP samples/sec`: distributed training throughput for the DDP step benchmark
- `DDP step avg ms`: average end-to-end time for one DDP training step
- `DDP step p95 ms`: tail latency for the DDP training step
- `Check`: sanity test for GPU visibility and writable cache paths

## Comparison Methodology
This repo is for fair container assessment, not maximum one-off tuning.

Keep these fixed between old and new runs:
- partition
- node count
- template
- benchmark mode
- benchmark parameters
- launcher environment

Use the same templates for both images and compare the generated `delta.json` files rather than isolated raw numbers.
