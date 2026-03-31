# LUMI Apptainer Benchmark

Minimal benchmark suite for comparing old vs new LUMI containers with
standardized `srun`/`sbatch` templates and structured JSON output. The
goal is to validate container changes against a small, repeatable set
of ML-relevant kernels and collectives.

## Getting Started
Clone the repository:
```bash
git clone https://github.com/aniskhan25/lumi-apptainer-bench.git
cd lumi-apptainer-bench
```

**What’s in this repo**
1. `bench/bench.py`: CLI entrypoint that emits JSON with system + test results.
2. `bench/tests/`: small, focused tests (`gemm`, `kernel_mix`, `allreduce`, `check`).
3. `templates/`: Slurm templates for common launch patterns.
4. `bench_results/`: captured runs + comparison deltas.
5. `bench/compare.sh`: helper for comparing two result JSONs.

## Running
Use the templates and point at the container image, for example:
```bash
./templates/single_8g_8r.sh /path/to/container.sif -- bench/run single --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_single.json
```

### Run The Full Benchmark On LUMI
The repository includes a runbook script that launches the standard benchmark set and the old-vs-new comparisons.

Set the required environment:
```bash
export PROJECT_NAME=project_462000131
export PARTITION=standard-g
export ACCOUNT="${PROJECT_NAME}"
```

Review the container paths at the top of [`scripts/run_benchmarks.sh`](/Users/anisrahm/Documents/lumi-apptainer-bench/scripts/run_benchmarks.sh), then run:
```bash
./scripts/run_benchmarks.sh
```

This runs:
1. Single-node compute benchmarks.
2. Single-node DDP benchmark.
3. Two-node allreduce and DDP benchmarks.
4. Filesystem check.
5. Old-vs-new comparison jobs.

Results are written under:
```bash
/scratch/$PROJECT_NAME/$USER/bench_results
```

To generate Markdown summary tables from an existing results directory:
```bash
python3 ./scripts/summarize_results.py /scratch/$PROJECT_NAME/$USER/bench_results
```

If you are running from inside this repo with local copied results, you can also use:
```bash
python3 ./scripts/summarize_results.py bench_results
```

Expected result files:
1. `lumi_single.json`
2. `lumi_ddp.json`
3. `lumi_single_16r.json`
4. `lumi_allreduce.json`
5. `lumi_multi.json`
6. `lumi_ddp_2n.json`
7. `lumi_check.json`

Expected comparison directories:
1. `lumi_check_compare/`
2. `lumi_ddp_compare/`
3. `lumi_multi_compare/`
4. `lumi_ddp_2n_compare/`
5. `lumi_single_16r_compare/`

### How To Read The Results
The main verdict lives in each comparison directory's `delta.json`.

Look at:
1. `metrics`: old value, new value, and percent change.
2. `regressions`: metrics that crossed the configured threshold.
3. `regression_count`: number of flagged regressions.

Typical interpretation:
1. `single_gemm_tflops`: single-GPU compute throughput.
2. `single_kernel_mix_p50_ms`: latency of the mixed kernel microbenchmark.
3. `multi_allreduce_bw_avg_gbps`: average allreduce bandwidth.
4. `multi_allreduce_lat_avg_us`: average allreduce latency.
5. `ddp_samples_per_sec`: DDP step throughput.
6. `ddp_step_time_ms_avg`: average DDP step time.

## Comparison Methodology
This repository is intended to answer a narrow question: does a new container perform at a similar level to a known stable container on the same LUMI infrastructure?

To keep the comparison fair:
1. Use the same partition, node count, and template for the old and new container.
2. Keep benchmark parameters fixed across runs, including message sizes, DDP settings, and output paths.
3. Use the same launcher defaults, including CPU binding, CPU allocation, and network environment.
4. Compare old and new results from the generated `delta.json` files rather than isolated raw numbers.

Practical guidance:
1. Treat the templates and [`scripts/run_benchmarks.sh`](/Users/anisrahm/Documents/lumi-apptainer-bench/scripts/run_benchmarks.sh) as the baseline execution path.
2. Avoid adding workload-specific tuning knobs when validating containers, unless the same knobs are applied to both images.
3. If a metric is noisy, repeat the comparison rather than changing tuning parameters mid-stream.

This repo is for fair container assessment, not for one-off maximum performance tuning.

## Latest Results Summary (2026-03-31)

**Single-node tests**
| Run | Nodes | Tasks | GPU/node | GEMM TFLOPS | GEMM p50 ms | KernelMix p50 ms | Timestamp UTC |
|---|---|---|---|---|---|---|---|
| lumi_single.json | 1 | 8 | 8 | 105.427 | 1.304 | 0.172 | 2026-03-31T09:50:14Z |
| lumi_single_16r.json | 1 | 16 | 8 | 105.436 | 1.304 | 0.173 | 2026-03-31T10:05:55Z |

**DDP tests**
| Run | Nodes | Tasks | GPU/node | Samples/sec | Step avg ms | Step p95 ms | Timestamp UTC |
|---|---|---|---|---|---|---|---|
| lumi_ddp.json | 1 | 8 | 8 | 187476.029 | 2.731 | 2.795 | 2026-03-31T10:00:49Z |
| lumi_ddp_2n.json | 2 | 16 | 8 | 278065.567 | 3.683 | 4.267 | 2026-03-31T10:29:34Z |

**Multi-node allreduce tests**
| Run | Nodes | Tasks | GPU/node | Avg BW (GB/s) | Avg Lat (us) | Timestamp UTC |
|---|---|---|---|---|---|---|
| lumi_allreduce.json | 2 | 16 | 8 | 1.057 | 281.590 | 2026-03-31T10:16:32Z |
| lumi_multi.json | 2 | 16 | 8 | 0.992 | 138.161 | 2026-03-31T10:27:02Z |

**Comparison deltas**
| Run | GEMM TFLOPS Δ% | KernelMix p50 Δ% | Allreduce BW Δ% | Allreduce Lat Δ% | DDP samples Δ% | DDP step Δ% | Regression count | Timestamp UTC |
|---|---|---|---|---|---|---|---|---|
| lumi_check_compare/delta.json |  |  |  |  |  |  | 0 | 2026-03-31T10:55:43Z |
| lumi_ddp_compare/delta.json |  |  |  |  | 8.048 | -7.449 | 0 | 2026-03-31T11:02:28Z |
| lumi_multi_compare/delta.json |  |  | 194.367 | -61.429 |  |  | 0 | 2026-03-31T11:16:33Z |
| lumi_ddp_2n_compare/delta.json |  |  |  |  | 295.510 | -74.716 | 0 | 2026-03-31T11:37:26Z |
| lumi_single_16r_compare/delta.json | -1.836 | 7.560 |  |  |  |  | 0 | 2026-03-31T11:54:57Z |

Notes:
1. Allreduce bandwidth values are in **GB/s** (bytes/sec / 1e9).
2. See `bench_results/` for full JSON payloads.

### Result Analysis
The new container passes the benchmark comparison on LUMI.

Key observations:
1. Single-node compute is effectively flat. GEMM stays around `105.4 TFLOPS` and no single-node regression is flagged.
2. Single-node DDP improves modestly, with `+8.048%` samples/sec and `-7.449%` average step time.
3. Multi-node allreduce improves significantly, with `+194.367%` bandwidth and `-61.429%` latency.
4. Two-node DDP improves substantially, with `+295.510%` samples/sec and `-74.716%` average step time.

Interpretation:
1. The new container preserves single-node GPU compute performance.
2. The biggest gains are in inter-node communication and distributed step execution.
3. For this benchmark suite, the new container performs at a similar or better level than the previous stable container on LUMI.
