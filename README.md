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

## Latest Results Summary (2026-02-04)

**Single-node tests**
| Run | Nodes | Tasks | GPU/node | GEMM TFLOPS | GEMM p50 ms | KernelMix p50 ms | Timestamp UTC |
|---|---|---|---|---|---|---|---|
| lumi_single.json | 1 | 8 | 8 | 104.518 | 1.315 | 0.188 | 2026-02-04T07:40:44Z |
| lumi_single_16r.json | 1 | 16 | 8 | 105.985 | 1.297 | 0.173 | 2026-02-04T07:51:09Z |

**Multi-node allreduce tests**
| Run | Nodes | Tasks | GPU/node | Avg BW (GB/s) | Avg Lat (us) | Timestamp UTC |
|---|---|---|---|---|---|---|
| lumi_allreduce.json | 2 | 16 | 8 | 0.014 | 14928.837 | 2026-02-04T07:47:07Z |
| lumi_multi.json | 2 | 16 | 8 | 0.019 | 10661.517 | 2026-02-04T07:49:53Z |

**Comparison deltas**
| Run | GEMM TFLOPS Δ% | KernelMix p50 Δ% | Allreduce BW Δ% | Allreduce Lat Δ% | Regression count | Timestamp UTC |
|---|---|---|---|---|---|---|
| lumi_single_16r_compare/delta.json | -0.571 | 10.456 |  |  | 0 | 2026-02-04T08:06:51Z |
| lumi_multi_compare/delta.json |  |  | -63.633 | 232.856 | 2 | 2026-02-04T08:02:57Z |

**Allreduce per-message-size (lumi_multi_compare)**
| Size (bytes) | Old BW (GB/s) | New BW (GB/s) | BW Δ% | Old Lat (us) | New Lat (us) | Lat Δ% |
|---|---|---|---|---|---|---|
| 1024 | 0.001 | 0.000 | -25.836 | 2000.264 | 2697.096 | 34.837 |
| 4096 | 0.001 | 0.000 | -75.960 | 3337.143 | 13881.446 | 315.968 |
| 16384 | 0.004 | 0.001 | -83.825 | 4400.247 | 27204.429 | 518.248 |
| 65536 | 0.020 | 0.003 | -83.236 | 3330.874 | 19868.904 | 496.507 |
| 262144 | 0.026 | 0.016 | -38.022 | 10067.695 | 16244.111 | 61.349 |
| 1048576 | 0.176 | 0.062 | -64.824 | 5941.163 | 16889.714 | 184.283 |

Notes:
1. Allreduce bandwidth values are in **GB/s** (bytes/sec / 1e9).
2. See `bench_results/` for full JSON payloads.
