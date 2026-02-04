# LUMI Apptainer Benchmark

Minimal benchmark suite for comparing old vs new LUMI containers with
standardized `srun`/`sbatch` templates and structured JSON output. The
goal is to validate container changes against a small, repeatable set
of ML-relevant kernels and collectives.

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

See `implementation_plan.md` for the full plan.
