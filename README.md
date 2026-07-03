# JAX Benchmarks on Roihu

This branch ports the JAX benchmarks from `feature/jax-benchmarks` (LUMI MI250X) to Roihu (NVIDIA GH200).
PyTorch comparison numbers are taken from the LUMI run (main branch) — nothing here re-collects them.

## System

Roihu GPU nodes: NVIDIA GH200 Grace Hopper superchips, 4 per node (96 GiB HBM3 each), 72 ARM cores per GPU, InfiniBand NDR 4×200 Gb/s inter-node.

## Prerequisites

JAX 0.10.2 is preinstalled on Roihu as a TYKKY module. Loading it sets `$SIF` and `APPTAINER_NV=true`:

```bash
module load python-jax
apptainer exec --bind="$(csc-common-bind)" $SIF python3 ...
```

## Run JAX Benchmarks on Roihu

```bash
export PROJECT_NAME=project_2014553
```

Single-node GEMM (uses `gputest` partition by default):
```bash
./templates/roihu_single.sh -- bench/run jax-single --out results/roihu_jax_single.json
```

Two-node allreduce sweep:
```bash
export NODES=2
./templates/roihu_multi.sh -- bench/run jax-multi --out results/roihu_jax_multi_2n.json
```

Single-node DDP (4 processes × 1 GPU, allreduce verified):
```bash
./templates/roihu_single.sh -- bench/run jax-ddp --out results/roihu_jax_ddp.json
```

Two-node DDP (8 processes × 1 GPU, allreduce verified):
```bash
export NODES=2
./templates/roihu_multi.sh -- bench/run jax-ddp-multi --out results/roihu_jax_ddp_multi.json
```

## Results

### GEMM (bfloat16, 4096×4096, single GPU)

| | LUMI MI250X | Roihu GH200 |
|--|------------|------------|
| TFLOPS | 120.6 | 530.9 |
| p50 ms | 1.14 | 0.26 |

### Allreduce (2-node, cross-node)

Two differences make these numbers hard to compare directly. First, LUMI ran with 2 nodes × 8 GPUs = 16 total ranks; Roihu ran with 2 nodes × 4 GPUs = 8 total ranks. Ring-allreduce traffic per rank scales with `(N-1)/N × message_size`, so 16-rank and 8-rank runs are not measuring the same operation at the same message size. Second, the interconnects are different technologies: LUMI uses HPE Slingshot (200 Gb/s), Roihu uses InfiniBand NDR (4×200 Gb/s). What the table shows is what each system achieved on its own 2-node run, not a network speed comparison.

| Size | LUMI JAX GB/s | Roihu JAX GB/s |
|------|--------------|---------------|
| 1 KB | 0.004 | 0.005 |
| 64 KB | 0.266 | 0.128 |
| 256 KB | 0.952 | 1.044 |
| 1 MB | 3.006 | 3.714 |

### DDP Step (batch 64, 4096×4096 weight, bfloat16, allreduce verified)

| Config | LUMI JAX samp/s | LUMI JAX ms | Roihu JAX samp/s | Roihu JAX ms |
|--------|----------------|------------|-----------------|-------------|
| 1-node | 482,000 | 1.06 | 679,000 | 0.38 |
| 2-node | 589,000 | 1.74 | 558,000 | 0.92 |

Samples/s is `(GPUs × 64) / step_time`, so it reflects both speed and GPU count. Roihu's 2-node step is faster (0.92ms vs 1.74ms) but processes half the samples per step (8 GPUs × 64 = 512) compared to LUMI (16 GPUs × 64 = 1024), which is why the aggregate throughput ends up similar despite the faster per-step time. The step time is the more meaningful comparison here: Roihu GH200 is ~4.4× faster on GEMM and ~2.8× faster per step on single-node DDP. The 2-node step time degrades more sharply on Roihu relative to 1-node (0.38→0.92ms, +142%) than on LUMI (1.06→1.74ms, +64%), suggesting cross-node allreduce overhead is proportionally larger on Roihu given its much faster local compute.
