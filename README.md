# JAX Benchmarks on Roihu

JAX benchmarks on Roihu (NVIDIA GH200). All LUMI numbers in this branch are also JAX, collected on the same benchmark suite, so the comparison is framework-consistent.

## System

Roihu GPU nodes: NVIDIA GH200 Grace Hopper superchips, 4 per node (96 GiB HBM3 each), 72 ARM cores per GPU, InfiniBand NDR 4×200 Gb/s inter-node.

## Setup

Clone the repo to your scratch directory on Roihu:

```bash
cd /scratch/project_2014553/$USER
git clone https://github.com/aniskhan25/lumi-apptainer-bench.git
git checkout feature/roihu-benchmarks
cd lumi-apptainer-bench
```

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

## Changing parameters

**Slurm / topology** — set before calling the template:

```bash
export NODES=4              # number of nodes (default: 1 for single, 2 for multi)
export GPUS_PER_NODE=4      # GPUs per node (default: 4)
export TIME_LIMIT=01:00:00  # wall time (default: 00:30:00)
export PARTITION=gpumedium  # partition (default: gputest for single, gpumedium for multi)
```

**Allreduce message sizes** — comma-separated bytes, passed after `--`:

```bash
./templates/roihu_multi.sh -- bench/run jax-multi \
  --message-sizes 1048576,67108864,268435456 \
  --out results/custom.json
```

**DDP batch and model size**:

```bash
./templates/roihu_single.sh -- bench/run jax-ddp \
  --batch-size 128 --input-size 8192 --output-size 8192 \
  --out results/custom.json
```

**Iteration count** (applies to all subcommands):

```bash
... bench/run jax-multi --iters 20 --out results/custom.json
```

Default iteration counts are low (5 for GEMM/allreduce, 10 for DDP). Results below reflect those defaults and do not include variance or multi-run statistics.

## Results

### GEMM (JAX, bfloat16, 4096×4096, single GPU, 5 iters)

| | LUMI MI250X | Roihu GH200 |
|--|------------|------------|
| TFLOPS | 120.6 | 530.9 |
| p50 ms | 1.14 | 0.26 |

Roihu GH200 is 4.4× faster on this single-GPU GEMM kernel.

### Allreduce (JAX, cross-node)

**Bandwidth is reported as payload GB/s = message\_size / elapsed\_time.** This is an application-level throughput metric, not raw fabric bandwidth or algorithmic bus bandwidth (which would account for the collective's data-movement factor).

Each MI250X GPU exposes 2 GCDs, so LUMI has 8 ranks per node; Roihu's GH200 has 4 ranks per node. Both systems have comparable node-level injection bandwidth (~800 Gb/s: LUMI has 4 Slingshot-11 links per node at 200 Gb/s each; Roihu has 4 InfiniBand NDR links at 200 Gb/s each). The interconnects differ in technology, topology, and latency characteristics, but not in raw per-node bandwidth.

**2-node runs — different rank counts (LUMI 16, Roihu 8), not rank-count comparable:**

| Size | LUMI JAX payload GB/s | Roihu JAX payload GB/s |
|------|----------------------|----------------------|
| 1 KB | 0.004 | 0.005 |
| 64 KB | 0.266 | 0.128 |
| 256 KB | 0.952 | 1.044 |
| 1 MB | 3.006 | 3.714 |

**16-rank runs — rank-count comparable (LUMI: 2 nodes × 8 GCDs, Roihu: 4 nodes × 4 GPUs):**

Note: rank count is equalized but node count and network topology differ (2 vs 4 nodes), so results reflect a combined effect of GPU architecture, collective implementation, rank placement, and topology — not a pure interconnect comparison.

| Size | LUMI JAX payload GB/s | Roihu JAX payload GB/s |
|------|----------------------|----------------------|
| 1 KB | 0.004 | 0.001 |
| 4 KB | 0.017 | 0.019 |
| 16 KB | 0.067 | 0.068 |
| 64 KB | 0.252 | 0.283 |
| 256 KB | 0.892 | 1.081 |
| 1 MB | 3.054 | 3.803 |
| 4 MB | 9.742 | 10.625 |
| 16 MB | 23.383 | 25.698 |
| 64 MB | 36.812 | 32.658 |
| 256 MB | 44.786 | 37.469 |

Roihu leads up to 16 MB; LUMI overtakes at 64 MB and extends its lead at 256 MB. One plausible explanation is node count: LUMI uses 2 physical nodes while Roihu uses 4, which likely results in fewer inter-node communication steps if the collective uses a hierarchical or topology-aware algorithm (intra-node reduction first, then inter-node). However, the exact collective path depends on the XLA/NCCL/RCCL implementation and rank ordering, and has not been verified with profiler output. This should be treated as a hypothesis.

### DDP Step (JAX, 4096×4096 weight, bfloat16, allreduce verified, 10 iters)

**samp/s = ranks × batch / step\_time** — it reflects both step speed and rank count. Since Roihu has half the ranks per node, these two metrics answer different questions: **ms per step** measures per-replica efficiency; **samp/s** measures node-level throughput for users allocating by node. Both are reported below.

**Single batch size (batch=64 per rank):**

| Config | LUMI ranks | LUMI samp/s | LUMI ms | Roihu ranks | Roihu samp/s | Roihu ms |
|--------|-----------|------------|--------|------------|-------------|--------|
| 1-node | 8 | 482,000 | 1.06 | 4 | 679,000 | 0.38 |
| 2-node | 16 | 589,000 | 1.74 | 8 | 558,000 | 0.92 |

On 1-node, Roihu is 2.8× faster per step. Scaling to 2 nodes adds more overhead on Roihu (+0.54ms, +142%) than LUMI (+0.68ms, +64%) relative to their 1-node baselines — at batch=64 the allreduce cost is a large fraction of total step time on both systems.

**Batch size sweep (2-node, step time only):**

| Batch per rank | LUMI ms | Roihu ms | Roihu advantage |
|---------------|--------|---------|----------------|
| 64 | 1.74 | 0.92 | 1.9× |
| 256 | 1.78 | 0.87 | 2.0× |
| 1024 | 1.94 | 0.88 | 2.2× |

Roihu's step time is nearly flat across batch sizes, suggesting the step remains allreduce-bound — GH200 compute finishes quickly enough that increasing batch barely changes total step time. LUMI's step time grows slowly, indicating MI250X compute starts contributing at larger batches. Roihu's per-step advantage therefore increases with batch size. These results use the default 10 iterations and do not include variance estimates.
