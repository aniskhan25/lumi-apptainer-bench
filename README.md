# JAX Benchmarks on Roihu

JAX benchmarks on Roihu (NVIDIA GH200), with PyTorch numbers from LUMI for reference.

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

## Results

### GEMM (bfloat16, 4096×4096, single GPU)

| | LUMI MI250X | Roihu GH200 |
|--|------------|------------|
| TFLOPS | 120.6 | 530.9 |
| p50 ms | 1.14 | 0.26 |

Roihu GH200 is 4.4× faster.

### Allreduce (cross-node)

Each MI250X GPU exposes 2 GCDs, so LUMI has 8 ranks per node; Roihu's GH200 has 4 ranks per node. To get the same 16 ranks for a fair comparison, LUMI needs 2 nodes and Roihu needs 4.

Interconnects: LUMI uses HPE Slingshot (200 Gb/s); Roihu uses InfiniBand NDR (800 Gb/s).

**2-node runs — different rank counts (LUMI 16, Roihu 8), not directly comparable:**

| Size | LUMI JAX GB/s | Roihu JAX GB/s |
|------|--------------|---------------|
| 1 KB | 0.004 | 0.005 |
| 64 KB | 0.266 | 0.128 |
| 256 KB | 0.952 | 1.044 |
| 1 MB | 3.006 | 3.714 |

**16-rank runs — directly comparable (LUMI: 2 nodes × 8 GCDs, Roihu: 4 nodes × 4 GPUs):**

| Size | LUMI JAX GB/s | Roihu JAX GB/s |
|------|--------------|---------------|
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

Roihu leads up to 16 MB, then LUMI takes over despite having 4× less interconnect bandwidth. The reason is node count: LUMI's 16 ranks fit on 2 nodes, so only 1 of the 15 ring steps crosses the network — the rest are intra-node over XGMI. Roihu needs 4 nodes, so 4 ring steps cross InfiniBand every pass. At large messages, fewer network hops matters more than faster links.

### DDP Step (4096×4096 weight, bfloat16, allreduce verified)

**samp/s = ranks × batch / step_time**, so it reflects both step speed and rank count. Since Roihu has half the ranks per node, ms is the fairer metric to compare step efficiency.

**Single batch size (batch=64 per rank):**

| Config | LUMI ranks | LUMI samp/s | LUMI ms | Roihu ranks | Roihu samp/s | Roihu ms |
|--------|-----------|------------|--------|------------|-------------|--------|
| 1-node | 8 | 482,000 | 1.06 | 4 | 679,000 | 0.38 |
| 2-node | 16 | 589,000 | 1.74 | 8 | 558,000 | 0.92 |

On 1-node, Roihu is 2.8× faster per step. Scaling to 2 nodes costs more on Roihu (+0.54ms, +142%) than LUMI (+0.68ms, +64%) — at batch=64 the step is short enough that the cross-node allreduce is the dominant cost on both systems.

**Batch size sweep (2-node):**

| Batch per rank | LUMI ms | Roihu ms | Roihu advantage |
|---------------|--------|---------|----------------|
| 64 | 1.74 | 0.92 | 1.9× |
| 256 | 1.78 | 0.87 | 2.0× |
| 1024 | 1.94 | 0.88 | 2.2× |

Roihu's step time is flat across batch sizes — it is allreduce-bound and compute finishes faster than the network sync. LUMI's step time grows because its slower MI250X compute starts adding measurable time at larger batches. As a result, Roihu's advantage increases with batch size: the faster GH200 compute keeps communication the bottleneck, while LUMI shifts toward compute-bound at larger batches.
