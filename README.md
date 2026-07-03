# JAX Benchmarks on Roihu

JAX benchmarks on Roihu (NVIDIA GH200). All LUMI numbers are also JAX on the same benchmark suite.

## System

Roihu: 4 × NVIDIA GH200 per node, 96 GiB HBM3 each, InfiniBand NDR 4×200 Gb/s.
LUMI: 4 × AMD MI250X per node (8 GCDs), HPE Slingshot 4×200 Gb/s.
Both have ~800 Gb/s node injection bandwidth.

## Setup

```bash
cd /scratch/project_2014553/$USER
git clone https://github.com/aniskhan25/lumi-apptainer-bench.git
git checkout feature/roihu-benchmarks
cd lumi-apptainer-bench
export PROJECT_NAME=project_2014553
```

JAX 0.10.2 is preinstalled as a module:

```bash
module load python-jax   # sets $SIF and APPTAINER_NV=true
```

## Run

Single-node GEMM (`gputest` partition):
```bash
./templates/roihu_single.sh -- bench/run jax-single --out results/jax_single.json
```

Two-node allreduce sweep:
```bash
export NODES=2
./templates/roihu_multi.sh -- bench/run jax-multi --out results/jax_multi.json
```

Single-node DDP (4 processes × 1 GPU):
```bash
./templates/roihu_single.sh -- bench/run jax-ddp --out results/jax_ddp.json
```

Two-node DDP (8 processes × 1 GPU):
```bash
export NODES=2
./templates/roihu_multi.sh -- bench/run jax-ddp-multi --out results/jax_ddp_multi.json
```

Key knobs (set before calling the template):

| Variable | Default | Description |
|----------|---------|-------------|
| `NODES` | 1 or 2 | Node count |
| `GPUS_PER_NODE` | 4 | GPUs per node |
| `PARTITION` | gputest / gpumedium | Slurm partition |
| `TIME_LIMIT` | 00:30:00 | Wall time |

Benchmark knobs (passed after `--`): `--batch-size`, `--input-size`, `--output-size`, `--message-sizes`, `--iters`.

## Results

All results use default iteration counts (5 for GEMM/allreduce, 10 for DDP). No variance estimates.

### GEMM (bfloat16, 4096×4096, single GPU)

| | LUMI MI250X | Roihu GH200 |
|--|------------|------------|
| TFLOPS | 120.6 | 530.9 |
| p50 ms | 1.14 | 0.26 |

GH200 is 4.4× faster on this single-GPU kernel.

### Allreduce (cross-node, payload GB/s = message\_size / elapsed\_time)

Each MI250X exposes 2 GCDs, so LUMI has 8 ranks per node vs Roihu's 4. A 2-node LUMI run has 16 ranks; a 2-node Roihu run has 8. The 16-rank table below equalises rank count (LUMI 2 nodes, Roihu 4 nodes) but node count and topology still differ, so results reflect a combined effect of hardware, collective implementation, and rank placement.

**2-node (LUMI 16 ranks, Roihu 8 ranks):**

| Size | LUMI GB/s | Roihu GB/s |
|------|----------|-----------|
| 1 KB | 0.004 | 0.005 |
| 64 KB | 0.266 | 0.128 |
| 256 KB | 0.952 | 1.044 |
| 1 MB | 3.006 | 3.714 |

**16 ranks (LUMI: 2 nodes × 8 GCDs, Roihu: 4 nodes × 4 GPUs):**

| Size | LUMI GB/s | Roihu GB/s |
|------|----------|-----------|
| 1 KB | 0.004 | 0.001 |
| 64 KB | 0.252 | 0.283 |
| 1 MB | 3.054 | 3.803 |
| 4 MB | 9.742 | 10.625 |
| 16 MB | 23.383 | 25.698 |
| 64 MB | 36.812 | 32.658 |
| 256 MB | 44.786 | 37.469 |

Roihu leads up to ~16 MB; LUMI overtakes at larger sizes. LUMI uses fewer physical nodes for the same rank count, which likely reduces inter-node traffic in the collective — though the exact cause depends on the XLA collective implementation and has not been verified with a profiler.

### DDP Step (bfloat16, 4096×4096 weight, allreduce verified)

**samp/s** = ranks × batch / step\_time (reflects both speed and rank count).
**ms** = per-step latency (per-replica efficiency, independent of rank count).

**batch=64 per rank:**

| Config | LUMI ranks | LUMI ms | Roihu ranks | Roihu ms |
|--------|-----------|--------|------------|---------|
| 1-node | 8 | 1.06 | 4 | 0.38 |
| 2-node | 16 | 1.74 | 8 | 0.92 |

**2-node, varying batch:**

| Batch | LUMI ms | Roihu ms | Roihu faster by |
|-------|--------|---------|----------------|
| 64 | 1.74 | 0.92 | 1.9× |
| 256 | 1.78 | 0.87 | 2.0× |
| 1024 | 1.94 | 0.88 | 2.2× |

Roihu's step time stays flat as batch grows — the step is allreduce-bound and GH200 compute finishes well within the sync window. LUMI's step time grows because MI250X compute becomes a measurable fraction at larger batches. Roihu's per-step advantage therefore increases with batch size.
