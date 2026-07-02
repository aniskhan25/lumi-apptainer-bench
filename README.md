# LUMI Apptainer Benchmark

Benchmark harness for measuring PyTorch and JAX performance on LUMI MI250X GPUs.

Covers:
- Single-node compute (GEMM, kernel mix)
- Single-node DDP step timing
- Two-node allreduce bandwidth sweep
- Two-node DDP step timing
- Runtime and filesystem sanity checks

Results are structured JSON files.

## Repo Layout
- `bench/bench.py`: benchmark CLI
- `bench/tests/`: individual benchmark tests
- `templates/`: LUMI Slurm launch templates
- `scripts/run_benchmarks.sh`: run the standard PyTorch benchmark set

## Clone
```bash
git clone https://github.com/aniskhan25/lumi-apptainer-bench.git
cd lumi-apptainer-bench
```

## Run PyTorch Benchmarks on LUMI
```bash
export PROJECT_NAME=project_462000131
export CONTAINER=/path/to/container.sif
./scripts/run_benchmarks.sh
```

Individual benchmarks:
```bash
# Single-node compute
./templates/single_8g_8r.sh $CONTAINER -- bench/run single --out results/lumi_single.json

# Single-node DDP
./templates/single_8g_8r.sh $CONTAINER -- bench/run ddp --out results/lumi_ddp.json

# Two-node allreduce
export NODES=2
./templates/multi_ng_8rpn.sh $CONTAINER -- bench/run multi --out results/lumi_allreduce.json

# Two-node DDP
export NODES=2
./templates/multi_ng_8rpn.sh $CONTAINER -- bench/run ddp --out results/lumi_ddp_2n.json
```

## Run JAX Benchmarks on LUMI
JAX requires a separate container built from the `feature/jax-lumi` branch of Extending-containers-on-LUMI. The Slurm scripts for JAX are in the scratch directory and use `ntasks-per-node=8` (8 processes × 1 GPU each).

```bash
# Single-GPU GEMM
bench/run jax-single --out results/lumi_jax_single.json

# Two-node allreduce (16 processes × 1 GPU)
bench/run jax-multi --out results/lumi_jax_multi_2n.json

# Single-node DDP (8 processes × 1 GPU, verified allreduce)
bench/run jax-ddp --out results/lumi_jax_ddp.json

# Two-node DDP (16 processes × 1 GPU, verified allreduce)
bench/run jax-ddp-multi --out results/lumi_jax_ddp_multi.json
```

## Metric Glossary
- `GEMM TFLOPS`: raw GPU matrix multiplication throughput
- `GEMM p50 ms`: median time for one matrix multiply
- `KernelMix p50 ms`: median time for a small transformer-like mix of GPU operations
- `Allreduce BW (GB/s)`: how fast data is reduced and exchanged across GPUs or nodes
- `Allreduce Lat (us)`: how long one allreduce operation takes
- `DDP samples/sec`: distributed training throughput for the DDP step benchmark
- `DDP step avg ms`: average end-to-end time for one DDP training step
- `allreduce_verified`: whether the JAX gradient sync was confirmed real via rank-specific gradient check

## JAX vs PyTorch Benchmark Results

Measured on LUMI MI250X (AMD Instinct, 8 GCDs per node) using 2 nodes.
Containers: `lumi-multitorch` (May 13 build) for PyTorch; `jax-lumi.sif` (JAX 0.10.2 + jax_rocm7_plugin) for JAX.
DDP model: N processes × 1 GPU each with `pmap` + cross-process `pmean` (same topology for both frameworks).

### GEMM (bfloat16, 4096×4096, single GPU)

| | TFLOPS | p50 ms |
|--|--------|--------|
| PyTorch | 103.3 | 1.33 |
| JAX | 120.6 | 1.14 |

JAX +17%. XLA JIT produces tighter kernels for square matrix multiply.

### Allreduce (2-node, 16 GPU, Slingshot)

| Size | PyTorch GB/s | JAX GB/s |
|------|-------------|---------|
| 1 KB | 0.013 | 0.004 |
| 64 KB | 0.265 | 0.266 |
| 256 KB | 2.273 | 0.952 |
| 1 MB | 5.097 | 3.006 |

PyTorch/RCCL is faster at raw collective bandwidth. Both converge around 64 KB.

### DDP Step (batch 64, 4096×4096 weight, bfloat16)

| Config | PyTorch samp/s | PyTorch ms | JAX samp/s | JAX ms |
|--------|---------------|-----------|-----------|--------|
| 1-node, 8 GPU | 194,000 | 2.64 | **482,000** | **1.06** |
| 2-node, 16 GPU | 307,000 | 3.34 | **589,000** | **1.74** |

JAX +149% on single-node, +92% on two-node.
Adding a second node costs JAX +0.68 ms (Slingshot allreduce overhead); PyTorch +0.70 ms.

**Why JAX wins on DDP despite losing standalone allreduce:**
XLA compiles the full training step — forward, backward, allreduce, weight update — as one fused program. It can overlap the collective with computation and eliminate intermediate copies. PyTorch treats computation and communication as separate operations and cannot fuse across that boundary.
