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

| | PyTorch | JAX | Δ |
|--|---------|-----|---|
| TFLOPS | 103.3 | 120.6 | **+17%** |
| p50 ms | 1.33 | 1.14 | **-14%** |

### Allreduce (2-node, 16 GPU, Slingshot)

| Size | PyTorch GB/s | JAX GB/s | Δ |
|------|-------------|---------|---|
| 1 KB | 0.013 | 0.004 | -69% |
| 64 KB | 0.265 | 0.266 | 0% |
| 256 KB | 2.273 | 0.952 | -58% |
| 1 MB | 5.097 | 3.006 | **-41%** |

### DDP Step (batch 64, 4096×4096 weight, bfloat16)

| Config | PyTorch samp/s | PyTorch ms | JAX samp/s | JAX ms | Δ samp/s |
|--------|---------------|-----------|-----------|--------|---------|
| 1-node, 8 GPU | 194,000 | 2.64 | 482,000 | 1.06 | **+149%** |
| 2-node, 16 GPU | 307,000 | 3.34 | 589,000 | 1.74 | **+92%** |

JAX is faster at raw compute (+17% GEMM) and dominates end-to-end DDP (+149% single-node, +92% two-node), but loses on standalone allreduce bandwidth (-41% at 1 MB). XLA compiles the full training step — forward, backward, allreduce, and weight update — as one fused program, allowing it to overlap communication with computation. PyTorch treats the two as separate operations and cannot fuse across that boundary, which explains why JAX wins on DDP despite losing the isolated collective benchmark.
