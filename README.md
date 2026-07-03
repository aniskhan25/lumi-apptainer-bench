# JAX Benchmarks on Roihu

This branch ports the JAX benchmarks from `feature/jax-benchmarks` (LUMI) to Roihu (`roihu-gpu.csc.fi`).
PyTorch comparison numbers are taken from the LUMI run (main branch) — nothing here re-collects them.

## Prerequisites

JAX is preinstalled on Roihu as a module — no custom container needed for JAX:
```bash
module load python-jax
```

The PyTorch container (`$SIF`) is still used to provide the runtime environment. Bind mounts are handled by the CSC helper:
```bash
apptainer exec --bind="$(csc-common-bind)" $SIF python ...
```

## Run JAX Benchmarks on Roihu

Set the GPU count per node before running (check with `sinfo`):
```bash
export PROJECT_NAME=project_462000131
export GPUS_PER_NODE=4          # adjust to match actual node topology
export SIF=/path/to/container.sif
```

Single-node GEMM + kernel mix:
```bash
./templates/roihu_single.sh $SIF -- bench/run jax-single --out results/roihu_jax_single.json
```

Two-node allreduce sweep:
```bash
export NODES=2
./templates/roihu_multi.sh $SIF -- bench/run jax-multi --out results/roihu_jax_multi_2n.json
```

Single-node DDP (allreduce verified):
```bash
./templates/roihu_single.sh $SIF -- bench/run jax-ddp --out results/roihu_jax_ddp.json
```

Two-node DDP (allreduce verified):
```bash
export NODES=2
./templates/roihu_multi.sh $SIF -- bench/run jax-ddp-multi --out results/roihu_jax_ddp_multi.json
```

For the full suite in a single Slurm job, use a batch script on scratch (see `jax_bench_roihu.sh` in the scratch results directory) modelled on the LUMI equivalent.

## Results

To be filled in after runs on Roihu.

LUMI MI250X results (from `feature/jax-benchmarks`) are shown below for reference. Roihu uses NVIDIA GPUs — results will differ.

### GEMM (bfloat16, 4096×4096, single GPU) — LUMI reference

| | PyTorch | JAX | Δ |
|--|---------|-----|---|
| TFLOPS | 103.3 | 120.6 | **+17%** |
| p50 ms | 1.33 | 1.14 | **-14%** |

### Allreduce (2-node, 16 GPU, cross-node) — LUMI reference

| Size | PyTorch GB/s | JAX GB/s | Δ |
|------|-------------|---------|---|
| 1 KB | 0.013 | 0.004 | -69% |
| 64 KB | 0.265 | 0.266 | 0% |
| 256 KB | 2.273 | 0.952 | -58% |
| 1 MB | 5.097 | 3.006 | **-41%** |

### DDP Step (batch 64, 4096×4096 weight, bfloat16) — LUMI reference

| Config | PyTorch samp/s | PyTorch ms | JAX samp/s | JAX ms | Δ samp/s |
|--------|---------------|-----------|-----------|--------|---------|
| 1-node, 8 GPU | 194,000 | 2.64 | 482,000 | 1.06 | **+149%** |
| 2-node, 16 GPU | 307,000 | 3.34 | 589,000 | 1.74 | **+92%** |
