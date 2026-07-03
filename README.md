# JAX Benchmarks on Roihu

This branch ports the JAX benchmarks from `feature/jax-benchmarks` (LUMI MI250X) to Roihu (NVIDIA GH200).
PyTorch comparison numbers are taken from the LUMI run (main branch) — nothing here re-collects them.

## Prerequisites

JAX 0.10.2 is preinstalled on Roihu via a TYKKY module. After loading it, `python3` transparently runs inside the JAX container — no `apptainer exec` needed:

```bash
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/appl/modulefiles/manual/aida/aarch64:/appl/modulefiles/manual/general/aarch64
module load python-jax
python3 -c "import jax; print(jax.devices())"
```

## Run JAX Benchmarks on Roihu

Set the project before running:
```bash
export PROJECT_NAME=project_2014553
```

Single-node GEMM + kernel mix (uses `gputest` partition by default):
```bash
./templates/roihu_single.sh -- bench/run jax-single --out results/roihu_jax_single.json
```

Two-node allreduce sweep:
```bash
export NODES=2
./templates/roihu_multi.sh -- bench/run jax-multi --out results/roihu_jax_multi_2n.json
```

Single-node DDP (allreduce verified):
```bash
./templates/roihu_single.sh -- bench/run jax-ddp --out results/roihu_jax_ddp.json
```

Two-node DDP (allreduce verified):
```bash
export NODES=2
./templates/roihu_multi.sh -- bench/run jax-ddp-multi --out results/roihu_jax_ddp_multi.json
```

For the full suite in a single Slurm job, use the batch script at
`/scratch/project_2014553/anisrahm/bench_results/jax_bench_roihu.sh`.

## Results

To be filled in after runs on Roihu (NVIDIA GH200, 4 GPUs per node).

LUMI MI250X results are shown below for reference.

### GEMM (bfloat16, 4096×4096, single GPU) — LUMI MI250X reference

| | PyTorch | JAX | Δ |
|--|---------|-----|---|
| TFLOPS | 103.3 | 120.6 | **+17%** |
| p50 ms | 1.33 | 1.14 | **-14%** |

### Allreduce (2-node, 16 GPU, cross-node) — LUMI MI250X reference

| Size | PyTorch GB/s | JAX GB/s | Δ |
|------|-------------|---------|---|
| 1 KB | 0.013 | 0.004 | -69% |
| 64 KB | 0.265 | 0.266 | 0% |
| 256 KB | 2.273 | 0.952 | -58% |
| 1 MB | 5.097 | 3.006 | **-41%** |

### DDP Step (batch 64, 4096×4096 weight, bfloat16) — LUMI MI250X reference

| Config | PyTorch samp/s | PyTorch ms | JAX samp/s | JAX ms | Δ samp/s |
|--------|---------------|-----------|-----------|--------|---------|
| 1-node, 8 GPU | 194,000 | 2.64 | 482,000 | 1.06 | **+149%** |
| 2-node, 16 GPU | 307,000 | 3.34 | 589,000 | 1.74 | **+92%** |
