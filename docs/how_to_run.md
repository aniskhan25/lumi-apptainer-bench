# How to run on LUMI

## Prereqs

- You are on LUMI and have a valid project (e.g., `project_465000001`).
- You have a container image (`.sif`) built for LUMI.

## Set required environment

```bash
export PROJECT_NAME=project_465000001
export PARTITION=standard-g
export ACCOUNT=${PROJECT_NAME}
```

## Single-node (8 GPUs, 8 ranks)

```bash
./templates/single_8g_8r.sh /path/to/container.sif -- bench/run single --out /scratch/${PROJECT_NAME}/bench_results/manual_single.json
```

## Multi-node (2 nodes, 8 ranks/node)

```bash
export NODES=2
./templates/multi_ng_8rpn.sh /path/to/container.sif -- bench/run multi --out /scratch/${PROJECT_NAME}/bench_results/manual_multi.json
```

## All-reduce sweep

```bash
export NODES=2
./templates/allreduce_sweep.sh /path/to/container.sif -- bench/run multi --allreduce --out /scratch/${PROJECT_NAME}/bench_results/manual_allreduce.json
```

## Filesystem checks

```bash
./templates/filesystem.sh /path/to/container.sif -- bench/run check --out /scratch/${PROJECT_NAME}/bench_results/manual_check.json
```

## A/B compare (old vs new)

```bash
export BENCH_TEMPLATE=./templates/single_8g_8r.sh
./bench/compare.sh --old /path/to/old.sif --new /path/to/new.sif --mode single --results-dir /scratch/${PROJECT_NAME}/bench_results/compare_run
```

## Quick sanity check (what to expect)

- `results.json` contains `schema_version`, `run_id`, and `tests` entries.
- `tests.check.status` is `pass` when ROCm tools are available and cache is writable.
- `tests.single.gemm.tflops` is non-null when PyTorch/ROCm are available.
- `tests.multi.allreduce.bandwidth_gbps` is populated for multi-node runs.

### Example results snippet

```json
{
  "schema_version": "1.0",
  "run_id": "20260115T143519Z",
  "container": {
    "image_path": "/path/to/container.sif"
  },
  "slurm": {
    "job_id": "123456",
    "nodes": 1,
    "ntasks": 8,
    "ntasks_per_node": 8,
    "gpus_per_node": 8,
    "cpus_per_task": 7,
    "distribution": "block",
    "cpu_bind": "map_cpu:49,57,17,25,1,9,33,41",
    "mpi_mode": "host"
  },
  "system": {
    "hostname_list": ["nid001234"],
    "partition": "standard-g",
    "rocm_version": "6.1.0",
    "gpu_count": 8
  },
  "paths": {
    "cache_root": "/scratch/project_465000001/bench_cache",
    "results_dir": "/scratch/project_465000001/bench_results/20260115T143519Z"
  },
  "tests": {
    "check": {
      "status": "pass"
    },
    "single": {
      "gemm": {
        "dtype": "bfloat16",
        "tflops": 125.4,
        "latency_p50_ms": 3.2,
        "latency_p95_ms": 3.6
      },
      "kernel_mix": {
        "latency_p50_ms": 1.8,
        "latency_p95_ms": 2.1
      }
    },
    "multi": {
      "allreduce": {
        "message_sizes_bytes": [1024, 1048576],
        "bandwidth_gbps": [12.3, 186.7],
        "latency_us": [45.1, 810.4],
        "checksum": "1024.0000"
      }
    }
  }
}
```

Notes:
- Templates set `BENCH_*` metadata automatically; use them for consistent JSON output.
- For container MPI, set `MPI_MODE=container` (uses `srun --mpi=pmi2`).
- For GPU mapping overrides, set `CPU_BIND` and `USE_ROCR_VISIBLE_DEVICES` explicitly.
- For kernel-mix stability control, use `--no-softmax-fp32` or set `BENCH_KERNEL_MIX_SOFTMAX_FP32=0`.
- For distributed runs, `MASTER_ADDR` and `MASTER_PORT` are derived from Slurm if not set.
