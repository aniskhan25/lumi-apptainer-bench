# LUMI Apptainer Benchmark

A small benchmark harness for one task: compare a new LUMI container against a known stable container under the same launch setup.

The repo stays narrow on purpose:
- single-node compute
- single-node DDP step timing
- two-node allreduce
- two-node DDP step timing
- runtime and filesystem sanity checks

The outputs are structured JSON files plus `delta.json` comparisons.

## Repo Layout
- `bench/bench.py`: benchmark CLI
- `bench/tests/`: individual benchmark tests
- `bench/compare.sh`: run old and new containers and write `delta.json`
- `templates/`: LUMI Slurm launch templates
- `scripts/run_benchmarks.sh`: run the standard benchmark set
- `scripts/summarize_results.py`: print Markdown tables from a results directory

## Clone
```bash
git clone https://github.com/aniskhan25/lumi-apptainer-bench.git
cd lumi-apptainer-bench
```

## Run The Full Benchmark On LUMI
Set the required environment:
```bash
export PROJECT_NAME=project_462000131
```

Optional overrides (defaults are set in the script):
```bash
export OLD_CONTAINER=/path/to/old.sif
export NEW_CONTAINER=/path/to/new.sif
export PARTITION=standard-g
export ACCOUNT="$PROJECT_NAME"
export RESULTS_ROOT=/scratch/$PROJECT_NAME/$USER/bench_results
```

Run the standard benchmark set:
```bash
./scripts/run_benchmarks.sh
```

This writes results under:
```bash
${RESULTS_ROOT:-/scratch/$PROJECT_NAME/$USER/bench_results}
```

## Run Individual Benchmarks
Single-node compute:
```bash
./templates/single_8g_8r.sh /path/to/container.sif -- bench/run single --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_single.json
```

Single-node DDP:
```bash
./templates/single_8g_8r.sh /path/to/container.sif -- bench/run ddp --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_ddp.json
```

Two-node allreduce:
```bash
export NODES=2
./templates/allreduce_sweep.sh /path/to/container.sif -- bench/run multi --allreduce --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_allreduce.json
```

Two-node DDP:
```bash
export NODES=2
./templates/multi_ng_8rpn.sh /path/to/container.sif -- bench/run ddp --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_ddp_2n.json
```

Sanity check:
```bash
./templates/filesystem.sh /path/to/container.sif -- bench/run check --out /scratch/$PROJECT_NAME/$USER/bench_results/lumi_check.json
```

## Compare Two Containers
Use the same template and benchmark mode for both containers.

Example:
```bash
export BENCH_TEMPLATE=./templates/multi_ng_8rpn.sh
./bench/compare.sh \
  --old "$OLD_CONTAINER" \
  --new "$NEW_CONTAINER" \
  --mode ddp \
  --results-dir /scratch/$PROJECT_NAME/$USER/bench_results/lumi_ddp_2n_compare
```

The main verdict is in `delta.json`.

## Summarize Results
Generate Markdown tables from an existing results directory:
```bash
python3 ./scripts/summarize_results.py /scratch/$PROJECT_NAME/$USER/bench_results
```

## Expected Outputs
Standard run files:
- `lumi_single.json`
- `lumi_ddp.json`
- `lumi_single_16r.json`
- `lumi_allreduce.json`
- `lumi_multi.json`
- `lumi_ddp_2n.json`
- `lumi_check.json`

Comparison directories:
- `lumi_check_compare/`
- `lumi_ddp_compare/`
- `lumi_multi_compare/`
- `lumi_ddp_2n_compare/`
- `lumi_single_16r_compare/`

## How To Read The Results
Read `delta.json` first.

Important fields:
- `metrics`: old value, new value, and percent delta for each metric
- `regressions`: metrics that crossed the configured threshold
- `regression_count`: number of flagged regressions

Typical interpretation:
- stable single-node compute with worse multi-node metrics suggests a communication or runtime issue
- stable allreduce with worse DDP step time suggests overhead outside the collective itself
- zero `regression_count` means no threshold-defined regressions were detected

## Metric Glossary
- `DDP`: PyTorch `DistributedDataParallel`, where each rank trains the same model and gradients are synchronized across ranks
- `GEMM TFLOPS`: raw GPU matrix multiplication throughput
- `GEMM p50 ms`: median time for one matrix multiply
- `KernelMix p50 ms`: median time for a small transformer-like mix of GPU operations
- `Allreduce BW (GB/s)`: how fast data is reduced and exchanged across GPUs or nodes
- `Allreduce Lat (us)`: how long one allreduce operation takes
- `DDP samples/sec`: distributed training throughput for the DDP step benchmark
- `DDP step avg ms`: average end-to-end time for one DDP training step
- `DDP step p95 ms`: tail latency for the DDP training step
- `Check`: sanity test for GPU visibility and writable cache paths

## Comparison Methodology
This repo is for fair container assessment, not maximum one-off tuning.

Keep these fixed between old and new runs:
- partition
- node count
- template
- benchmark mode
- benchmark parameters
- launcher environment

Use the same templates for both images and compare the generated `delta.json` files rather than isolated raw numbers.
