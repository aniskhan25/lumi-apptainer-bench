**Repo:** `lumi-ai-factory/laifs-container-tests`
**Title:** Add a multi-node all-to-all test with 8 ranks per node

---

## Summary

The suite has no all-to-all test. Every collective test is either an allreduce (the DDP and
DeepSpeed tests) or a point-to-point transfer (the OSU tests), and the inter-node OSU test runs
one process per node. As a result the suite cannot reach failures driven by all-to-all traffic
patterns or by per-node endpoint counts — which is the class a user report describes hitting in
production while the release tests passed.

## Current coverage

From `lumi-multitorch-u24r70f21m50t210-20260513_121430-tests.md` (suite commit `13a3c42`),
18 tests:

```
accelerate-big-model-inference      pytorch-ddp-singlenode-srun
bitsandbytes-inference-int8         pytorch-ddp-singlenode-torchrun
bitsandbytes-inference              pytorch-ds-multinode-srun
osu-inter-node-gcd2gcd-bw   (x32)   pytorch-ds-multinode-torchrun
osu-intra-node-gcd2gcd-bw   (x32)   pytorch-ds-singlenode-srun
peft-alora-finetuning               pytorch-ds-singlenode-torchrun
pytorch-ddp-multinode-srun          pytorch-singlegpu
pytorch-ddp-multinode-torchrun      transformers-inference      (x2)
                                    vllm-bench-full-node-gpt-oss-120b
                                    vllm-bench-single-gpu-llama31-8b
```

Multi-node *is* covered, via `pytorch-ddp-multinode-*` and `osu-inter-node-gcd2gcd-bw`. The two
specific gaps are:

1. **No all-to-all.** Allreduce is typically ring- or tree-based; all-to-all establishes
   connections to every peer. They exercise different amounts of fabric state.
2. **`osu-inter-node-gcd2gcd-bw` uses one process per node**, so it opens a handful of endpoints.
   An 8-rank-per-node all-to-all opens many more. Failures driven by finite per-NIC resources are
   unreachable while the peak endpoint count stays that low.

This matters concretely: expert parallelism in a Mixture-of-Experts model *is* an all-to-all, and
that is the workload pattern in the user report.

## Suggested test

A `torch.distributed.all_to_all_single` test, at least 2 nodes × 8 ranks, asserting:

- **Correctness, not just completion.** Fill the chunk destined for peer `j` with a value
  encoding `(my_rank, j)` and verify the chunk received from peer `j` encodes `(j, my_rank)`. A
  summed checksum cannot distinguish a correct exchange from a permuted one.
- **Uneven splits with zero-token peers**, via `input_split_sizes`/`output_split_sizes`. MoE
  routing produces uneven token counts per expert and some experts receive nothing; an
  implementation can pass every equal-split test and still fail here.
- **A group-size sweep.** On LUMI-G, 8 ranks stays intra-node on XGMI while 16 or more crosses
  Slingshot, and those are very different paths. Group size maps directly onto expert-parallel
  degree.
- **Repeated communicator create/destroy**, so resource leaks appear.

Two implementation notes that cost real debugging time:

- `dist.new_group()` does not create the RCCL communicator — that happens lazily on first use. A
  test that creates groups without exercising them counts Python objects and consumes no fabric
  resources.
- Bind the device before the first collective — `torch.cuda.set_device(local_rank)`, and optionally
  `device_id=` on `init_process_group` to make initialisation eager. With one visible GCD per rank
  every rank sees index 0, so PyTorch's fallback guess of `rank N -> device N` is wrong, and we
  measured a hang once a rank held more than one communicator. (We added `set_device` and `device_id`
  in the same change, so which one is load-bearing was not isolated; `set_device` is the more likely
  candidate.)

A working implementation is at
[`bench/tests/alltoall.py`](https://github.com/aniskhan25/lumi-apptainer-bench/blob/feature/laif-container-validation/bench/tests/alltoall.py)
with gate thresholds in `manifests/gates/alltoall_*.json` — reuse or adapt freely.

## Reference numbers

Measured on `full-u24r70f21m50t210-20260513_121430`, `standard-g`, 8 ranks/node,
`--cpu-bind=cores` (not the LUMI NUMA masks, so these are conservative). Per-rank bandwidth,
bytes leaving the rank over the p50 of a max-across-ranks reduction:

| Message | EP=8 (1 node, XGMI) | EP=16 (2 nodes) | EP=32 (4 nodes) |
| --- | --- | --- | --- |
| 16 KiB | 0.201 GB/s | 0.090 | 0.070 |
| 1 MiB | 11.404 | 4.210 | 4.051 |
| 4 MiB | 29.543 | 10.544 | 7.689 |
| 16 MiB | 49.627 | 8.930 | 6.675 |

Two things worth knowing when setting thresholds:

- Run-to-run variance at 16 MiB is roughly ±10%, and ±36% at 16 KiB, so a threshold set from a
  single sample will be flaky. Use repetitions.
- Reduce bandwidth on the largest message size rather than averaging across the sweep —
  latency-bound small messages dominate the mean and make it useless as a bandwidth gate.

Also useful as a guard: assert the group genuinely spanned nodes. A misconfigured job can satisfy
every correctness check while never leaving the node, which is exactly how an inter-node problem
survives validation.

## Related

- `laifs-container-recipes` #20, #28, #30 — all fabric-adjacent, all found by users rather than
  by the suite.
