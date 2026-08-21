# Gate suite run against the current release

First end-to-end run of the validation gates against the container they are meant to gate.

**Image:** `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`
(`lumi-multitorch-latest.sif`), digest `d70ec87f…`
**Date:** 2026-08-21 · **Partition:** `standard-g` · **Repo:** `ee10c41`

## Results

| Gate group | Nodes | Result | Failing gate |
| --- | --- | --- | --- |
| `phase1_probe` | 1 | **FAIL** | `fi_info_runs` |
| `phase2_alltoall_ep8` | 2 | PASS | — |
| `phase2_alltoall_ep16` | 2 | PASS | — |
| `phase3_alltoall_ep32` | 4 | PASS | — |

Per-rank all-to-all bandwidth at 16 MiB, against the thresholds frozen from the
`20260513_121430` baselines (both measured with `--cpu-bind=cores`, masks off):

| Group size | Span | Floor | Baseline | **Now** |
| --- | --- | --- | --- | --- |
| EP=8 | 1 node (XGMI) | 15.0 | 49.627 | **50.359** |
| EP=16 | 2 nodes | 1.5 | 8.930 | **14.610** |
| EP=32 | 4 nodes | 1.5 | 6.675 | **6.999** |

Correctness gates passed everywhere: rank-tagged payload verification, zero mismatches,
uneven splits with zero-token peers, communicator churn, and the topology guards
(`stayed_intra_node` for EP=8, `actually_crossed_nodes` for EP=16/32).

EP=16 is 64% above the baseline, well outside the ±10% run-to-run variance we measured at
16 MiB. `aws-ofi-nccl` moved `1.19.1-git-206c02c` → **`1.20.0-git-a2a6d08`** between the two
images, which is the plausible cause, but this is a single sample and not a controlled
comparison. Treated as a reason to recalibrate the floors from repetitions, not as a result.

## The one failure is E1, detected automatically

```
FAIL  fi_info_runs: expected True, got False
```

That is the suite finding the shadowed `fi_info` shim on its own, which is the intended
behaviour. But it has a consequence I had not identified:

```
cxi_provider_visible   skipped   None
```

The provider-enumeration gate **cannot run** when `fi_info` is broken, because the probe
gets its provider list from `fi_info -p cxi`. So the shim does not merely cost a human a
diagnostic — it silently removes a fabric-verification gate from an automated suite, and the
suite reports one failure rather than one failure plus one blind spot. This is the strongest
argument for E1 and it came out of running the suite rather than reading the image.

## Two defects in the suite itself

Both cost real allocation before being found, and both are now fixed.

**1. The suite could not run end to end.** Phases 2 and 3 aborted at launch:

```
srun: error: CPU binding outside of job step allocation, allocated CPUs are:
  0x001E1E1E1E1E1E1E001E1E1E1E1E1E1E
```

`alltoall_sweep.sh` was the only template in the validation path still defaulting
`ENABLE_LUMI_CPU_MASKS=1`. The LUMI NUMA masks need 7 cores per GPU group (`0xfe`); a shared
allocation gives 4 (`0x1E`). It also did not match how the baselines were produced. Fixed in
`a988f47`.

**2. A fixed rendezvous port.** The EP=8 run then failed with every rank on the second node
reporting:

```
The client socket has timed out after 600000ms while trying to connect to (nid005216, 29500)
```

A c10d TCPStore timeout, not a fabric problem — it consumed a 2-node allocation for 10
minutes and produced nothing (job 21428186). `MASTER_PORT` was hardcoded to 29500, which is
unsafe for a suite that runs several jobs back to back. Now derived from `SLURM_JOB_ID`, the
same scheme the LUMI AI Guide uses for this reason. Fixed in `ee10c41`; EP=8 passed on retry.

## Known gap in the gate design

`triton_cache_off_lustre` and `inductor_cache_off_lustre` both pass — but they pass because
`templates/lumi_common.sh` sets `LAIF_CACHE_MODE=tmp`, not because the container does. They
validate our own launcher, so the suite does **not** detect the `$HOME`-default exposure
described in `E3-jit-cache-defaults.md`. Testing that would need a probe arm that deliberately
clears the cache variables.

## Not run

Phases 4–5 (`jit_cache` at 64+ ranks, `comm_count`) need 8–16 nodes and were not run here. The
`communicators_per_rank` floor was lowered 16 → 8 in `52315b3` before any such run, since no
run has ever cleared 8 and the old floor would have failed every healthy job.
