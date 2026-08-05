# Communicator-count stress results

**Image:** `lumi-multitorch-latest.sif` → `...full-u24r70f21m50t210-20260513_121430.sif`
**Partition:** `standard-g`, 8 ranks/node, up to 8 live world-spanning communicators per rank
**Date:** 2026-08-04 / 2026-08-05
**Jobs:** 20711452, 20724354, 20724372, 20724753, 20724970

Motivation: every group-*shape* test in this suite is clean at sizes 8/16/32 across 1–16
nodes, so the untested variable for report §4.1 was communicator *population* — a Megatron
MoE job holds data-, pipeline-, expert- and expert-data-parallel communicators at once, while
those tests created one or four.

---

## Result 1 — no communicator-count limit found

| Nodes | Ranks | Communicators/rank | Per node | Wall | Result |
| --- | --- | --- | --- | --- | --- |
| 2 | 16 | **8/8** | 64 | 37 s | PASS |
| 4 | 32 | — | — | timeout | **hung** (see Result 3) |
| 8 | 64 | **8/8** | 64 | 45 s | PASS |
| 16 | 128 | **8/8** | 64 | 47 s | PASS |

Where it runs, it runs cleanly and fast: 8 live world-spanning communicators per rank — 64
per node — at every scale up to 128 ranks, in under a minute. Every rank reached the limit
(`communicators_min == communicators_max == 8`).

A Megatron MoE job needs well under 8 communicators per rank. **The communicator-population
hypothesis for report §4.1 is not supported.** Combined with the clean group-shape results,
no pure-PyTorch collective configuration tried so far reproduces `PTLTE_NOT_FOUND`.

---

## Result 2 — hidden memory per communicator, quantified

Report §4.2 states that "the HIP context and the RCCL/CXI communicator buffers are registered
outside PyTorch's caching allocator, so they are invisible to it", but could not put a number
on it. Measured via `torch.cuda.mem_get_info()` after each communicator:

| World size | Device MiB per communicator |
| --- | --- |
| 16 | 651.5 |
| 64 | 626.8 |
| 128 | 631.8 |

**≈630–650 MiB per RCCL communicator, essentially independent of world size.** Throughout
all of it `torch.cuda.memory_allocated()` reported **0.0 MiB** — PyTorch cannot see any of it.

The marginal curve at 16 ranks, which separates one-time cost from per-communicator cost:

| Communicators | Device used (MiB) | Marginal |
| --- | --- | --- |
| baseline (HIP context only) | 90 | — |
| 1 | 1040 | +950 |
| 2 | 1692 | +652 |
| 3 | 2346 | +654 |
| … | … | +652–656 |
| 16 | 10838 | +656 |

So: ~90 MiB HIP context, ~950 MiB for the first communicator (RCCL one-time init included),
then a flat ~653 MiB each.

**This plausibly accounts for report §4.2's memory wall.** They could not exceed ~57 GiB of
PyTorch-allocated memory against a 63.98 GiB nameplate — a gap of roughly 7 GiB. A Megatron
MoE job holding on the order of 8–10 communicators lands at
`950 + 9 × 653 ≈ 6.8 GiB` of invisible RCCL/CXI memory, which is the same magnitude as the
gap they measured across three independent configurations. Their ~40 GB planning figure is
consistent with this rather than pessimistic.

This is the most directly reusable number to come out of the exercise: model-sizing guidance
should subtract roughly `1 + 0.65 × (communicators − 1)` GiB per GCD before counting
parameters and activations.

---

## Result 3 — the hangs are node-specific, not count-specific

**Retracting an intermediate claim.** After the first 128-rank run stalled with every rank
having completed exactly 3 communicators, I described that as a ceiling at 3. The sweep
disproves it: a later 128-rank run created all 8, and the arm that hung was **4 nodes** while
8 and 16 nodes passed. A non-monotonic ceiling is not a ceiling.

The node assignments explain it:

| Job | Nodes | Node list | Result |
| --- | --- | --- | --- |
| 20724354 | 2 | `nid[005556-005557]` | PASS |
| 20724372 | 4 | `nid[007038-007041]` | **TIMEOUT** |
| 20724753 | 8 | `nid[006186-006193]` | PASS |
| 20724970 | 16 | `nid[005724-005729,006186-006195]` | PASS |
| 20711452 | 16 | `nid[007769-007784]` | **TIMEOUT** |

Both hangs were on `nid007xxx`; all three passes were on `nid005xxx`/`nid006xxx`. This
project's existing known-bad list already contains five `nid007xxx` entries
(`nid007812`, `nid007679`, `nid007680`, `nid007405`, `nid007406`) — none of which overlap the
two ranges seen here, so the list is incomplete rather than wrong.

This is a correlation across five runs, not a proven cause, but it is actionable: hangs
attributed to scale or to configuration may simply be node placement. Any startup or
communicator measurement on LUMI is uninterpretable without recording the node list, which is
why `expanded_hostnames()` was added to the probe.

It also bears on the report: §4.4's EP=32 initialisation failure and §4.5's very long
bootstraps are both the shape this flakiness produces. Neither can be attributed to the
container without the node lists from those runs.

---

## Result 4 — a real gotcha: omitting `device_id` hangs at scale

Found while chasing Result 3, and worth passing on independently.

`torch.distributed.init_process_group()` was being called without `device_id`. PyTorch warns
about exactly this:

```
ProcessGroupNCCL.cpp:5138: Guessing device ID based on global rank.
This can cause a hang if rank to GPU mapping is heterogeneous.
```

On LUMI with `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`, every rank sees a single device at
index 0, so guessing rank 5 → device 5 is wrong. Effects measured at 128 ranks:

| | Without `device_id` | With `device_id` |
| --- | --- | --- |
| First communicator init | 13.87 s | 1.14 s |
| Second communicator | **hung indefinitely** | 0.29 s |
| "Guessing device ID" warnings | present | none |

A single communicator per rank survives the wrong guess, which is why every earlier test in
this suite passed — `alltoall.py` gives each rank exactly one group. The failure needs two or
more communicators per rank, which is precisely the Megatron-like case.

Fixed in `bench/tests/distributed.py` for all tests. This is a launcher/user-code issue rather
than a container defect, but it is a strong candidate for user-reported hangs on LUMI and
costs nothing to document.

---

## Test defect fixed: no evidence from a timeout

The first 128-rank run wrote its per-rank record only after the whole loop, so when it was
killed mid-loop it produced **zero** rank files — a 16-node allocation that yielded nothing
beyond "did not finish". Records are now flushed after every step via a temp-file rename, so a
timeout still reveals which step stalled and how long the preceding ones took. Both Results 3
and 4 depend on that data.

The same rule was already applied in `jit_cache.py` and was wrongly not carried over here:
**write evidence before the operation that might not return.**

---

## Gate status

`gate6-concurrent-communicator-count` at 8 communicators/rank: PASS at 2, 8 and 16 nodes;
no aggregate at 4 nodes, which the gates score as failure-by-missing-value rather than as a
pass — correct behaviour for a hang.

The provisional `communicators_per_rank` floor of 16 is above what was tested here (8) and
should be lowered to 8 or the sweep extended, since a floor no run has ever cleared cannot
distinguish a healthy container from a broken one.

---

## Where report §4.1 stands

Exhausted without reproduction: group size (8/16/32), node span (1/2/4/16), concurrent
disjoint meshes (4), communicator population per rank (8), and uneven/zero-token dispatch.

Untried and now the only substantial leads:
- **Megatron's real expert-dispatch path**, which uses its own buffer management and
  `all_to_all` variants rather than plain `all_to_all_single`.
- **The `nid007xxx` correlation** — if the original failures were node-placement artefacts,
  there may be nothing container-side to find, which is itself a reportable conclusion.
