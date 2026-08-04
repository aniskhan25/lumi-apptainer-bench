# EP=32 at 16 nodes — the matching topology for report §4.4

Follow-up to the open question left by `docs/PHASE3_RESULTS.md`: Phase 3 ran one 32-rank
mesh on four nodes, while the report's failure was at 16 nodes, where EP=32 means **four
concurrent 32-rank meshes among 128 ranks**. This run matches that topology.

**Image:** `lumi-multitorch-latest.sif` → `...full-u24r70f21m50t210-20260513_121430.sif`
**Partition:** `standard-g`, 16 nodes, 8 ranks/node, world size 128, `NCCL_DEBUG=WARN`
**Date:** 2026-08-04
**Jobs:** 20677206 (r1), 20677327 (r2)

---

## Result: passes, 2/2

| Run | World | Group | Concurrent groups | Correctness | Uneven (zero-token peers) | Churn | Wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | 128 | 32 | 4 | pass, 0 mismatches | pass (11) | pass | 33 s |
| r2 | 128 | 32 | 4 | pass, 0 mismatches | pass (11) | pass | 35 s |

No NCCL warnings, no `PTLTE_NOT_FOUND`, no "unhandled system error", no
`no local path from gpu N to net`, no hang. Both runs initialised 128 ranks, built four
32-rank communicators, completed the full message sweep and the churn test in about half a
minute of wall time including container startup and Python imports.

**The EP=32 initialisation failure reported in §4.4 does not reproduce on the current
container, in its own topology.**

---

## Why this matters more than the earlier EP=32 result

Phase 3's clean result was weak evidence because it tested one mesh, not four. This run
closes that gap on every axis the report specifies:

| | Report §4.4 | Phase 3 | This run |
| --- | --- | --- | --- |
| Job size | 16 nodes / 128 ranks | 4 nodes / 32 ranks | **16 nodes / 128 ranks** |
| Group size | 32 | 32 | **32** |
| Nodes spanned per group | 4 | 4 | **4** |
| Concurrent groups | 4 | 1 | **4** |

The reported symptom was a failure *during initialisation*, which is where total rank count
and concurrent-communicator count should bite. Both now match and it still passes.

---

## The likely explanation: they were on the April build

Report §4.4 was observed on the **April 2026** container — the build they pinned after the
May regression described in §4.1. This run is on the **May** build. So the most economical
reading is that the two findings are not in tension:

- §4.4 (EP=32 fails to initialise) was an **April-build problem that the May build fixed**.
- §4.1 (multi-node all-to-all breaks) is the **May-build regression** they hit and worked
  around by going back to April.

The `aws-ofi-nccl` bump from `1.18.0-git-c1b89cc` to `1.19.1-git-206c02c` is the one
comms-stack change between the builds and is a plausible source of both, in opposite
directions.

This branch is scoped to the latest container only, so no A/B run was made and this
inference is not tested. It does not need to be tested to be actionable: what matters
operationally is that EP=32 works on the image users get today, which the report explicitly
could not tell them — "we have no throughput number for EP=32 at all because it never
initialised."

---

## Numbers for the EP=32 configuration the report could never measure

Per-rank bandwidth, GB/s:

| Message | 4 nodes (mean of 3) | 16 nodes r1 | 16 nodes r2 |
| --- | --- | --- | --- |
| 16 KiB | 0.069 | 0.070 | 0.045 |
| 256 KiB | 1.012 | 1.052 | 0.840 |
| 1 MiB | 3.908 | 4.051 | 3.301 |
| 4 MiB | 7.544 | 7.689 | 7.562 |
| 16 MiB | 7.450 | 6.675 | 6.634 |

Latency at 16 MiB: 2434.7 and 2449.9 µs (vs ~2100–2450 µs at four nodes).

**Per-rank cost is flat in job size.** At 4 MiB the three configurations agree to within 2%
(7.544 / 7.689 / 7.562), despite the job growing 4× and the number of concurrent meshes
growing 4×. That is the expected behaviour — each rank still exchanges with 31 peers
regardless of how many other meshes exist — and it says the fabric absorbs four concurrent
32-rank all-to-alls without measurable contention at these message sizes.

At 16 MiB the 16-node runs sit ~11% below the four-node mean (6.65 vs 7.45), which is at the
edge of the ±10% variance measured in Phase 3, so it is suggestive of mild contention rather
than established.

Small messages are noticeably noisier at 128 ranks: 16 KiB bandwidth differs by 36% between
the two runs (0.070 vs 0.045) and latency by 55% (227.9 vs 353.8 µs). Small-message
all-to-all is latency- and jitter-bound, and more ranks means more exposure to the slowest
one — the max-across-ranks reduction in the metric makes that visible by design.

---

## Incidental: no sign of a slow bootstrap at this scale

Report §4.5 describes ~45 minutes of Gloo/NCCL bootstrap before iteration 1 at 1024 ranks.
At 128 ranks the entire job — container start, imports, four communicator groups, full
sweep, churn test — took 33–35 s. Two caveats before reading anything into that: 128 ranks
is one eighth of the reported scale and bootstrap cost is not expected to be linear, and
these nodes were warm. It does establish that nothing pathological happens at 128 ranks on
this image.

---

## What is still not tested

The pure-PyTorch collective does not reconstruct Megatron's communicator topology. A
Megatron MoE job builds data-parallel, tensor-parallel, pipeline-parallel, expert-parallel
and expert-data-parallel groups simultaneously, so each rank holds many more concurrent
communicators than the four this test creates. If §4.1's `PTLTE_NOT_FOUND` is driven by
total communicator or endpoint count rather than by any single group's shape, this test
cannot reach it.

That points at a specific, cheap next experiment: a **communicator-count stress test** —
create an increasing number of overlapping process groups per rank and find where
initialisation breaks. It targets the suspected mechanism directly, needs no Megatron, and
would either produce the failure or bound it. Recommended over further group-size sweeps,
which have now been clean at 8, 16 and 32 across 1, 2, 4 and 16 nodes.
