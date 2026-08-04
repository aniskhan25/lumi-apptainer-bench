# Phase 3 results — four nodes, EP=32

**Image:** `lumi-multitorch-latest.sif` → `...full-u24r70f21m50t210-20260513_121430.sif`
**Digest:** `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
**Partition:** `standard-g`, 4 nodes, 8 ranks/node, world size 32
**Date:** 2026-08-04
**Jobs:** 20676671, 20676681, 20676702 (EP=32 ×3), 20676712 (EP=16 on 4 nodes)
**Binding:** `--cpu-bind=cores`, `EXCLUDE_NODES` set

---

## EP=32 does not reproduce the reported failure — 3/3 clean

| Run | World | Group | Groups | Spans nodes | Correctness | Uneven | Churn | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | 32 | 32 | 1 | true | pass | pass | pass | **PASS** |
| r2 | 32 | 32 | 1 | true | pass | pass | pass | **PASS** |
| r3 | 32 | 32 | 1 | true | pass | pass | pass | **PASS** |
| EP=16, 4n | 32 | 16 | 2 | true | pass | pass | pass | **PASS** |

Report §4.4 states that a 32-rank expert all-to-all produced an NCCL "unhandled system
error" during initialisation and that they "have no throughput number for EP=32 at all
because it never initialised". A 32-rank all-to-all group spanning four nodes initialises
and exchanges correctly here, three times out of three, with uneven splits and zero-token
peers, and survives repeated communicator create/destroy.

### The configuration tested is not the configuration that failed

This is the important qualifier and it limits the conclusion sharply.

The report's failure was at **16 nodes** with expert parallelism spanning 32 ranks. At
16 nodes and 8 ranks per node that is 128 ranks carrying **four concurrent 32-rank
all-to-all meshes**. What ran here is 4 nodes, 32 ranks, **one** mesh.

| | Report §4.4 | Phase 3 |
| --- | --- | --- |
| Job size | 16 nodes / 128 ranks | 4 nodes / 32 ranks |
| Group size | 32 | 32 |
| Nodes spanned per group | 4 | 4 |
| Concurrent groups | **4** | **1** |

Group size and nodes-spanned match; total rank count and concurrent-mesh count do not.
Since the reported symptom was a failure *during initialisation*, and communicator setup
cost scales with the total number of ranks and communicators in the job rather than with
one group's size, the untested variable is plausibly the one that matters.

**So Phase 3 narrows the question rather than closing it.** What is now established: group
size 32 and spanning four nodes are not by themselves sufficient to break the collective on
this image. What is not established: whether 128 ranks with four concurrent 32-rank meshes
does. That configuration is 16 nodes, which is inside this branch's ceiling, and is the
obvious next run.

The standing caveat also still applies: a pure-PyTorch `all_to_all_single` does not
reconstruct Megatron's expert-dispatch path, so none of these clean results rule out
report §4.1.

---

## Corrected: cross-node bandwidth saturates, it does not collapse

Phase 2 flagged a "collapse" in EP=16 bandwidth from 10.544 GB/s at 4 MiB to 8.930 at
16 MiB and attributed it to the cross-node path. **That finding is retracted.** Repetition
shows it was a single-sample artefact.

Bandwidth per rank, GB/s:

| Message | EP=32 r1 | EP=32 r2 | EP=32 r3 | EP=32 spread | EP=16 (4n) | EP=16 (2n, Phase 2) |
| --- | --- | --- | --- | --- | --- | --- |
| 16 KiB | 0.063 | 0.070 | 0.072 | ±7% | 0.092 | 0.090 |
| 256 KiB | 0.938 | 1.044 | 1.054 | ±6% | 1.436 | 1.332 |
| 1 MiB | 3.741 | 3.778 | 4.206 | ±6% | 4.845 | 4.210 |
| 4 MiB | 7.704 | 7.350 | 7.579 | ±2% | 8.218 | 10.544 |
| 16 MiB | 7.656 | 6.631 | 8.064 | **±10%** | 10.052 | 8.930 |

Two things follow:

1. **Run-to-run variance at 16 MiB is ±10% for EP=32**, and the EP=16 four-node run *rose*
   over 4→16 MiB (8.218 → 10.052) where the two-node run fell (10.544 → 8.930). The
   direction of the 4→16 MiB change is not consistent, so it is noise, not an effect.
2. The corrected description is **saturation**: cross-node per-rank bandwidth plateaus in
   the 6.6–10.1 GB/s band beyond about 4 MiB regardless of group size or node count. That
   is a ceiling, not a regression.

This is why the Phase 2 numbers carried an explicit "single run, no variance estimate"
caveat, and it is a concrete argument for requiring repetitions before any threshold is
promoted from provisional.

---

## Refined topology finding: the cliff is the node boundary, not the group size

Per-rank bandwidth at 16 MiB across everything measured so far:

| Configuration | Nodes spanned | BW (GB/s) | vs EP=8 |
| --- | --- | --- | --- |
| EP=8 (intra-node, XGMI) | 1 | 49.63 | 1.0× |
| EP=16 (2 nodes) | 2 | 8.93 | 5.6× slower |
| EP=16 (4 nodes, 2 groups) | 2 | 10.05 | 4.9× slower |
| EP=32 (4 nodes, mean of 3) | 4 | 7.45 | **6.7× slower** |

Going from 1 node to 2 costs roughly 5×. Going from 2 nodes to 4 — doubling the group again
— costs only a further ~1.2×. **The penalty is almost entirely paid at the first node
boundary and then largely flattens.**

That sharpens the report's design rule in a useful way. The rule is not "smaller expert
groups are proportionally faster"; it is "keep the expert all-to-all *on the node*". Once a
group has left the node, growing it from 16 to 32 ranks is comparatively cheap. For a user
who cannot fit EP=8, this says there is little throughput reason to prefer EP=16 over EP=32
— the decision has already been made by crossing the boundary. The report could not measure
this because EP=32 never initialised for them.

---

## Gate status

All four runs pass `gate4-alltoall-cross-node`, including `actually_crossed_nodes`, so the
guard confirms the meshes genuinely spanned nodes rather than silently collapsing to a
node-local exchange.

Provisional bandwidth floors were cleared with wide margins again (worst case 6.631 GB/s
against a 1.5 GB/s floor). With variance now measured at ±10% at the top size, a
recalibrated cross-node floor of roughly 5 GB/s at 16 MiB would be defensible — still
generous, but it would actually bite. Deferred to Phase 5.

---

## Next

The one configuration that would genuinely test report §4.4 is **16 nodes / 128 ranks with
EP=32**, i.e. four concurrent 32-rank meshes. It is within the node ceiling for this branch
and is the highest-value remaining collective run.

Phase 4 also still holds the JIT cache work (`LAIF_CACHE_MODE=lustre` at 64+ ranks, report
§4.7), which needs 8+ nodes and is independent of the above.
