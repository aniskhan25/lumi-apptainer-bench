# Phase 2 results — two nodes, all-to-all

**Image:** `lumi-multitorch-latest.sif` → `...full-u24r70f21m50t210-20260513_121430.sif`
**Digest:** `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
**Partition:** `standard-g`, 2 nodes, 8 ranks/node, world size 16
**Date:** 2026-08-04
**Jobs:** 20674950 (EP=8, `nid[006146,006992]`), 20674957 (EP=16, `nid[005136,005384]`)
**Binding:** `--cpu-bind=cores` — **not** the LUMI NUMA masks, see the caveat below

---

## Headline: the reported all-to-all failure does not reproduce

Both gate groups pass on the current default container.

| | EP=8 (intra-node) | EP=16 (cross-node) |
| --- | --- | --- |
| Gate group | `gate3-alltoall-intra-node` **PASS** | `gate4-alltoall-cross-node` **PASS** |
| Payload correctness | pass, 0 mismatches | pass, 0 mismatches |
| Uneven MoE-shaped splits | pass, 3 zero-token peers | pass, 6 zero-token peers |
| Communicator churn | pass | pass |
| `spans_nodes` | false (control) | true |

No `PTLTE_NOT_FOUND`, no "unhandled system error", no hang. A 16-rank all-to-all spanning
two nodes initialises and exchanges correctly, including with uneven splits and zero-token
peers, and survives repeated communicator create/destroy.

**This does not clear the container.** The report's failure is at **EP=32 across four
nodes** (§4.4), which is Phase 3. What Phase 2 establishes is that cross-node all-to-all
per se is not broken on this image, so a Phase 3 failure — if it occurs — would be specific
to the 32-rank group size or the four-node topology rather than to crossing a node boundary
at all. Report §4.1's `PTLTE_NOT_FOUND` arose in a real Megatron MoE workload, which this
pure-PyTorch collective does not reconstruct; a clean result here does not rule it out.

---

## The EP=8 intra-node rule, measured

The report calls constraining expert parallelism to 8 ranks "the largest single throughput
win of the whole effort" and quotes +64% against EP=16 at 16 nodes. Measured at the
collective level:

| Message | EP=8 BW (GB/s) | EP=16 BW (GB/s) | EP=8 advantage | EP=8 lat (µs) | EP=16 lat (µs) |
| --- | --- | --- | --- | --- | --- |
| 16 KiB | 0.201 | 0.090 | **2.2×** | 71.4 | 171.2 |
| 256 KiB | 3.261 | 1.332 | **2.4×** | 70.3 | 184.6 |
| 1 MiB | 11.404 | 4.210 | **2.7×** | 80.5 | 233.5 |
| 4 MiB | 29.543 | 10.544 | **2.8×** | 124.2 | 372.9 |
| 16 MiB | 49.627 | 8.930 | **5.6×** | 295.8 | 1761.4 |

Bandwidth is bytes leaving the rank (`chunk × (group_size − 1)`) divided by the p50 of a
max-across-ranks reduction, so it is a per-rank figure for the slowest participant.

The collective-level gap (2.2×–5.6×) is much larger than the reported +64% end-to-end, and
that is the expected relationship: the all-to-all is one component of a training step, so a
~2.8× collective speedup dilutes to a smaller whole-step gain. **The measurement supports
and explains the report's design rule** — keeping the expert all-to-all inside one node on
XGMI rather than crossing Slingshot is worth a large constant factor, and LUMI users
planning MoE work should treat EP≤8 as the default.

Note EP=16 is a mixed case, not a pure cross-node one: of each rank's 15 peers, 7 are
on-node over XGMI and 8 are off-node over Slingshot. The cross-node component is therefore
worse than the aggregate column suggests.

---

## New finding: EP=16 bandwidth is non-monotonic — it collapses at 16 MiB

The EP=16 curve rises to 10.544 GB/s at 4 MiB and then **falls to 8.930 GB/s at 16 MiB**,
while latency rises 372.9 → 1761.4 µs. That is a 4.7× latency increase for a 4× data
increase, so the largest message size is losing throughput rather than amortising better.

EP=8 over the same range behaves as expected, climbing monotonically 29.5 → 49.6 GB/s. So
this is specific to the cross-node path, not to large messages generally.

Candidate explanations, none tested:

- a protocol threshold crossing (eager → rendezvous) between 4 and 16 MiB;
- memory-registration or bounce-buffer limits in `aws-ofi-nccl` / the CXI provider;
- congestion once 8 ranks per node simultaneously push 8 MiB each off-node.

Worth pursuing because MoE expert-dispatch buffers sit at exactly these sizes, and because
this is the region where the reporter's workload operated. A finer sweep between 4 and
64 MiB, plus `FI_CXI_*` and RCCL protocol-threshold variation, would localise it. This is
also the reason the bandwidth gates reduce with `last` (largest message) rather than `max`:
a `max` reducer would have reported 10.544 GB/s and hidden the collapse entirely.

---

## Caveats on these numbers

- **Not NUMA-optimal.** Ran with `--cpu-bind=cores` because the LUMI GPU/CPU mask list
  could not be satisfied (see below). Absolute bandwidth is therefore pessimistic. The
  EP=8 vs EP=16 *ratio* is much more robust than the absolute values, since both arms ran
  under identical binding.
- **Single run per configuration.** No repetition, so no variance estimate. Roadmap §3.1
  asks for a reproduction rate; these are 1/1. The gates should not be tightened from a
  single sample.
- **Warm nodes.** Both jobs completed in ~30 s wall time with no RCCL cold-start delay.
  `EXCLUDE_NODES` was set to this project's known-bad list throughout.

---

## Blocker resolved along the way: LUMI CPU bind masks abort the step

Three consecutive attempts (jobs 20671589, 20674720/20674808, 20674821/20674823) died
before launching:

```
srun: error: CPU binding outside of job step allocation,
      allocated CPUs are: 0x001E1E1E1E1E1E1E001E1E1E1E1E1E1E
srun: error: Unable to satisfy cpu bind request
```

The canonical mask list addresses 7 cores in each of 8 GPU groups (56 cores). The step was
granted 28 — `0x1E` is 4 cores, across 7 groups. Tested and ruled out:

| Attempt | Result |
| --- | --- |
| `dev-g` instead of `standard-g` | same 28-core mask |
| `--exclusive` | same 28-core mask |
| `--hint=nomultithread` | same 28-core mask |
| `ENABLE_LUMI_CPU_MASKS=0` (`--cpu-bind=cores`) | **launches** |

So a bare `srun` launched from a login node does not receive the full node's cores on this
project, and neither exclusivity nor SMT hints change it; the mask path needs an `sbatch`
allocation that holds the whole node. `templates/lumi_common.sh` now documents this at the
point of use, and `templates/probe.sh` defaults masks off.

This is a harness/platform interaction rather than a container defect, but it is a good
example of the diagnostic problem this project exists to address: the error names neither
the masks nor the allocation, and costs three failed submissions to interpret.

---

## Gate status

| Gate | EP=8 | EP=16 |
| --- | --- | --- |
| `alltoall_passed` | pass | pass |
| `payload_correct` | pass | pass |
| `zero_mismatches` | pass (0) | pass (0) |
| `uneven_splits_correct` | pass | pass |
| `has_zero_token_peers` | pass (3) | — |
| `communicator_churn_clean` | pass | pass |
| `stayed_intra_node` | pass | — |
| `actually_crossed_nodes` | — | pass |
| `bandwidth_large_message_gbps` | pass (49.6 ≥ 15.0) | pass (8.93 ≥ 1.5) |

Both provisional bandwidth floors were cleared with wide margins, so they remain
uncalibrated rather than validated. Recalibration belongs in Phase 5, after repeated runs
under proper NUMA binding.

---

## Next

Phase 3, four nodes, EP=32 — the configuration report §4.4 says never initialised. That is
now the sharpest open question, because Phase 2 has shown that crossing a node boundary is
not itself the problem on this image.
