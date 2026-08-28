Follow-up comment for [#44](https://github.com/lumi-ai-factory/laifs-container-recipes/issues/44),
supplying the RCCL-internal view offered in the issue. Not posted.

---

Ran the reproducer with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`, one log file per rank. It still
hangs with debug on, 2 of 4 attempts, so the extra logging does not mask it. Job 21562035, 4 nodes /
32 ranks, `nid[005480,006148,006396,007492]`.

**The stall is inside `ncclCommInitRankConfig_impl`, and every rank stops at the same point.**

Counting completions per rank in a run that hung while bringing up its 7th communicator:

| | passing run | hung run |
| --- | --- | --- |
| `Connected all rings` | all 32 ranks | all 32 ranks |
| `Connected all trees` | all 32 ranks | all 32 ranks |
| `Init COMPLETE` | 9 | **6** |

All 32 ranks have an identical last line, with no rank diverging:

```
NCCL INFO Connected to proxy localRank N -> connection 0x...
```

For comparison, a healthy communicator logs this immediately after that same point and then finishes:

```
NCCL INFO threadThresholds 8/8/64 | 256/8/64 | 256 | 256
NCCL INFO 16 coll channels, 16 collnet channels, 0 nvls channels, 16 p2p channels, 2 p2p channels per peer
NCCL INFO ncclCommInitRankConfig_impl comm 0x... rank 2 nranks 32 ... - Init COMPLETE
NCCL INFO Init timings - ncclCommInitRankConfig_impl: rank 2 nranks 32 total 1.55
  (kernels 0.00, alloc 0.03, bootstrap 0.01, allgathers 0.12, topo 0.88, graphs 0.00, connections 0.51, rest 0.01)
```

So ring and tree connection both complete, proxy connections are established, and then
`ncclCommInitRankConfig_impl` never returns. A healthy init of the same communicator takes about
1.55 s.

Worth noting this differs from #28, where the report was that all ranks completed `Connected all
rings` but none reached `Connected all trees`. Here trees complete for every rank and the failure is
later.

Caveat on precision: RCCL logs at discrete points, so the last line bounds where execution stopped
rather than pinpointing it. Full per-rank logs available if useful.
