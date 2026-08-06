# Phase 4 results — JIT cache under rank pressure (report §4.7)

**Image:** `lumi-multitorch-latest.sif` → `...full-u24r70f21m50t210-20260513_121430.sif`
**Partition:** `standard-g`, 16 nodes, 8 ranks/node, world size 128
**Date:** 2026-08-04
**Jobs:** 20684269 (lustre), 20684441 (tmp), plus earlier 64-rank runs

---

## REPRODUCED — and the mechanism is more specific than "Lustre writes aren't atomic"

| Arm | Ranks | Compilations/rank | Failed ranks | Corruption | Compile time | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| `LAIF_CACHE_MODE=lustre` | 128 | 18–24 | **[20]** | **rank 20** | 42.0–43.2 s | **FAIL** |
| `LAIF_CACHE_MODE=tmp` | 128 | 24/24 all ranks | none | none | 28.3–28.5 s | **PASS** |

Same container, same workload, same rank count, same node count. The only difference is
where the cache lives. This is the controlled reproduction report §4.7 describes.

### Root cause: the cache reader opens other processes' in-flight temp files

> **Corrected 2026-08-06.** This section first attributed the failure to `pid`/`tid`
> collisions between nodes. That was an over-inference from the filenames. Reading the torch
> source and mapping ranks to nodes gives a simpler and better-supported mechanism, below.
> No measurement changed; the interpretation did.

The failing path is `torch/_inductor/codecache.py:1040`, and the error is not a corrupt read
but a vanished file:

```
[rank81] FileNotFoundError: [Errno 2] No such file or directory:
  '<lustre>/inductor/fxgraph/qz/fqzsuavdiy.../.45092.22875271438464.tmp'
[rank35] FileNotFoundError: [Errno 2] No such file or directory:
  '<lustre>/inductor/fxgraph/qz/fqzsuavdiy.../.45092.22875271438464.tmp'
```

Two pieces of torch source explain it. The writer, `write_atomic` at `codecache.py:454`,
places its temp file in the **same directory as the target**:

```python
tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
```

The reader, `GuardedCache.iterate_over_candidates` around `codecache.py:1031–1044`, lists that
directory and opens **every** entry, with no filtering of temp files:

```python
subdir = cls._get_tmp_dir_for_key(key)
if os.path.exists(subdir):
    for path in sorted(os.listdir(subdir)):
        try:
            with open(os.path.join(subdir, path), "rb") as f:   # <- opens .tmp files too
                ...
        except Exception:
            log.warning("fx graph cache unable to load compiled graph", exc_info=True)
```

So a reader lists the directory, sees another rank's in-flight `.{pid}.{tid}.tmp`, and by the
time it calls `open()` the writer has renamed it away. `FileNotFoundError`, logged at :1040.

**The evidence supports this rather than a collision.** Mapping the affected ranks to nodes
(8 ranks/node):

| Temp filename | Ranks reporting it | Node indices |
| --- | --- | --- |
| `.45092.22875271438464.tmp` | 81, 35, 107, 33 | 4, 10, 13 |
| `.45088.22948413677696.tmp` | 5, 79, 72, 33 | 0, 4, 9 |
| `.45089.23386800717952.tmp` | 19, 92 | 2, 11 |

Several readers on *different* nodes report the *same* temp filename. That is what one
writer's temp file seen by many readers looks like — every reader reports the writer's
`pid.tid`. It does not require two processes to have picked the same PID.

**Why per-node `/tmp` fixes it.** Two effects compound. The directory is shared by 8 ranks
instead of 128, so there are far fewer readers to lose the race; and on `tmpfs` the
write→rename window is far narrower than on Lustre, which is consistent with the 1.5× compile
time difference measured below. Widening that window is what turns a theoretical race into a
frequent one.

80 warnings landed across the cache subdirectories — `inductor/codecache` (70),
`aotautograd` (6), `fxgraph` (4) — hitting **9 distinct ranks**. The `tmp` arm produced
**zero**.

### Most ranks survive; one does not

Nearly all of those 80 events were logged as warnings and recovered by recompiling.
Rank 20 escalated to a hard `InductorError` wrapping a `SubprocException` from the Triton
compile worker for `triton_poi_fused_add_gelu_0`, at shape 1664, and did not recover.

So the failure is probabilistic: the race is common, recovery is usual, and occasionally a
rank loses outright. At 128 ranks × 24 compilations that came out as 1 hard failure. A real
training job compiles far more often over far longer, so the expected number of hard
failures grows accordingly — which matches the report's account of this being invisible at
small scale and expensive at large scale.

### Relationship to the reported symptom

The report saw `JSONDecodeError`; this run saw `FileNotFoundError`. Both are the cache
*reader* losing a race against a writer on a shared directory: a partially written file
decodes as bad JSON, a file already renamed away fails to open. Which of the two you get
depends on where in the writer's sequence the reader lands. The report's exact exception was
not reproduced; the mechanism behind it was.

One deliberate difference: their failure killed a rank, which hung the rest at the next
collective. This test catches per-shape exceptions and keeps going, so the failing rank
still reports and still reaches the barrier (`barrier_after_compile_ok: true`). That
converts a hang into a diagnosable failure on purpose — a hang produces no evidence.

---

## Secondary finding: Lustre caching also costs ~50% compile time

Cold-cache compile time per rank, 24 kernels:

| Arm | min | max |
| --- | --- | --- |
| lustre | 42.0 s | 43.2 s |
| tmp | 28.3 s | 28.5 s |

Roughly **1.5× slower on Lustre** even when nothing fails, and the spread across 128 ranks
is tight in both arms (≈1 s), so this is a systematic metadata/latency cost rather than
noise. Cache placement is therefore a throughput question as well as a correctness one.

---

## The container's defaults are the thing being validated here

`templates/lumi_common.sh` already defaults `LAIF_CACHE_MODE=tmp` and keys the cache by
container ID. Phase 4 is the evidence that this default is load-bearing rather than
cosmetic: the same job fails with the caches on Lustre and passes with them on `/tmp`.

The container itself sets none of `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`TORCH_EXTENSIONS_DIR` or `MIOPEN_USER_DB_PATH`, so a user who does not set them gets
whatever the framework defaults to — typically `$HOME`, which is also shared. Setting safe
per-node defaults in the image, or documenting them prominently, would remove this failure
class for everyone. This is the strongest actionable recommendation to come out of the
exercise so far.

---

## Two defects in my own test, found and fixed

Recording these because both would have produced a false clean result, and one already did.

**1. Dynamo collapsed the shape sweep (silent no-op).** The first 128-rank run passed with
24 shapes — but only the first two shapes actually compiled (9.9 s, 1.8 s) and shapes 3–24
took **0.00 s**. Dynamo's `automatic_dynamic_shapes` notices a changing dimension after the
second recompile and emits one dynamic kernel serving all later shapes, so a 24-shape run
generated exactly as many cache entries as a 6-shape one (82 Triton files in both). The
escalation applied no additional pressure and the "pass" meant nothing.

Fixed with `dynamic=False`, `automatic_dynamic_shapes=False`, and `torch._dynamo.reset()`
before each shape — the reset also forces the on-disk cache to be re-read, which is the path
the report's `JSONDecodeError` came from. After the fix: 962 cache files and 24 real
compilations per rank, and the failure appeared.

A `compilations_min` guard gate now fails any run where Dynamo collapsed the sweep, so this
cannot recur silently. Verified by replaying the old numbers: `enough_real_compilations: 2
below minimum 6`.

**2. The `len` reducer treated an empty list as missing data.** `eval_gates.py` returned
`None` for `len([])`, and since a missing value counts as a failure, a perfectly clean run
(`ranks_missing: []`) was reported as failing. Fixed so `len` is defined on empty lists while
`avg`/`last`/`max`/`min` still report missing on no data.

---

## Earlier 64-rank runs — superseded

The first Phase 4 runs (64 ranks, 6 shapes, both modes) both passed, as did the 128-rank
24-shape runs before the Dynamo fix. All four are superseded by defect 1 above: they
performed only 2 real compilations each. They are not evidence that Lustre caching is safe
at 64 ranks — they are evidence that the test was not yet applying pressure. Re-running the
64-rank case with forced compilation would establish where the threshold actually sits.

---

## Gate status

`gate5-jit-cache-under-rank-pressure`, 128 ranks, forced compilation:

| Gate | lustre | tmp |
| --- | --- | --- |
| `jit_cache_passed` | **fail** | pass |
| `no_missing_ranks` | pass (0) | pass (0) |
| `no_cache_corruption` | **fail** (1) | pass (0) |
| `no_failed_ranks` | **fail** (1) | pass (0) |
| `post_compile_barrier_ok` | pass | pass |
| `all_ranks_reported` | pass (128) | pass (128) |
| `cache_entries_created` | pass (962) | pass (962) |
| `enough_real_compilations` | pass (18) | pass (24) |

The suite now has a demonstrated true positive: it fails on a real, reproducible defect and
passes on the corrected configuration. That was the outstanding gap noted in
`docs/VALIDATION.md` — a gate suite that has never failed has not been shown to detect
anything.

---

## Next

- Re-run 64 ranks with forced compilation to locate the scale threshold.
- The `pid.tid` collision is an upstream PyTorch issue worth reporting, independent of LUMI.
- Still open from earlier phases: the communicator-count stress test proposed in
  `docs/EP32_16NODE_RESULTS.md`, which remains the best remaining lead on report §4.1.
