**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** Set per-node JIT cache defaults in the image (Triton/Inductor caches on shared storage corrupt at scale)

---

## Summary

The image sets none of `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR` or
`MIOPEN_USER_DB_PATH`. A user who does not set them gets the framework default, which is under
`$HOME` — shared across all nodes. At scale that produces cache-read failures, one of which can
kill a rank and hang the rest of the job at the next collective.

Setting per-node defaults in the image would remove the failure class for every user. It is
invisible below roughly 64 ranks, so users hit it only after scaling up, and the error message
does not name its cause.

## Reproduction

`20260513_121430`, 16 nodes / 128 ranks / 8 ranks per node, 24 forced Inductor compilations per
rank against a cold cache. The two arms are identical except for cache location:

| Cache location | Ranks failing | Warnings | Compile time/rank | Result |
| --- | --- | --- | --- | --- |
| Lustre (`/scratch/...`) | **1 of 128** | 80 across 9 ranks | 42.0–43.2 s | **fail** |
| per-node `/tmp` | none | **0** | 28.3–28.5 s | pass |

Jobs 20684269 (Lustre) and 20684441 (`/tmp`).

The surviving ranks recover by recompiling; rank 20 escalated to a hard `InductorError`
wrapping a `SubprocException` from the Triton compile worker for `triton_poi_fused_add_gelu_0`
and did not recover. So the failure is probabilistic — its rate scales with how much a job
compiles, which is why it is invisible in short tests.

## Mechanism (upstream, but the container controls the mitigation)

`torch/_inductor/codecache.py`:

- `write_atomic` (line 454) places its temp file in the **same directory as the target**:
  `tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"`
- `GuardedCache.iterate_over_candidates` (≈lines 1031–1044) lists that directory and `open()`s
  **every** entry, with no filtering of temp files.

So a reader lists the cache directory, sees another rank's in-flight `.tmp`, and by the time it
opens it the writer has renamed it away:

```
[rank81] torch/_inductor/codecache.py:1040] FileNotFoundError: [Errno 2] No such file or directory:
  '<lustre>/inductor/fxgraph/qz/fqzsuavdiy.../.45092.22875271438464.tmp'
```

Per-node `/tmp` fixes it for two compounding reasons: the directory is shared by 8 ranks instead
of 128, and the `tmpfs` write→rename window is far narrower than Lustre's — consistent with the
1.5× compile-time difference above.

Warnings landed across `inductor/codecache` (70), `aotautograd` (6) and `fxgraph` (4).

I am filing the reader-side behaviour separately against `pytorch/pytorch`; this issue is about
the container's defaults, which are effective regardless of whether upstream changes.

## Secondary benefit

Lustre-backed caching costs ~1.5× compile time even when nothing fails (42–43 s vs 28.3–28.5 s
per rank for the same 24 kernels), with a tight ≈1 s spread across 128 ranks in both arms — so
it is a systematic metadata/latency cost, not noise. Cache placement is a throughput question
as well as a correctness one.

## Suggested change

Set the four variables in the image to per-node paths keyed by container identity, so an
incompatible image cannot reuse another's artifacts and the compile cost is paid once per
container per node rather than once per job:

```bash
LAIF_CACHE_ROOT=/tmp/laif-cache-${USER}/<container-id>
TRITON_CACHE_DIR=${LAIF_CACHE_ROOT}/triton
TORCHINDUCTOR_CACHE_DIR=${LAIF_CACHE_ROOT}/inductor
TORCH_EXTENSIONS_DIR=${LAIF_CACHE_ROOT}/extensions
MIOPEN_USER_DB_PATH=${LAIF_CACHE_ROOT}/miopen
```

Two caveats worth deciding on:

1. The directories must be created per node inside the job step, not at image build time.
2. If this is done via the OCI `ENTRYPOINT`, it will not take effect for users launching with
   `apptainer exec`, which is the documented LUMI pattern — see the separate issue on the
   ENTRYPOINT/`exec` interaction. `ENV` directives in the image would apply under both `exec`
   and `run`; only the `mkdir` needs to happen in the step.

Even documenting the four variables prominently would help, but a default is better: the
failure is silent at small scale and expensive at large scale.

## Environment

- Image: `lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif`
- Digest: `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
- PyTorch `2.10.0+rocm7.0` (LUMI build `20260513142306`), Triton `3.6.0`
- LUMI `standard-g`, 16 nodes, 8 ranks/node
