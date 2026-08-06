**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** Set per-node JIT cache defaults in the image (Triton/Inductor caches on shared storage corrupt at scale)

---

## Summary

The image sets none of `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR` or
`MIOPEN_USER_DB_PATH`, so each falls back to its framework default. Measured inside the `full`
image, those defaults are split between `$HOME` and node-local `/tmp`:

| Variable | Default in this image | Shared across nodes? |
| --- | --- | --- |
| `TRITON_CACHE_DIR` | `/users/$USER/.triton/cache` | **yes** (`$HOME`) |
| `TORCH_EXTENSIONS_DIR` | `/users/$USER/.cache/torch_extensions/py312_cpu` | **yes** (`$HOME`) |
| `MIOPEN_USER_DB_PATH` | `~/.config/miopen/` | **yes** (`$HOME`) |
| `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torchinductor_$USER` | no (node-local) |

Two problems follow. The `$HOME` defaults put JIT artifacts on a shared, quota-limited filesystem
that every LUMI guide tells users to stay off. And once a user acts on that advice and redirects
caches to `/scratch`, they land on a genuine failure: at 128 ranks a shared Inductor cache on
Lustre produced a hard `InductorError` that killed a rank, which in a distributed job hangs the
rest at the next collective.

Setting per-node defaults in the image would remove the failure class for every user. It is
invisible below roughly 64 ranks, so users hit it only after scaling up, and the error message does
not name its cause.

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

Note that this arm set `TORCHINDUCTOR_CACHE_DIR` explicitly. The default would have been
node-local, so this is not the out-of-the-box configuration — see "How users reach the failing
configuration" below.

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

## How users reach the failing configuration

The Inductor default is node-local, so a completely naive user does not hit the failure above.
They are actively steered into it instead.

The LUMI-AI-Guide sets, in **17 of its job scripts**, a block whose stated purpose is to keep
caches off `$HOME`:

```bash
# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# Set your TORCH_HOME cache to scratch to avoid saving to home directory
export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
```

(from `05-multi-gpu-and-node/run_ddp_srun_4.sh` @ `3705c3c`; the 10-LLM-inference scripts do the
same with `HF_HOME`.)

So the ecosystem's guidance is explicit — *redirect caches to `/scratch`, do not write to
`$HOME`* — and the block covers MIOpen's kernel cache and the torch hub cache but **none** of
`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` or `TORCH_EXTENSIONS_DIR`. A user who notices those
three are missing and completes the pattern by pointing them at `/scratch`, exactly as instructed
for the others, has built the failing configuration. That is what our reproduction is.

Meanwhile the two variables the guide leaves alone default to `$HOME`, which on LUMI is a 20 GB
quota shared by all nodes of the job. `TORCH_EXTENSIONS_DIR` in particular is a build directory
with lock files, written concurrently by every rank in the job.

Either way the outcome is bad, and the user has no way to pick correctly from the current
documentation.

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

`TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR` and `MIOPEN_USER_DB_PATH` change behaviour today (they
move off `$HOME`). `TORCHINDUCTOR_CACHE_DIR` is already node-local by default; setting it
explicitly is about pinning that guarantee and keying it by container, so a user redirecting caches
does not silently break it.

Two caveats worth deciding on:

1. The directories must be created per node inside the job step, not at image build time.
2. If this is done via the OCI `ENTRYPOINT`, it will not take effect for users launching with
   `apptainer exec` — see the separate issue on the ENTRYPOINT/`exec` interaction. `ENV`
   directives in the image apply under both `exec` and `run` and are the right vehicle here; only
   the `mkdir` needs to happen in the step.

If a default is not wanted, documenting the four variables — and stating that they must point at
node-local storage, not `/scratch` — would still be a large improvement over the current state,
where the only published guidance points the other way.

## Environment

- Image: `lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif`
- Digest: `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
- PyTorch `2.10.0+rocm7.0` (LUMI build `20260513142306`), Triton `3.6.0`
- LUMI `standard-g`, 16 nodes, 8 ranks/node
- Defaults in the table measured inside the image itself (`triton.knobs.cache.dir`,
  `torch._inductor.runtime.cache_dir_utils.cache_dir()`,
  `torch.utils.cpp_extension._get_build_directory`, and `strings libMIOpen.so`)
