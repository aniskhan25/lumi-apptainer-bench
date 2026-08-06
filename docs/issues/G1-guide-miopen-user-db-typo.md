**Repo:** `Lumi-supercomputer/LUMI-AI-Guide`
**Title:** `MIOPEN_USER_DB` is not read by MIOpen (should be `MIOPEN_USER_DB_PATH`), and the cache block omits the Triton/Inductor caches

---

## Summary

Two issues in the cache-setup block that is repeated across the guide's job scripts.

1. **`MIOPEN_USER_DB` is not a MIOpen environment variable.** The variable MIOpen reads is
   `MIOPEN_USER_DB_PATH`. As written, the export has no effect and MIOpen's user performance
   database stays at its default under `$HOME`.
2. **The block covers MIOpen and `TORCH_HOME` but not `TRITON_CACHE_DIR`,
   `TORCHINDUCTOR_CACHE_DIR` or `TORCH_EXTENSIONS_DIR`**, two of which also default to `$HOME`.

Both are small changes with a real effect on `$HOME` quota usage and, at multi-node scale, on job
reliability.

## 1. The MIOpen variable name

The block appears in **17** job scripts (`01-quickstart/run.sh`,
`03-file-formats/run-scripts/training-benchmarks/run-comp-vision-transformer.sh`,
`04-data-storage/run_ramfs.sh`, all eight `05-multi-gpu-and-node/run_{ddp,ds}_*.sh`,
`06-monitoring-and-profiling/run.sh`, `07`/`08`/`09` visualization scripts, and both
`10-LLM-inference` scripts), always as:

```bash
# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config
```

`MIOPEN_USER_DB_PATH` does not appear anywhere in the repository (0 occurrences).

Checked against the MIOpen shipped in the LAIF container
(`lumi-multitorch-full-u24r70f21m50t210-20260513_121430`, `libMIOpen.so.1.0.70002`) — the only
user-DB environment variable present in the library is `MIOPEN_USER_DB_PATH`:

```console
$ strings /opt/rocm/lib/libMIOpen.so | grep -oE 'MIOPEN_[A-Z0-9_]*' | sort -u | grep -E 'DB|CACHE'
MIOPEN_CUSTOM_CACHE_DIR
MIOPEN_DEBUG_DISABLE_FIND_DB
MIOPEN_DISABLE_CACHE
MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE
MIOPEN_SYSTEM_DB_PATH
MIOPEN_USER_DB_PATH
```

The library's default paths are also visible in the same binary — `~/.cache/miopen/` (kernel
cache, correctly overridden by `MIOPEN_CUSTOM_CACHE_DIR`) and `~/.config/miopen/` (user DB, not
overridden).

So the kernel cache is being redirected as intended and the user performance database is not.

**Suggested fix:** rename the variable in all 17 scripts.

```diff
-export MIOPEN_USER_DB=$MIOPEN_DIR/config
+export MIOPEN_USER_DB_PATH=$MIOPEN_DIR/config
```

## 2. The Triton, Inductor and C++ extension caches

Measured inside the same image, the defaults for the remaining JIT caches are:

| Variable | Default | On `$HOME`? |
| --- | --- | --- |
| `TRITON_CACHE_DIR` | `/users/$USER/.triton/cache` | **yes** |
| `TORCH_EXTENSIONS_DIR` | `/users/$USER/.cache/torch_extensions/py312_cpu` | **yes** |
| `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torchinductor_$USER` | no (node-local) |

The comment in the guide's own block — *"Set your TORCH_HOME cache to scratch to avoid saving to
home directory"* — applies just as much to the first two, and they are not covered. Any script
using `torch.compile`, Triton kernels, or a package that JIT-builds a C++ extension writes to
`$HOME` against a 20 GB quota.

The safe destination for these is **node-local `/tmp`**, not `/scratch`. That differs from
`TORCH_HOME` and `HF_HOME`, where `/scratch` is right because the content is large, read-mostly,
and genuinely shared. JIT caches are the opposite: small files, write-heavy, and written
concurrently by every rank.

This distinction matters concretely. On a shared Lustre Inductor cache at 128 ranks (16 nodes ×
8) with 24 forced compilations per rank, we measured 80 `FileNotFoundError` warnings across 9
ranks and one rank escalating to a hard `InductorError` that killed it and hung the job; the same
workload with the cache on node-local `/tmp` produced zero warnings and completed ~1.5× faster
(28.3–28.5 s vs 42.0–43.2 s per rank). Details and the upstream cause:
https://github.com/aniskhan25/lumi-apptainer-bench/blob/feature/laif-container-validation/docs/PHASE4_RESULTS.md

**Suggested addition** to the same block, reusing the existing per-node temp directory:

```bash
export TRITON_CACHE_DIR=$MIOPEN_DIR/triton
export TORCHINDUCTOR_CACHE_DIR=$MIOPEN_DIR/inductor
export TORCH_EXTENSIONS_DIR=$MIOPEN_DIR/torch_extensions
```

(With a rename of `MIOPEN_DIR` to something like `JIT_CACHE_DIR`, since it would no longer be
MIOpen-specific.)

One thing to note if you take this: `mktemp -d` runs once in the batch script, on the first node
only, so the directory exists only there and the other nodes get a path that does not yet exist.
That is harmless — `/tmp` is node-local on LUMI and all four frameworks create their cache
directories on demand — but it does mean the name is shared while the contents are per-node, which
is exactly the desired behaviour and worth a one-line comment so nobody "fixes" it by moving the
directory to `/scratch`.

I am filing a parallel request against `laifs-container-recipes` asking for these to be set as
image defaults, which would make the guide-side change unnecessary. The guide fix is worth making
regardless, since it applies to any container.

## Environment

- Guide at `3705c3c9a3ec0fd7f9e73980ab3cd41d29170c48`
- Container `lumi-multitorch-full-u24r70f21m50t210-20260513_121430`, PyTorch `2.10.0+rocm7.0`,
  Triton `3.6.0`, ROCm 7.0
- Defaults read from `triton.knobs.cache.dir`,
  `torch._inductor.runtime.cache_dir_utils.cache_dir()`, and
  `torch.utils.cpp_extension._get_build_directory()` inside the container
