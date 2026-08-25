**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** Set node-local JIT cache defaults in the image. Triton and C++ extension caches land on `$HOME`

**STATUS: NOT FILED. Treated as handled via `Lumi-supercomputer/LUMI-AI-Guide#112` (2026-08-21).**

For the record, the state of the public tracking as of 2026-08-21: #112 is open with 0 comments,
last updated 2026-08-06, and the three variables do not yet appear in guide `main`; the cache block
there still covers only `MIOPEN_*` and `TORCH_HOME`. So "handled" means tracked upstream, not yet
implemented.

One residual difference, noted and not pursued: #112 is guide-side documentation, which reaches users
who follow the guide's scripts. The ask below was an image-side `ENV` default, which additionally
reaches derived images and users who never read the guide. That difference is real but modest, and a
maintainer could reasonably decline it, so it is dropped rather than argued. Retained below as the
measurement record: the observed `$HOME` cache evidence is the part worth keeping.

---

## Summary

The image sets none of `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR` or
`MIOPEN_USER_DB_PATH`, so each falls back to a framework default. Two of them land on `$HOME`, which
on LUMI is Lustre with a 20 GB quota, shared by every node in the job.

## Reproduce

```bash
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif

# the image sets none of them
singularity inspect --environment "$SIF" | grep -cE 'TRITON_CACHE|TORCHINDUCTOR|TORCH_EXTENSIONS|MIOPEN'

# so where do they go?
singularity exec "$SIF" python3 -c "
from triton import knobs
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch.utils.cpp_extension import _get_build_directory
print('TRITON_CACHE_DIR       ', knobs.cache.dir)
print('TORCHINDUCTOR_CACHE_DIR', cache_dir())
print('TORCH_EXTENSIONS_DIR   ', _get_build_directory('x', False).rsplit('/',1)[0])"

df -hT "$HOME" | tail -1
```

```
0
TRITON_CACHE_DIR        /users/<user>/.triton/cache                      <- $HOME, Lustre
TORCHINDUCTOR_CACHE_DIR /tmp/torchinductor_<user>                        <- node-local, fine
TORCH_EXTENSIONS_DIR    /users/<user>/.cache/torch_extensions/py312_cpu  <- $HOME, Lustre
... lustre 20G ... /pfs/lustrep2
```

`MIOPEN_USER_DB_PATH` defaults to `~/.config/miopen/`, also `$HOME`.

And this is not only the computed default. It is where artifacts actually accumulate. On an account
that has run GPU work with `TRITON_CACHE_DIR` unset:

```console
$ find ~/.triton -type f | wc -l
386
$ du -sh ~/.triton
7.9M
$ find ~/.triton -type f | head -1
~/.triton/cache/UOWGAR7HQECQIKTN5LUUT5VSLTIKCRC3YJP3VNY7BQ7WNZ3I26XA/_gt_bwd_dst_pass.hsaco
$ find ~/.triton -type f -printf '%TY-%Tm\n' | sort | uniq -c
      1 2026-04
     10 2026-05
     36 2026-06
    339 2026-07
```

Those are real compiled AMD GPU binaries (`.hsaco`, plus `.llir` and `.json`), accumulated across
months of ordinary use. So the `$HOME` default is what users are living with, not a theoretical
path.

## Why it matters

Triton's cache is where every compiled kernel lands, and `TORCH_EXTENSIONS_DIR` is a build directory
with lock files. Both get written concurrently by every rank in the job, to one Lustre directory
under a 20 GB quota.

We measured the cost of a shared JIT cache directly: 128 ranks (16 nodes × 8), 24 forced Inductor
compilations per rank, identical runs except cache location.

| Cache location | Failing ranks | Warnings | Compile time/rank |
| --- | --- | --- | --- |
| shared, on Lustre | **1 of 128** | 80 across 9 ranks | 42.0–43.2 s |
| node-local `/tmp` | none | 0 | 28.3–28.5 s |

The one failing rank raised a hard `InductorError` and did not recover, which in a distributed job
hangs the remaining ranks at the next collective. Lustre was also ~1.5× slower even when nothing
failed. Jobs 20684269 and 20684441.

(Those runs were on `20260513_121430` and set `TORCHINDUCTOR_CACHE_DIR` explicitly, since its
default is already node-local. The underlying race is `pytorch#172144`, fixed upstream in the 2.11
line but **not** in the 2.10.0 these images ship (see the separate issue on that). Node-local cache
placement is the mitigation while on 2.10.)

Users are also steered toward the failing configuration. The LUMI-AI-Guide repeats a cache block in
18 job scripts that sends MIOpen's caches to node-local `/tmp` (correctly, and created per node with
`srun mkdir -p`; see `Lumi-supercomputer/LUMI-AI-Guide#108`) and `TORCH_HOME` to `/scratch`, while
covering none of the three torch/Triton JIT variables. Extending that block by analogy with the
`TORCH_HOME` line, rather than the MIOpen lines, builds exactly the arm that failed above.

## Suggested fix

Set the four variables as image defaults, on node-local storage, keyed by container identity:

```bash
LAIF_CACHE_ROOT=/tmp/laif-cache-${USER}/<container-id>
TRITON_CACHE_DIR=${LAIF_CACHE_ROOT}/triton
TORCHINDUCTOR_CACHE_DIR=${LAIF_CACHE_ROOT}/inductor
TORCH_EXTENSIONS_DIR=${LAIF_CACHE_ROOT}/extensions
MIOPEN_USER_DB_PATH=${LAIF_CACHE_ROOT}/miopen
```

Two notes:

1. Use `ENV`, not the `ENTRYPOINT`; `apptainer exec` does not run an `ENTRYPOINT`.
2. The directories must be created per node inside the job step, not at build time. The guide's
   `srun mkdir -p` in #108 is the working precedent.

## Relationship to LUMI-AI-Guide #112

`Lumi-supercomputer/LUMI-AI-Guide#112` ("Document more environment variables") already tracks
documenting `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and `TORCH_EXTENSIONS_DIR`, so the
documentation half is recognised upstream and this issue is not asking for that again.

The ask here is different and complementary: an **image default**, so the safe value applies to users
who never read the guide, and to anyone building a derived image. Documentation alone leaves the
default wrong for everyone who does not act on it; and the problem is invisible at small scale, so
most users will not know they need to.

## Environment

- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`
  (digest `d70ec87fda17e97ff3b3241bcb34774365bba5f7b9172a22b8fda0897213bc81`)
- PyTorch 2.10.0+rocm7.0, Triton 3.6.0
- `$HOME` on Lustre, 20 GB quota
