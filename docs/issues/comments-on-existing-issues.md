Comments to add to existing issues rather than filing duplicates. Targets both
`laifs-container-recipes` and `Lumi-supercomputer/LUMI-AI-Guide`.

---

## Comment on #30 — "Setting `NCCL_NET_GDR_LEVEL` may cause jobs to hang"

> Independent confirmation on `u24r70f21m50t210-20260513_121430`, in case another data point is
> useful.
>
> Our multi-node benchmark templates had `NCCL_NET_GDR_LEVEL=PHB` and
> `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3` set together. A 2-node allreduce hung indefinitely;
> removing both made the same job complete in 24 s (job 19624583). We now default them off and
> keep them behind an explicit opt-in used only for deliberate fabric tuning.
>
> Matches your observation that performance is fine without forcing the GDR level — we see no
> measurable loss from leaving it unset.

---

## Comment on #20 — "RCCL communications sometimes hang with PyTorch DDP"

> Data point on whether this persists in the ROCm 7 / PyTorch 2.10 release you expected the fix
> in. We are on `full-u24r70f21m50t210-20260513_121430` (PyTorch `2.10.0+rocm7.0`, RCCL `2.26.6`)
> and still see intermittent hangs — but with a **node correlation** that may be worth checking
> against your own failures.
>
> Across a 5-run communicator-creation sweep (2/4/8/16 nodes, 8 ranks/node, up to 8 concurrent
> world-spanning communicators per rank), two runs hung and three passed:
>
> | Job | Nodes | Node list | Result |
> | --- | --- | --- | --- |
> | 20724372 | 4 | `nid[007038-007041]` | hang (killed at 10 min) |
> | 20711452 | 16 | `nid[007769-007784]` | hang (killed at 25 min) |
> | 20724354 | 2 | `nid[005556-005557]` | pass, 37 s |
> | 20724753 | 8 | `nid[006186-006193]` | pass, 45 s |
> | 20724970 | 16 | `nid[005724-005729,006186-006195]` | pass, 47 s |
>
> Both hangs on `nid007xxx`; all three passes on `nid005xxx`/`nid006xxx`. The result is
> non-monotonic in scale — 4 nodes hung while 8 and 16 passed — which argues against a
> rank-count or configuration cause and for node placement.
>
> Suggestion: if the `pytorch-ddp-multi-node` test failures are recorded with node lists, it may
> be worth checking whether they cluster the same way. If they do, part of this is a node-state
> problem rather than a container or PyTorch one, and `CUDA_LAUNCH_BLOCKING=1` may be masking a
> different cause than assumed.
>
> Separately, and possibly relevant to the straggler-rank theory: we found that omitting
> `device_id` from `init_process_group` reliably hung us at 128 ranks once a rank held more than
> one communicator (first communicator 13.9 s vs 1.14 s with `device_id`, second one never
> returning vs 0.29 s). Your #28 reproducer already passes `device_id`, so this is probably not
> your case — noting it because a single communicator per rank survives the wrong guess, so the
> symptom only appears in multi-communicator jobs.

---

## Comment on #28 — "Multi-node `torch.distributed.init` fails."

> One data point from validating the current release, in case it is useful for scoping.
>
> We could not reproduce a cross-node all-to-all or init failure at any group size (8/16/32) or
> node count (1/2/4/16), with rank-tagged correctness checking and uneven/zero-token dispatch —
> including four concurrent 32-rank meshes across 128 ranks, 2/2 runs, in ~33 s. Details and job
> IDs:
> https://github.com/aniskhan25/lumi-apptainer-bench/blob/feature/laif-container-validation/docs/FINDINGS.md
>
> Possibly relevant to the straggler-rank theory: omitting `device_id` from `init_process_group`
> reliably hung us at 128 ranks once a rank held more than one communicator (first communicator
> 13.9 s vs 1.14 s with `device_id`, second one never returning vs 0.29 s). Your reproducer already
> passes `device_id`, so this is probably not your case — noting it because a single communicator
> per rank survives the wrong guess, so the symptom only shows up in multi-communicator jobs.

---

## Comment on LUMI-AI-Guide #112 — "Document more environment variables"

> Data for the three on the list, measured inside
> `lumi-multitorch-full-u24r70f21m50t210-20260807_115122`. The main point is that **the value
> matters as much as documenting the name** — two of the three want node-local storage, not
> `/scratch`.
>
> The image sets none of them, so they fall back to framework defaults, and the three are not
> equivalent:
>
> | Variable | Default | Shared across the job's nodes? |
> | --- | --- | --- |
> | `TRITON_CACHE_DIR` | `/users/$USER/.triton/cache` | **yes** — `$HOME`, Lustre, 20 GB quota |
> | `TORCH_EXTENSIONS_DIR` | `/users/$USER/.cache/torch_extensions/py312_cpu` | **yes** — same |
> | `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torchinductor_$USER` | no — already node-local |
>
> Reproduce:
>
> ```bash
> singularity exec "$SIF" python3 -c "
> from triton import knobs
> from torch._inductor.runtime.cache_dir_utils import cache_dir
> from torch.utils.cpp_extension import _get_build_directory
> print(knobs.cache.dir)
> print(cache_dir())
> print(_get_build_directory('x', False))"
> ```
>
> So `TRITON_CACHE_DIR` and `TORCH_EXTENSIONS_DIR` are the two that change behaviour today.
>
> **The caveat worth writing down.** The guide's existing cache block sends `TORCH_HOME` to
> `/scratch` with the comment "to avoid saving to home directory". That is right for `TORCH_HOME`,
> which is large and read-mostly, but wrong for JIT caches, which are many small files written
> concurrently by every rank. Anyone extending the block by analogy would point these three at
> `/scratch` and make things worse.
>
> We measured that arm: 128 ranks (16 nodes x 8), 24 forced Inductor compilations per rank,
> identical except cache location.
>
> | Cache location | Failing ranks | Warnings | Compile time/rank |
> | --- | --- | --- | --- |
> | shared, on Lustre | 1 of 128 (hard `InductorError`, did not recover) | 80 across 9 ranks | 42.0-43.2 s |
> | node-local `/tmp` | none | 0 | 28.3-28.5 s |
>
> In a distributed job the dead rank hangs the rest at the next collective. Lustre was also ~1.5x
> slower even when nothing failed.
>
> Suggestion: add them to the existing per-node temp block rather than alongside `TORCH_HOME`, and
> add `MIOPEN_USER_DB_PATH` to the list too (see #81 — there is a variable-name bug there).
>
> ```bash
> JIT=$(mktemp -d)            # per node, inside the job step
> export TRITON_CACHE_DIR=$JIT/triton
> export TORCHINDUCTOR_CACHE_DIR=$JIT/inductor
> export TORCH_EXTENSIONS_DIR=$JIT/extensions
> ```

---

## Comment on LUMI-AI-Guide #81 — "Document the correct setting of temp dir for MIOpen" (closed)

> This may be worth reopening: the code that landed does not match what the issue prescribed.
>
> The issue body specifies:
>
> ```bash
> export MIOPEN_USER_DB_PATH=$MIOPEN_DIR/config
> ```
>
> The scripts in the guide use:
>
> ```bash
> export MIOPEN_USER_DB=$MIOPEN_DIR/config
> ```
>
> `MIOPEN_USER_DB` is not read by MIOpen. Checked against the library shipped in the current
> container (`libMIOpen.so.1.0.70002`), the only user-DB variable that exists is
> `MIOPEN_USER_DB_PATH`:
>
> ```console
> $ strings /opt/rocm/lib/libMIOpen.so | grep -oE 'MIOPEN_[A-Z0-9_]*' | sort -u | grep -E 'DB|CACHE'
> MIOPEN_CUSTOM_CACHE_DIR
> MIOPEN_DEBUG_DISABLE_FIND_DB
> MIOPEN_DISABLE_CACHE
> MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE
> MIOPEN_SYSTEM_DB_PATH
> MIOPEN_USER_DB_PATH
> ```
>
> `MIOPEN_USER_DB_PATH` appears 0 times in the repository; `MIOPEN_USER_DB` appears in 17 job
> scripts.
>
> Effect: `MIOPEN_CUSTOM_CACHE_DIR` is redirected as intended, so the kernel cache is fine. The user
> performance database is not redirected and stays at MIOpen's default `~/.config/miopen/`. So the
> permission-clash problem described in this issue is half-addressed — the half on `/tmp` is fixed,
> the half on a shared `$HOME` is not.
>
> One-line fix in each of the 17 scripts:
>
> ```diff
> -export MIOPEN_USER_DB=$MIOPEN_DIR/config
> +export MIOPEN_USER_DB_PATH=$MIOPEN_DIR/config
> ```

---

## Comment on laifs-container-recipes #39 — "vLLM and compressed-tensors dependency conflict"

> Two data points on the current release, `20260807_115122`.
>
> **1. The conflict is in the shipped images, not only in derived ones.** No derivation or extra
> `pip install` needed:
>
> ```console
> $ singularity exec lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif pip check
> vllm 0.22.1+lumi.aif.gfx90a.0decac0 has requirement compressed-tensors==0.15.0.1,
>   but you have compressed-tensors 0.17.1.
> ```
>
> Same in `plus`. So the reporter's derived build inherited it rather than caused it.
>
> **2. It appears to be a metadata problem rather than a functional one.** Every
> compressed-tensors module in vLLM imports cleanly against 0.17.1 — all 32 submodules, including
> the whole `compressed_tensors_moe` family and every scheme:
>
> ```bash
> singularity exec "$SIF" python3 -c "
> import importlib, pkgutil
> root='vllm.model_executor.layers.quantization.compressed_tensors'
> pkg=importlib.import_module(root)
> mods=sorted(m.name for m in pkgutil.walk_packages(pkg.__path__, root+'.'))
> for m in mods: importlib.import_module(m)
> print(len(mods), 'modules imported OK')"
> # -> 32 modules imported OK
> ```
>
> That is import coverage only — it does not exercise a quantized model at runtime, and Triton is
> disabled on a login node, so any Triton-only path is untested here.
>
> So the actionable part is probably relaxing or correcting vLLM's pin rather than downgrading
> compressed-tensors, but that depends on whether 0.15.0.1 was pinned for a real incompatibility.
