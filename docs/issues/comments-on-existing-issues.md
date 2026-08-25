Comments to add to existing issues rather than filing duplicates. Targets both
`laifs-container-recipes` and `Lumi-supercomputer/LUMI-AI-Guide`.

---

## #30; "Setting `NCCL_NET_GDR_LEVEL` may cause jobs to hang". DO NOT POST

Checked 2026-08-20: 8 comments, active to 2026-07-27. The last comment already reaches our
conclusion; *"the main point is that setting `NCCL_NET_GDR_LEVEL` should not be required anymore
anyways."* Our independent confirmation (job 19624583, indefinite hang → 24 s once removed) is
redundant. Kept here only as the provenance for why our templates default it off.

Worth taking *from* that thread rather than adding to it: `FI_MR_CACHE_MONITOR=userfaultfd` is
reported to have resolved a hang-before-training on some LUMI tickets. Untested by us, and a
candidate if we see startup hangs again.

---

## Comment on #20; "RCCL communications sometimes hang with PyTorch DDP", POSTABLE

Reproduced and localised on the current release, under Chapter 5's documented launch configuration.
**The first collective on a newly created process group intermittently never returns**, and PyTorch's
watchdog cannot see it.

> `lumi-multitorch-full-u24r70f21m50t210-20260807_115122` (digest `d70ec87f…`), `standard-g`,
> 4 nodes / 32 ranks. Launch configuration is Chapter 5's srun recipe unmodified, `singularity run`,
> `--ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7 --mem-per-gpu=60G`, the guide's
> `CPU_BIND_MASKS`, `MASTER_PORT="1${SLURM_JOB_ID:0-4}"`, `WORLD_SIZE=$SLURM_NPROCS`,
> `RANK`/`LOCAL_RANK` exported inside the container, no `ROCR_VISIBLE_DEVICES`, and no `device_id`.
>
> **Where it stops.** Instrumenting each step and logging from *every* rank:
>
> ```python
> for i in range(1, 9):
>     log(f"A calling new_group {i}")
>     g = dist.new_group()
>     log(f"B group {i} object created")
>     x = torch.ones(1 << 18, device=f"cuda:{LR}")
>     dist.all_reduce(x, group=g)
>     log(f"C allreduce {i} enqueued")
>     torch.cuda.synchronize()
>     log(f"D communicator {i}/8 live")
> ```
>
> In every hung run, **all 32 ranks print B and none print C**:
>
> | Attempt | Stalls on group | Ranks reaching B | Ranks reaching C |
> | --- | --- | --- | --- |
> | 1 | 5 | 32 | **0** |
> | 2 | 4 | 32 | **0** |
> | 3 | 2 | 32 | **0** |
> | 4 | 8 | 32 | **0** |
>
> So `new_group()` returns fine, being only bookkeeping, and the block is inside the **first
> `all_reduce` on that group**, which is where RCCL lazily initialises the communicator. The group
> index varies (5, 4, 2, 8), so it is not a ceiling or a count-dependent limit. Each preceding
> communicator comes up in ~1.5 s.
>
> **There is no straggler rank.** All 32 ranks block at the same call. This is not one slow
> participant. The whole world stops together.
>
> **PyTorch's timeout machinery does not catch it.** With `timeout=timedelta(seconds=90)` on the
> process group and a 240 s wall cap, across four hangs: **zero** `Watchdog caught` messages, zero
> `DistBackendError`, and **zero** flight-recorder dumps despite `TORCH_FR_BUFFER_SIZE=2000` and
> `TORCH_NCCL_DUMP_ON_TIMEOUT=1`. That is consistent with the block occurring *before* a `WorkNCCL`
> is enqueued: there is no work item to time out and nothing for the recorder to record. The
> practical consequence is that the job hangs indefinitely producing no error and no diagnostics
> which matches this issue's title.
>
> **Rates and node lists** (each row a separate allocation):
>
> | Job | Nodes | Attempts | Hung |
> | --- | --- | --- | --- |
> | 21436817 | `nid[007434,007455,007457,007461]` | 5 | **5** |
> | 21499264 | `nid[007745-007748]` | 5 | **4** |
> | 21492040 | `nid[006972-006975]` | 3 | 1 |
>
> Also seen with our own harness (restricted visibility via `ROCR_VISIBLE_DEVICES` plus `device_id`,
> so eager init): 2 of 5 allocations at 4 nodes, and **3 of 3 at 16 nodes**. Under eager init the
> stall moves into `init_process_group`, because that is where the first communicator is then built
> same failure, different call site.
>
> The `nid007xxx` allocations fared much worse than the `nid006xxx` one (9/10 vs 1/3). We previously
> tested and rejected a node-placement hypothesis on a different dataset, so we are not asserting one
> but the node lists are here in case they line up with your failures.
>
> **Limitations.** `exit 124` is our own wall cap. We did not test whether these would eventually
> return. Repeated attempts within one allocation are not independent samples. We have not captured
> RCCL-internal state during the stall; `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,NET` on this
> reproducer would be the next step, and we are happy to run it.

---

## #28; "Multi-node `torch.distributed.init` fails.". DO NOT POST

Checked 2026-08-20: 7 comments, and the last one (2026-04-23, from the reporter) is *"Training job
on 16 nodes seems to run nice and stable!"* The thread has moved on. Our contribution was "we could
not reproduce it either", which agrees with a resolved issue and adds nothing. Our runs were also on
`20260513_121430`, so we could not have claimed the current release without re-running.

---

## Comment on LUMI-AI-Guide #112; "Document more environment variables"

**STATUS: noted by the guide maintainer (2026-08-20).** Kept for reference; no further action
needed unless they ask for the underlying data.

> The block merged in #108 is a good template for these three; same shape, same place. Two things
> we measured that might be worth folding in.
>
> **1. The three are not equivalent.** Measured inside
> `lumi-multitorch-full-u24r70f21m50t210-20260807_115122`, which sets none of them:
>
> | Variable | Default | Shared across the job's nodes? |
> | --- | --- | --- |
> | `TRITON_CACHE_DIR` | `/users/$USER/.triton/cache` | **yes**; `$HOME`, Lustre, 20 GB quota |
> | `TORCH_EXTENSIONS_DIR` | `/users/$USER/.cache/torch_extensions/py312_cpu` | **yes**, same |
> | `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torchinductor_$USER` | no; already node-local |
>
> ```bash
> singularity exec "$SIF" python3 -c "
> from triton import knobs
> from torch._inductor.runtime.cache_dir_utils import cache_dir
> from torch.utils.cpp_extension import _get_build_directory
> print(knobs.cache.dir); print(cache_dir()); print(_get_build_directory('x', False))"
> ```
>
> So `TRITON_CACHE_DIR` and `TORCH_EXTENSIONS_DIR` are the two that change behaviour today.
> `TORCHINDUCTOR_CACHE_DIR` is worth setting anyway, to pin the guarantee rather than inherit it.
>
> **2. Node-local, not `/scratch`.** The block above `TORCH_HOME` is the right model, not `TORCH_HOME`
> itself. `TORCH_HOME` on `/scratch` is correct: large, read-mostly, genuinely shared. JIT caches are
> the opposite: many small files, write-heavy, written concurrently by every rank. Extending the
> `TORCH_HOME` line by analogy would make things worse, so the distinction may be worth stating
> explicitly.
>
> We measured that arm: 128 ranks (16 nodes x 8), 24 forced Inductor compilations per rank, identical
> except cache location.
>
> | Cache location | Failing ranks | Warnings | Compile time/rank |
> | --- | --- | --- | --- |
> | shared, on Lustre | 1 of 128 (hard `InductorError`, no recovery) | 80 across 9 ranks | 42.0-43.2 s |
> | node-local `/tmp` | none | 0 | 28.3-28.5 s |
>
> In a distributed job the dead rank hangs the rest at the next collective. Lustre was also ~1.5x
> slower even when nothing failed.
>
> Suggested addition, following #108 exactly:
>
> ```bash
> export TRITON_CACHE_DIR="/tmp/triton-cache-${USER}"
> export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-cache-${USER}"
> export TORCH_EXTENSIONS_DIR="/tmp/torch-extensions-${USER}"
> srun mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
> ```

---

## Comment on laifs-container-recipes #39; "vLLM and compressed-tensors dependency conflict"

**STATUS: shared with the container maintainer (2026-08-20).** Kept for reference.

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
> compressed-tensors module in vLLM imports cleanly against 0.17.1; all 32 submodules, including
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
> That is import coverage only; it does not exercise a quantized model at runtime, and Triton is
> disabled on a login node, so any Triton-only path is untested here.
>
> So the actionable part is probably relaxing or correcting vLLM's pin rather than downgrading
> compressed-tensors, but that depends on whether 0.15.0.1 was pinned for a real incompatibility.

---

## Comment on LUMI-AI-Guide #111; "Not all VRAM can be used by PyTorch"

> **STATUS: POSTED 2026-08-19.** A condensed version went up as the fourth comment on the issue,
> carrying the load-bearing parts: ~1 GiB for the initial RCCL/communicator overhead plus ~0.65 GiB
> per additional communicator per rank, the fact that none of it appears in
> `torch.cuda.memory_allocated()`, and that it scales with communicator count rather than world size.
> No maintainer reply yet.
>
> Not included, and still available if the thread continues: the per-world-size table below
> (651.5 / 626.8 / 631.8 MiB at 16 / 64 / 128 ranks, which is the evidence that the cost is flat in
> world size), the ~90 MiB HIP-context figure, and the `expandable_segments` no-op. The last of these
> is the most likely to be useful, since PyTorch's own OOM message recommends setting it.
>
> The draft below is kept as written for reference.

> We measured this on `20260513_121430` while investigating a user report that ran into it, so here
> are numbers for the footnote. On the sizing question in the thread: the effect is small for a plain
> DDP job and large for multi-dimensional parallelism, which may be worth reflecting in where it goes.
>
> **Per-communicator cost, measured at three world sizes:**
>
> | Ranks | Device memory per RCCL communicator |
> | --- | --- |
> | 16 | 651.5 MiB |
> | 64 | 626.8 MiB |
> | 128 | 631.8 MiB |
>
> So ~**630-650 MiB per communicator**, essentially independent of world size. The marginal curve at
> 16 ranks: ~90 MiB for the HIP context, ~950 MiB for the *first* communicator (RCCL one-time init
> included), then a flat ~653 MiB for each additional one.
>
> **It is invisible to PyTorch.** Throughout all of the above, `torch.cuda.memory_allocated()`
> reported **0.0 MiB**. The cost only shows in `torch.cuda.mem_get_info()`. That is why it surprises
> people: the obvious API says the memory is free.
>
> **Why the impact is uneven.** Communicator count is what scales it:
>
> | Job shape | Communicators/rank | Hidden cost |
> | --- | --- | --- |
> | plain DDP | 1 | ~1 GiB, negligible |
> | Megatron-style TP/PP/DP(/EP) | 8-10 | `950 + 9 x 653` ≈ **6.8 GiB**, ~11% of a 63.98 GiB GCD |
>
> This matches a user report we were validating: they could not exceed ~57 GiB against 63.98 GiB
> nameplate; a ~7 GiB gap, consistent across three independent configurations, and eventually
> settled on ~40 GB as their planning figure after some weeks. So it is rare in headcount but
> expensive when it lands, and it lands on exactly the large-scale jobs that are hardest to debug.
>
> Suggested sizing rule if useful for chapter 10: subtract roughly
>
> ```
> 1 + 0.65 x (communicators - 1)   GiB
> ```
>
> before counting parameters, activations and optimizer state.
>
> **Related, same chapter:** `expandable_segments` is silently unsupported on this platform, so the
> allocator cannot compact fragmentation and it is *reserved* rather than *allocated* memory that
> determines failure:
>
> ```
> UserWarning: expandable_segments not supported on this platform
>   (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:40.)
> ```
>
> The option is accepted and ignored, which matters because PyTorch's own OOM message recommends
> setting it. Worth one line next to the VRAM note so users do not spend time on a no-op.
