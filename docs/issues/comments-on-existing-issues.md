Comments to add to existing issues rather than filing duplicates. Targets both
`laifs-container-recipes` and `Lumi-supercomputer/LUMI-AI-Guide`.

---

## #30 — "Setting `NCCL_NET_GDR_LEVEL` may cause jobs to hang" — DO NOT POST

Checked 2026-08-20: 8 comments, active to 2026-07-27. The last comment already reaches our
conclusion — *"the main point is that setting `NCCL_NET_GDR_LEVEL` should not be required anymore
anyways."* Our independent confirmation (job 19624583, indefinite hang → 24 s once removed) is
redundant. Kept here only as the provenance for why our templates default it off.

Worth taking *from* that thread rather than adding to it: `FI_MR_CACHE_MONITOR=userfaultfd` is
reported to have resolved a hang-before-training on some LUMI tickets. Untested by us, and a
candidate if we see startup hangs again.

---

## Comment on #20 — "RCCL communications sometimes hang with PyTorch DDP" — POSTABLE

Reproduced 2026-08-21 on the current release, **under the LUMI AI Guide's own documented launch
configuration**, 5/5 attempts. Creating world-spanning RCCL communicators intermittently never
returns.

> Reproduced on `lumi-multitorch-full-u24r70f21m50t210-20260807_115122` (digest `d70ec87f…`),
> `standard-g`, 4 nodes / 32 ranks.
>
> **Launch configuration is Chapter 5's, unmodified:** `singularity run`, `--ntasks-per-node=8
> --gpus-per-node=8 --cpus-per-task=7 --mem-per-gpu=60G`, the guide's `CPU_BIND_MASKS`,
> `MASTER_PORT="1${SLURM_JOB_ID:0-4}"`, `WORLD_SIZE=$SLURM_NPROCS`, `RANK`/`LOCAL_RANK` exported
> inside the container, no `ROCR_VISIBLE_DEVICES` (all 8 GCDs visible), and
> `init_process_group(backend="nccl")` with no `device_id`. The workload — several world-spanning
> groups — is the Megatron-like part and is what the guide does not cover.
>
> **Result: 5 of 5 attempts hung.** Each ran fine up to a point and then stopped, at a *different*
> communicator each time:
>
> | Attempt | Last line printed | Hung while creating |
> | --- | --- | --- |
> | 1 | `first collective on default PG done` (14.1 s) | communicator 1 |
> | 2 | `communicator 2/8 live` (13.7 s) | communicator 3 |
> | 3 | `communicator 3/8 live` (15.2 s) | communicator 4 |
> | 4 | `communicator 4/8 live` (16.7 s) | communicator 5 |
> | 5 | `communicator 2/8 live` (13.6 s) | communicator 3 |
>
> So it is not a fixed ceiling and not a resource limit at a particular count — each communicator
> before the stall is created in ~1.5 s, then one simply never completes. `init_process_group` itself
> returned in under 1.2 s every time; the stall is always in a later `new_group` + first collective.
>
> **Reproducer** (job 21436817, nodes `nid[007434,007455,007457,007461]`):
>
> ```python
> # comms_guide.py
> import os, time, torch, torch.distributed as dist
> T0 = time.time()
> LR = int(os.environ["LOCAL_RANK"]); R = int(os.environ["RANK"])
> def log(m):
>     if R == 0: print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
>
> log(f"visible GCDs={torch.cuda.device_count()} | LOCAL_RANK={LR}")
> torch.cuda.set_device(LR)
> dist.init_process_group(backend="nccl")
> log(f"init returned | world={dist.get_world_size()}")
> t = torch.ones(1 << 18, device=f"cuda:{LR}")
> dist.all_reduce(t); torch.cuda.synchronize()
> log("first collective on default PG done")
> for i in range(8):
>     g = dist.new_group()                                  # world-spanning
>     x = torch.ones(1 << 18, device=f"cuda:{LR}")
>     dist.all_reduce(x, group=g); torch.cuda.synchronize()  # forces the communicator to exist
>     log(f"communicator {i+1}/8 live")
> dist.barrier(); log("ALL DONE")
> ```
>
> ```bash
> #SBATCH --nodes=4 --gpus-per-node=8 --ntasks-per-node=8 --cpus-per-task=7 --mem-per-gpu=60G
> export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
> export MASTER_PORT="1${SLURM_JOB_ID:0-4}"
> export WORLD_SIZE=$SLURM_NPROCS
> CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"
> for i in 1 2 3 4 5; do
>   timeout 300 srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity run $SIF bash -c \
>     "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python3 -u comms_guide.py"
>   echo "attempt $i -> exit $? (124 = hung)"
> done
> ```
>
> **Corroborating runs**, separate allocations, our own harness (which additionally restricts
> visibility with `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID` and passes `device_id`):
>
> | Scale | Allocations | Hung | Passed |
> | --- | --- | --- | --- |
> | 4 nodes | 5 | 2 | 2 (+1 unrelated error) |
> | 16 nodes | 3 | **3** | 0 |
>
> So it occurs with and without restricted device visibility, and with both eager and lazy
> initialisation. Under eager init (`device_id` passed) the stall moves *into*
> `init_process_group`, since that is where the first communicator is then built.
>
> **Limitations, stated plainly:**
>
> - The 5/5 attempts shared **one allocation** on one set of four nodes, so that is one independent
>   sample, not five. It establishes that the documented configuration hangs; it does not establish a
>   rate. The per-allocation numbers in the table above are the better rate estimate.
> - `exit 124` is our own 300 s cap. We know these did not finish in 300 s; we did not test whether
>   they would ever finish.
> - We have not captured which rank is the straggler. `TORCH_NCCL_TRACE_BUFFER_SIZE` plus
>   `TORCH_NCCL_DUMP_ON_TIMEOUT` on this reproducer would give that, and is the obvious next step if
>   useful to you.

---

## #28 — "Multi-node `torch.distributed.init` fails." — DO NOT POST

Checked 2026-08-20: 7 comments, and the last one (2026-04-23, from the reporter) is *"Training job
on 16 nodes seems to run nice and stable!"* The thread has moved on. Our contribution was "we could
not reproduce it either", which agrees with a resolved issue and adds nothing. Our runs were also on
`20260513_121430`, so we could not have claimed the current release without re-running.

---

## Comment on LUMI-AI-Guide #112 — "Document more environment variables"

**STATUS: noted by the guide maintainer (2026-08-20).** Kept for reference; no further action
needed unless they ask for the underlying data.

> The block merged in #108 is a good template for these three — same shape, same place. Two things
> we measured that might be worth folding in.
>
> **1. The three are not equivalent.** Measured inside
> `lumi-multitorch-full-u24r70f21m50t210-20260807_115122`, which sets none of them:
>
> | Variable | Default | Shared across the job's nodes? |
> | --- | --- | --- |
> | `TRITON_CACHE_DIR` | `/users/$USER/.triton/cache` | **yes** — `$HOME`, Lustre, 20 GB quota |
> | `TORCH_EXTENSIONS_DIR` | `/users/$USER/.cache/torch_extensions/py312_cpu` | **yes** — same |
> | `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torchinductor_$USER` | no — already node-local |
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
> itself. `TORCH_HOME` on `/scratch` is correct — large, read-mostly, genuinely shared. JIT caches are
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

## Comment on laifs-container-recipes #39 — "vLLM and compressed-tensors dependency conflict"

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

---

## Comment on LUMI-AI-Guide #111 — "Not all VRAM can be used by PyTorch"

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
> | plain DDP | 1 | ~1 GiB — negligible |
> | Megatron-style TP/PP/DP(/EP) | 8-10 | `950 + 9 x 653` ≈ **6.8 GiB**, ~11% of a 63.98 GiB GCD |
>
> This matches a user report we were validating: they could not exceed ~57 GiB against 63.98 GiB
> nameplate — a ~7 GiB gap, consistent across three independent configurations — and eventually
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
