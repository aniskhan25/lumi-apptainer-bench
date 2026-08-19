# LAIF container validation — consolidated findings

Response to the `project_465003047` (VNGRS) LUMI-G experience report, from measurements on the
container that is current today.

**Image under test**

```
/appl/local/laifs/containers/lumi-multitorch-latest.sif
  -> lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
sha256 d70ec87fda17e97ff3b3241bcb34774365bba5f7b9172a22b8fda0897213bc81
```

**Scope.** The current release only. No comparison against earlier builds — whether a finding is a
regression is explicitly out of scope. Up to 16 nodes / 128 ranks. Pure PyTorch: no Megatron-Core,
so findings that require it are marked untested rather than refuted.

**Provenance.** The container-inspection findings below were verified on the image named above. The
multi-node GPU measurements (§4.7, §4.2, §4.1/§4.4) were taken on `20260513_121430`, an earlier
release, and have not been re-run on the current one; each is labelled where it appears.

**Environment confirmed loaded inside the job:** PyTorch `2.10.0+rocm7.0` (LUMI build
`20260513142306`), HIP `7.0.51831`, RCCL `2.26.6`, Triton `3.6.0`, flash-attn `2.8.4`,
apex `1.10.0`, Megatron-Core `0.15.0rc8`, aws-ofi-nccl `1.19.1-git-206c02c`, MI250X
`gfx90a:sramecc+:xnack-`, 63.98 GiB/GCD. Every version the report quotes matches exactly.

---

## Summary

| # | Report finding | Status |
| --- | --- | --- |
| 4.1 | Container update regressed inter-node all-to-all (`PTLTE_NOT_FOUND`) | **Not reproduced** — hypotheses exhausted |
| 4.2 | Usable HBM materially below nameplate | **Confirmed and quantified** |
| 4.3 | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` from `torch.compile` | Not tested |
| 4.4 | 32-rank expert all-to-all fails on the fabric | **Not reproduced** in its own topology |
| 4.5 | Very long collective bootstrap at high rank counts | Not reproduced at ≤128 ranks; confound found |
| 4.6 | Two jobs on one Lustre dataset halves throughput | Not tested |
| 4.7 | Lustre-backed JIT caches corrupt under rank pressure | **Reproduced, root cause identified** |
| 4.8 | `--ckpt-format torch_dist` hangs on Lustre | Not tested (needs Megatron) |
| 4.9 | Per-partition QOS limit discoverability | Out of scope (Slurm policy) |
| 4.10 | Minor items (`realpath` binds, login Python, 383 vs 191.5 TF/s) | Partly addressed |
| §6 | Megatron-Core observations | Not tested; bundled version confirmed |

Plus seven issues found that the report does not raise — see [Independent findings](#independent-findings).

---

## 4.7 — Lustre JIT cache corruption: REPRODUCED

The one report finding fully reproduced, with a mechanism more specific than the report could
determine. Detail: [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md). Jobs 20684269, 20684441.

128 ranks, 24 forced compilations per rank, identical in every respect except cache location:

| Cache | Failed ranks | Corruption | Compile time | Result |
| --- | --- | --- | --- | --- |
| Lustre | **1 of 128** (rank 20) | yes | 42.0–43.2 s | **FAIL** |
| per-node `/tmp` | none | none | 28.3–28.5 s | PASS |

**Root cause.** Two pieces of torch source. `write_atomic` (`codecache.py:454`) writes its
temp file into the *same directory* as the target:
`tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"`. And
`GuardedCache.iterate_over_candidates` (`codecache.py:1031-1044`) lists that directory and
`open()`s **every** entry with no filtering of temp files. So a reader sees another rank's
in-flight `.tmp`, and by the time it opens it the writer has renamed it away —
`FileNotFoundError`, logged at :1040.

Several readers on *different* nodes reported the *same* temp filename
(`.45092.22875271438464.tmp` from ranks 81/35/107/33, i.e. node indices 4/10/13), which is one
writer's file seen by many readers rather than a PID collision. 80 such events hit 9 distinct
ranks; the `/tmp` arm produced zero.

Per-node `/tmp` fixes it for two compounding reasons: the directory is shared by 8 ranks
instead of 128, and on `tmpfs` the write→rename window is far narrower than on Lustre —
consistent with the 1.5x compile-time difference.

Most ranks recover by recompiling; one did not. The failure is probabilistic, so its rate
grows with how much a job compiles — matching the report's "invisible at small scale and
expensive at large scale".

The report saw `JSONDecodeError`; this run saw `FileNotFoundError`. Both are the cache *reader*
failing on an entry another rank was mid-write — a partially written file decodes as bad JSON,
a renamed-away file fails to open. Same race, different point in the writer's sequence. Their
exact exception was not reproduced; the mechanism behind it was.

**Also:** Lustre caching costs ~1.5× compile time even when nothing fails.

**Upstream.** The reader not skipping temp files is a PyTorch issue independent of LUMI and
worth filing — see [`ESCALATION.md`](ESCALATION.md) U1.

---

## 4.2 — Usable HBM: CONFIRMED, and the gap is now accounted for

Detail: [`PHASE1_RESULTS.md`](PHASE1_RESULTS.md), [`PHASE5_COMM_COUNT_RESULTS.md`](PHASE5_COMM_COUNT_RESULTS.md).

**`expandable_segments` is unsupported, verbatim:**

```
UserWarning: expandable_segments not supported on this platform
  (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:40.)
```

The allocation still succeeds — the option is accepted and silently ignored, which is why the
OOM message keeps recommending it. The allocator therefore cannot compact fragmentation, and
it is reserved rather than allocated memory that determines failure.

**The invisible memory, measured.** ~**630–650 MiB of device memory per RCCL communicator**,
essentially independent of world size (651.5 / 626.8 / 631.8 MiB at 16 / 64 / 128 ranks).
Throughout, `torch.cuda.memory_allocated()` reported **0.0 MiB** — PyTorch sees none of it.
Marginal curve at 16 ranks: ~90 MiB HIP context, ~950 MiB for the first communicator (RCCL
one-time init included), then a flat ~653 MiB each.

This accounts for the report's wall. They could not exceed ~57 GiB against 63.98 GiB
nameplate, a gap of ~7 GiB, consistent across three independent configurations. A job holding
8–10 communicators costs `950 + 9 × 653 ≈ 6.8 GiB` invisibly — the same magnitude. **Their
~40 GB planning figure is well-founded, not pessimistic.**

**Recommendation.** Publish a practical per-GCD ceiling, and note that sizing should subtract
roughly `1 + 0.65 × (communicators − 1)` GiB before counting parameters and activations.

---

## 4.1 and 4.4 — Fabric failures: NOT REPRODUCED

Detail: [`PHASE2_RESULTS.md`](PHASE2_RESULTS.md), [`PHASE3_RESULTS.md`](PHASE3_RESULTS.md),
[`EP32_16NODE_RESULTS.md`](EP32_16NODE_RESULTS.md), [`PHASE5_COMM_COUNT_RESULTS.md`](PHASE5_COMM_COUNT_RESULTS.md).

Every configuration tried passes, including the one §4.4 specifies exactly:

| Configuration | Nodes | Runs | Result |
| --- | --- | --- | --- |
| EP=8 (intra-node XGMI control) | 2 | 1 | pass |
| EP=16 (crosses Slingshot) | 2, 4 | 2 | pass |
| EP=32, one mesh | 4 | 3 | pass 3/3 |
| **EP=32, four concurrent meshes, 128 ranks** | **16** | **2** | **pass 2/2** |
| 8 communicators/rank (64/node) | 2, 8, 16 | 3 | pass |

All with rank-tagged payload verification, uneven MoE-shaped splits including zero-token
peers, and repeated communicator create/destroy. No `PTLTE_NOT_FOUND`, no "unhandled system
error".

**Hypotheses exhausted:** group size (8/16/32), node span (1/2/4/16), concurrent disjoint
meshes (4), communicator population per rank (8), uneven and zero-token dispatch.

---

## 4.5 — Long bootstrap: NOT REPRODUCED, and a confound identified

At 128 ranks a full job — container start, imports, four communicator groups, message sweep,
churn test — completes in **33–35 s**. Nothing pathological occurs at this scale. But 128
ranks is one eighth of the reported 1024 and bootstrap cost is not expected to be linear, so
this does not refute the 45-minute figure.

**The confound matters more than the null result.** Across a five-run communicator sweep, two
runs hung and three passed — non-monotonically, with 4 nodes hanging while 8 and 16 passed.
Node lists explain it:

| Job | Nodes | Result |
| --- | --- | --- |
| 20724372 | `nid[007038-007041]` | TIMEOUT |
| 20711452 | `nid[007769-007784]` | TIMEOUT |
| 20724354 | `nid[005556-005557]` | pass |
| 20724753 | `nid[006186-006193]` | pass |
| 20724970 | `nid[005724-005729,006186-006195]` | pass |

Both hangs on `nid007xxx`; all passes on `nid005xxx`/`nid006xxx`. This project's pre-existing
known-bad list already contains five `nid007xxx` entries, none overlapping these ranges — so
it is incomplete, not wrong.

**Consequence for the report.** §4.4's init failure and §4.5's long bootstraps are both the
shape this flakiness produces. Neither can be attributed to the container without the node
lists from those runs. **This is checkable against their job records and worth asking for.**

---

## Not tested

Stated plainly rather than left implied.

| Finding | Why | What it needs |
| --- | --- | --- |
| 4.3 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | Out of time; the JIT work exercised compile pressure but not HSA queue/event exhaustion | Variable-shape `torch.compile` loop with queue/fd/thread counters, `GPU_MAX_HW_QUEUES` sweep |
| 4.6 Two jobs on one Lustre dataset | Not attempted | Two concurrent jobs mapping one dataset; measure per-iteration time |
| 4.8 `torch_dist` checkpoint hang | Needs Megatron | Megatron save/restore at 1B–30B on Lustre |
| §6 Megatron observations | Deferred by scope | Bundled 0.15.0rc8 vs their 0.16.1 overlay |
| 4.9 QOS limits | Slurm policy, not container | — |

On §6 one fact is confirmed: the container ships Megatron-Core **`0.15.0rc8`**, so the version
they exercised (0.16.1, via `PYTHONPATH`) was never the version shipped. Their flag-level
findings are untested against either.

On 4.10: `/projappl` **is** visible inside the container (bindings module 1.0.1 binds `/pfs`
explicitly), so the `realpath` problem does not reproduce. The 383 vs 191.5 TF/s per-GCD
distinction is correct and affects this repo's own reporting.

---

## Independent findings

Issues found that the report does not raise.

### 1. `fi_info` / `fi_pingpong` shadowed by Intel MPI shims — container defect

```
$ singularity exec <full>.sif fi_info --version
/opt/venv/bin/fi_info: line 34: /opt/mpi/libfabric/bin/fi_info: No such file or directory
$ singularity exec <full>.sif /usr/bin/fi_info --version
/usr/bin/fi_info: 2.1.0
```

**Correction.** This section originally said the tool was missing from all four variants. Both
halves were wrong. `/usr/bin/fi_info` is present and works (libfabric 2.1.0, `cxi` provider
included); `libfabric`, `mpich` and `torch` are unaffected. Only `full` and `plus` fail, and the
cause is shadowing rather than absence: `impi-rt` 2021.18.1 — pulled in transitively by
`oneccl 2022.1.1` — installs its own `fi_info` wrapper into `/opt/venv/bin`, which is first on
`PATH`. That wrapper execs `$I_MPI_ROOT/opt/mpi/libfabric/bin/fi_info`, and `I_MPI_ROOT` is unset.

`fi_info` is the first tool anyone reaches for when debugging a Portals/CXI error. It appears
installed, then fails with a message naming neither libfabric nor the real problem. **The
reporter had no way to enumerate fabric providers from inside the container while diagnosing
§4.1 and §4.4** — though the workaround, had anyone known it, was `/usr/bin/fi_info`.

Confirmed unchanged on `20260807_115122`. Fix is `rm` of two files, or dropping `oneccl`.

### 1b. A duplicate `libmpi.so.12` from the same dependency — low severity

`impi-rt` also installs Intel MPI's `libmpi.so.12` into `/opt/venv/lib`, sharing a soname with the
system MPICH 5.0.1. Import order decides which a process gets.

**Investigated and largely benign.** On a GPU node, `import torch` then `from mpi4py import MPI`
gives **MPICH 5.0.1** and `MPI_Init` succeeds — the correct library. No shipped package forces the
other order: `megatron`, `vllm`, `transformer_engine` and `apex` never reference mpi4py, `deepspeed`
and `lightning` import torch first, and `h5py` is built without MPI support so its mpio path is
closed. The failing order (`mpi4py` before `torch`, which raises
`OSError: libmpicxx.so.12: undefined symbol: MPIX_Win_create_errhandler_x`) is not a pattern that
arises on LUMI, where ranks come from Slurm and collectives go through RCCL.

Recorded because it is fixed for free by the same `oneccl`/`impi-rt` removal as finding 1, not
because it needs its own escalation. An earlier version of this section claimed mpi4py could not use
Slingshot; that was wrong for the realistic import order.

### 2. GPU binding is inert under `apptainer exec` — container defect + docs

The image sets runtime variables via an OCI `ENTRYPOINT`
(`Containerfile:256–268`). `apptainer exec` does not run an ENTRYPOINT; only `run` does. Three
arms at one node:

| Launcher binds | Mode | `ROCR_VISIBLE_DEVICES` | Devices/rank |
| --- | --- | --- | --- |
| yes | `exec` | `'0'` | 1 |
| no | `exec` | `'0,1,2,3,4,5,6,7'` | **8** |
| no | `run` | `'0'` | 1 |

`ROCR_USE_SLURM_LOCALID=1` was present *inside* the container in arm 2 and binding still did
not happen, so the cause is the ENTRYPOINT not executing. The release notes describe the
refactor but carry **no caveat that `exec` bypasses an ENTRYPOINT**.

A user using `exec` with `--ntasks-per-node=8` and no launcher-side binding gets every rank
seeing all 8 GCDs. This is not support for §4.1 — on their `torchrun` pattern `SLURM_LOCALID` is
0 anyway and workers select by `LOCAL_RANK` — but it is a real defect on its own terms.

**Correction (2026-08-06).** An earlier version of this section called `exec` "the documented
pattern". It is not, for these images: all 20 launch commands in LUMI-AI-Guide @ `3705c3c` and all
three examples on the LAIF software-environment docs page use `run`. `exec` comes from generic LUMI
container docs, from this repo's harness, and from the reporter's scripts — plus one abbreviated
snippet in the guide's own `05-multi-gpu-and-node/README.md:297`, five lines below the same command
written with `run`.

The finding survives in a different form, because the binding is **doubly** opt-in. The image sets
no default for `ROCR_USE_SLURM_LOCALID` or `MAP_HIP_TO_ROCR_VISIBLE_DEVICES` (`singularity inspect
--environment`), and neither name appears in the release notes, the LUMI docs search index, or any
guide script. The guide's own `run_ddp_srun_4.sh` runs 8 tasks per node under `run` and sets
neither, so it gets no container-side binding either; it works because the training script binds
from `LOCAL_RANK`. The #6/#13 mechanism therefore reaches essentially nobody by default.

### 3. The container sets no cache variables — container gap

None of `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`,
`MIOPEN_USER_DB_PATH` are set in the image, so each falls back to its framework default. Measured
inside the image: `TRITON_CACHE_DIR` → `/users/$USER/.triton/cache`, `TORCH_EXTENSIONS_DIR` →
`/users/$USER/.cache/torch_extensions/py312_cpu`, `MIOPEN_USER_DB_PATH` → `~/.config/miopen/` —
all on `$HOME`, shared across the job and under a 20 GB quota — while
**`TORCHINDUCTOR_CACHE_DIR` → `/tmp/torchinductor_$USER`, which is node-local.**

That last one matters for how finding 4.7 is reached: our failing arm set it explicitly, so the
out-of-the-box default is safe for that specific failure. Users are steered off it instead. The
LUMI-AI-Guide repeats a cache block in 17 job scripts whose stated purpose is "to avoid saving to
home directory", sending MIOpen's kernel cache to node-local temp and `TORCH_HOME` to `/scratch`,
and covering none of the three torch/Triton JIT variables. Completing that pattern by pointing them
at `/scratch` is precisely the configuration that failed. (The same block also exports
`MIOPEN_USER_DB`, which MIOpen does not read — the variable is `MIOPEN_USER_DB_PATH`, verified
against the shipped `libMIOpen.so` — so the user perf DB stays on `$HOME` regardless.)

**Setting safe per-node defaults in the image would remove that entire failure class for every
user.** This is still the single highest-value change available.

### 4. Release artifacts exist but are undiscoverable — docs

The report asks for a changelog and a known-issues note. Both already ship with every release:
`*-release.md` with a full APT/PyPI diff, a recipe `compare/` link, a known-issues label query,
and `*-tests.md` with the automated test results. The reporter, who diagnosed the regression in
detail, never found any of it. The remedy is a pointer from the software-environment docs, not
new artifacts.

### 5. The regression was never reported upstream

The release's known-issues label carries exactly two issues — #27 (bitsandbytes int8
performance) and #24 (transformers/Gemma). **Neither concerns the fabric.** The reporter
diagnosed §4.1, worked around it by pinning April, and it never reached the maintainers. It is
still live in the default image and nobody upstream is tracking it. Filing it costs nothing.

Note #27 corresponds to the two FAILs in the shipped `*-tests.md`, so LAIF does track its own
failing tests — they are filed, just not release-blocking. The release *gate* is what needs
changing, not the tracking.

### 6. Omitting `device_id` hangs at scale — user-side gotcha

`init_process_group()` without `device_id` makes PyTorch warn `Guessing device ID based on
global rank. This can cause a hang if rank to GPU mapping is heterogeneous.` With
`ROCR_VISIBLE_DEVICES=$SLURM_LOCALID` every rank sees one device at index 0, so the guess is
wrong. At 128 ranks:

| | Without `device_id` | With |
| --- | --- | --- |
| First communicator | 13.87 s | 1.14 s |
| Second communicator | **hung indefinitely** | 0.29 s |

One communicator per rank survives the bad guess, so single-group jobs pass and only
multi-communicator (Megatron-like) jobs hang. Not a container defect, but a strong candidate
for user-reported hangs and free to document.

---

## Recommendations, in order of expected benefit

1. **Set per-node JIT cache defaults in the image** (finding 3 + 4.7). Removes a whole failure
   class that is invisible below ~64 ranks.
2. **Fix `fi_info`** (finding 1). One packaging fix; restores the primary fabric diagnostic.
3. **Publish a practical per-GCD memory ceiling** and the ~0.65 GiB-per-communicator overhead
   (4.2). Prevents a recurring class of confusing OOMs.
4. **Add an all-to-all test to the release suite** with ≥8 ranks/node across ≥2 nodes. The
   current suite has none, and its inter-node test runs one process per node, so it cannot
   reach endpoint-count-driven failures.
5. **Fix the guide's cache block** — `MIOPEN_USER_DB` → `MIOPEN_USER_DB_PATH`, and add the three
   torch/Triton JIT variables pointing at node-local storage (finding 3). Two lines, and it is the
   user-side half of recommendation 1, available without a container release.
6. **Document `ROCR_USE_SLURM_LOCALID` / `MAP_HIP_TO_ROCR_VISIBLE_DEVICES` and the `run`
   requirement** (finding 2). The binding feature currently activates for no documented workflow.
7. **Point users at the per-release artifacts** (finding 4) and **file the fabric regression**
   (finding 5).
8. **Document `device_id`** in the reference launch recipe (finding 6).
8. **Extend the known-bad node list** and record node lists with all timing data (4.5).

## Ask of the reporter

The node lists for the jobs behind §4.4 and §4.5. If those ran on `nid007xxx`, part of what
was attributed to the container may be node placement — which would be worth knowing before
any further container work.

---

## Reproducing any of this

```bash
export PROJECT_NAME=project_462000131
export EXCLUDE_NODES=...              # see docs/VALIDATION.md
./scripts/run_validation.sh           # phases 1-2
PHASE=3 ./scripts/run_validation.sh   # 4-node EP=32
NODES=16 LAIF_CACHE_MODE=lustre ./templates/cache_stress.sh <container.sif>
NODES=16 MAX_GROUPS=8 ./templates/comm_count.sh <container.sif>
```

Gate specs are in `manifests/gates/`. Every run records job ID, container digest, full
environment, node list and package versions. See [`VALIDATION.md`](VALIDATION.md).
