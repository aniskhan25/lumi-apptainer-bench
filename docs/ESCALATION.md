# Escalation status: issues, comments, and what is left

Single index for everything raised or considered. Last revised 2026-08-21.

Provenance note: the `laifs-container-recipes` issue sweep was done 2026-08-06; guide-repo and
container state were re-checked against `main` / `20260807_115122` on 2026-08-20 and 2026-08-21.
Several drafts changed materially on re-checking — where that happened it is recorded in the draft
itself rather than quietly edited away.

---

## 1. Issues

| # | What | Target | Status |
| --- | --- | --- | --- |
| **E1** | [`fi_info`/`fi_pingpong` shadowed by Intel MPI shims](issues/E1-fi_info-broken.md) | `laifs-container-recipes` | **Not filed — candidate.** Low severity, one-line fix. Strengthened 2026-08-21: it also *skips* our `cxi_provider_visible` gate, so it blinds automated fabric checks |
| **E2** | [#6/#13 GPU-binding fix is doubly opt-in](issues/E2-entrypoint-not-run-under-exec.md) | `laifs-container-recipes` | **Not filed — optional.** Verified against guide `main`: no documented workflow is affected. Dead feature + docs gap |
| **E3** | [JIT cache defaults](issues/E3-jit-cache-defaults.md) | — | **Handled** via guide #112. Measurement record retained |
| **E5** | [torch 2.10.0 predates `pytorch#172144`](issues/E5-torch-predates-inductor-cache-fix.md) | `laifs-container-recipes` | **Parked.** Fix is in the 2.11 line, which needs ROCm ≥7.1; arrives free with the coming ROCm upgrade |
| **G2** | [Guide ch.5 uses `exec` in one snippet](issues/G2-guide-ch5-exec-snippet.md) | `LUMI-AI-Guide` | **Not filed — cosmetic.** No functional effect today |
| **T1** | [Add a multi-node all-to-all test](issues/T1-add-alltoall-test.md) | `laifs-container-tests` | **Not filed — unexamined.** Never re-checked against that repo's current state |
| ~~G3~~ | Guide ch.5 `device_id` | — | **Withdrawn.** The guide binds the device before its first collective, which is what matters; our measurement never isolated `device_id` from `set_device` |
| ~~E4~~ | mpi4py on Intel MPI | — | **Withdrawn.** Benign in the realistic import order; folded into E1 as a note |
| ~~G1~~ | `MIOPEN_USER_DB` typo | — | **Withdrawn.** Fixed upstream in guide #108 before we raised it |
| ~~U1~~ | Inductor cache reader | — | **Withdrawn.** Already fixed upstream; became E5 |

## 2. Comments

| Target | Status |
| --- | --- |
| **guide #111** — VRAM not all usable | **POSTED** 2026-08-19. No maintainer reply yet |
| **guide #112** — document more env vars | **Noted** by the guide maintainer |
| **recipes #39** — vLLM/compressed-tensors | **Shared** with the container maintainer |
| **recipes #20** — RCCL hangs with DDP | **POSTABLE — the only one left.** 0 comments, stale since 2026-03-27, labelled only for the `u24r64` generation. Rewritten to drop the `nid007xxx` placement theory, which a larger sample contradicted |
| **recipes #28** — multi-node init fails | **DO NOT POST.** Reporter's last comment reports 16 nodes stable |
| **recipes #30** — `NCCL_NET_GDR_LEVEL` | **DO NOT POST.** Last comment already states our conclusion |
| ~~guide #81~~ — MIOpen temp dir | **Dropped.** Resolved by #108 |

## 3. TODO

**Escalation**
- [ ] Post the **#20** comment — the only remaining one worth sending. Keeps both halves together:
      #20 is about hangs, so the uncaused 2-of-5 observation and the `device_id` mechanism both
      belong there rather than being split off
- [ ] Decide on **E1** (cheap, low severity) and **E2**/**G2** (optional)
- [ ] Re-check **T1** against `laifs-container-tests` before filing, given that five of nine drafts
      collapsed on contact with their target repos
- [ ] Unpark **E5** if the ROCm upgrade slips, or close it once 2.11 lands

**Gate suite**
- [ ] Run phases 4–5 (`jit_cache` at 64+ ranks, `comm_count`) against `20260807_115122` — 8–16 nodes,
      never run against this image
- [ ] Recalibrate the bandwidth floors from repetitions. EP=16 measured 64% above the frozen
      baseline on a single sample; the floors are provisional
- [ ] Close the gate-design gap: `triton_cache_off_lustre` / `inductor_cache_off_lustre` pass because
      *our* template sets `LAIF_CACHE_MODE=tmp`, so the suite does not detect the `$HOME` default.
      Needs a probe arm that deliberately clears the cache variables
- [ ] Make the `fi_info` probe distinguish *shadowed* from *absent* by also recording
      `/usr/bin/fi_info`, so the failure is diagnosable from the gate output alone

**Untested report findings**
- [ ] §4.3 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` from `torch.compile` — reachable without Megatron
- [ ] §4.6 two jobs on one Lustre dataset — reachable without Megatron
- [ ] §4.8 `torch_dist` checkpoint hang — needs Megatron
- [ ] §6 Megatron-Core observations — needs Megatron

**When the ROCm upgrade lands**
- [ ] Re-run the frozen gates across the ROCm 7.0→7.1+ / torch 2.10→2.11 boundary. This is the
      largest change the suite will have seen, our whole evidence base is on the old pair, and it
      should also confirm E5 resolved itself

---

## Detail: the container-repo findings

### E1. `fi_info` / `fi_pingpong` shadowed by Intel MPI shims — highest confidence, low severity

No existing issue. Two-line reproducer, no interpretation required — the most certain finding here,
though not the most important one. Nothing breaks; it costs a diagnostic.

Verified on a GPU node that the shadowed binary works and enumerates all 4 CXI NICs
(`cxi0`–`cxi3`), so the shadowing does deny users real information rather than hiding a tool that
would not have worked anyway.

```console
$ singularity exec <full-or-plus>.sif fi_info --version
/opt/venv/bin/fi_info: line 34: /opt/mpi/libfabric/bin/fi_info: No such file or directory
$ singularity exec <full-or-plus>.sif /usr/bin/fi_info --version
/usr/bin/fi_info: 2.1.0
```

**Correction to an earlier version of this section.** It said the tool was missing from all four
variants. Both halves were wrong. `/usr/bin/fi_info` is present and works; `libfabric`, `mpich` and
`torch` are all fine. Only `full` and `plus` fail, because `impi-rt` (pulled in transitively by
`oneccl`) installs a wrapper into `/opt/venv/bin`, which is first on `PATH`, and that wrapper execs
`$I_MPI_ROOT/opt/mpi/libfabric/bin/fi_info` with `I_MPI_ROOT` unset. So it is shadowing, not
absence. Confirmed unchanged on `20260807_115122`.

Fix is one line (`rm` the two shims) or dropping `oneccl`.

### E2. The #6 / #13 GPU-binding fix is doubly opt-in and undocumented

No existing issue, and it is best framed as a gap against the *intent* of closed issues
**#6** ("Copy ROCR_VISIBLE_DEVICES to HIP_VISIBLE_DEVICES at container startup") and **#13**
("Environment variable HIP_VISIBLE_DEVICES set incorrectly").

The `20260513` release moved runtime variables from a SIF runscript into an OCI `ENTRYPOINT`
(`Containerfile:256–268`). `apptainer exec` does not run an ENTRYPOINT; only `run` does.
Measured at one node, 8 ranks:

| Launcher binds | Mode | `ROCR_VISIBLE_DEVICES` seen | Devices/rank |
| --- | --- | --- | --- |
| yes | `exec` | `'0'` | 1 |
| no | `exec` | `'0,1,2,3,4,5,6,7'` | **8** |
| no | `run` | `'0'` | 1 |

`ROCR_USE_SLURM_LOCALID=1` and `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1` were confirmed present
*inside* the container in the middle row and the exports still did not happen, so the cause is
the ENTRYPOINT not executing rather than an unset variable.

**Correction to an earlier version of this section.** It said "the LUMI AI Guide launch pattern,
and the reporter's, is `exec`". The reporter's is; the guide's is not. Every runnable example for
these images uses `run` — all 20 launch commands in LUMI-AI-Guide @ `3705c3c`, and all three
examples on `docs.lumi-supercomputer.eu/laif/software/ai-environment/`. `exec` appears in generic
LUMI container docs, and once inside the guide itself (`05-multi-gpu-and-node/README.md:297`) as an
abbreviated snippet five lines below the same command written with `run`.

That narrows the verb half of the finding but the second condition then removes nearly everyone
who is left: `singularity inspect --environment` shows the image sets **no default** for
`ROCR_USE_SLURM_LOCALID` or `MAP_HIP_TO_ROCR_VISIBLE_DEVICES`, and neither name appears in the
release notes, the LUMI docs search index, or any guide script. The guide's own
`run_ddp_srun_4.sh` uses `run` with 8 tasks per node and sets neither, so no binding happens there
either — it works only because the training script binds from `LOCAL_RANK`.

So the mechanism reaches only users who use `run` *and* independently found two undocumented
variables. Ask is now primarily documentation: state that `run` is required, document the two
variables, and consider `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1` as an `ENV` default since `ENV`
applies under both verbs.

### E3. JIT cache defaults — HANDLED UPSTREAM, not filed

Treated as handled via `Lumi-supercomputer/LUMI-AI-Guide#112` (2026-08-21). Public state at that
date: #112 open, 0 comments, last updated 2026-08-06, and the three variables still absent from guide
`main`, whose cache block covers only `MIOPEN_*` and `TORCH_HOME`. So it is tracked upstream rather
than implemented.

Residual difference, noted and not pursued: #112 is guide-side documentation and reaches users who
follow the guide's scripts; our ask was an image-side `ENV` default, which would also reach derived
images and users who never read the guide. Modest, declinable, dropped.

The measurement record is worth keeping regardless, and lives in
[`E3-jit-cache-defaults.md`](issues/E3-jit-cache-defaults.md) and
[`PHASE4_RESULTS.md`](PHASE4_RESULTS.md):

| Cache | Failed ranks | Compile time | Result |
| --- | --- | --- | --- |
| shared, on Lustre | **1 of 128** + 80 recovered warnings across 9 ranks | 42–43 s | fail |
| per-node `/tmp` | none | 28.3–28.5 s | pass |

Plus the observed default: `~/.triton` accumulating 386 real `.hsaco`/`.llir` files over 2026-04 to
2026-07 with `TRITON_CACHE_DIR` unset, on a `$HOME` that is Lustre at 18G of 20G. The underlying race
is `pytorch#172144`, fixed in the 2.11 line but not in the shipped 2.10 — see E5.

---

## Detail: existing issues we checked against

Re-checked 2026-08-20. Statuses are in [section 2](#2-comments); this table records why.

| Our finding | Existing issue | Why |
| --- | --- | --- |
| `NCCL_NET_GDR_LEVEL=PHB` hangs collectives | **#30** (open, 8 comments) | Last comment already concludes the variable should not be needed. Our confirmation (job 19624583: hang → 24 s once removed) is redundant. Retained as the provenance for our templates defaulting it off. Useful *from* that thread: `FI_MR_CACHE_MONITOR=userfaultfd` reportedly fixed a hang-before-training on some tickets — untested by us |
| Intermittent RCCL hangs | **#20** (open, 0 comments) | Stale since 2026-03-27 and labelled only for `u24r64`, while it expected a fix in the ROCm 7 / torch 2.10 line we tested. The one comment still worth posting |
| Multi-node init deadlock | **#28** (open, 7 comments) | Reporter's last comment (2026-04-23) reports 16 nodes stable. Nothing to add |
| JIT cache variables | **guide #112** (open) | Covers the documentation half of E3 |
| MIOpen temp dir | **guide #108** (merged 2026-08-17) | Fixed `MIOPEN_USER_DB` → `MIOPEN_USER_DB_PATH` across 18 scripts and added per-node `srun mkdir -p`, before we raised it |
| VRAM not all usable | **guide #111** (open) | Our measurements posted 2026-08-19 |
| vLLM/compressed-tensors | **recipes #39** (open) | Shared with the maintainer: the constraint violation is in the shipped images, not only derived ones |
| Inductor cache temp-file race | **`pytorch#172144`** (merged 2026-01) | Already fixed upstream. Absent from the 2.10 line the images ship → became E5 |

## Do not file: report §4.1 itself

We could not reproduce it, so filing "inter-node all-to-all is broken" would be filing an
unverified claim. Exhausted without reproduction: group size 8/16/32, node span 1/2/4/16, four
concurrent disjoint meshes, 8 communicators per rank, uneven and zero-token dispatch — all on
the current image.

The right next step is not an issue but a request to the reporter for the node lists behind
§4.4 and §4.5, plus their `PTLTE_NOT_FOUND` logs. Both of our own hangs were on `nid007xxx`
while every pass was on `nid005xxx`/`nid006xxx`; if their runs landed there too, part of what
was attributed to the container is placement.

---

## File elsewhere — not the container repo

| # | Finding | Where | Note |
| --- | --- | --- | --- |
| U1 | Inductor's cache reader `open()`s other processes' in-flight `.{pid}.{tid}.tmp` files, failing at `torch/_inductor/codecache.py:1040` | **pytorch/pytorch** | Root cause of §4.7 and of E3. `iterate_over_candidates` lists the cache dir and opens every entry without skipping temp files, while `write_atomic` puts its temp file in that same dir. Genuinely upstream and not LUMI-specific. |
| T1 | Release test suite has no all-to-all test, and its inter-node test runs one process per node | **lumi-ai-factory/laifs-container-tests** | Separate repo. 18 tests, all collectives are allreduce or point-to-point, so endpoint-count-driven failures are unreachable by construction. |
| G1 | `MIOPEN_USER_DB` is not a MIOpen variable (it is `MIOPEN_USER_DB_PATH`), in 17 guide scripts; and the same cache block omits `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` / `TORCH_EXTENSIONS_DIR` | **Lumi-supercomputer/LUMI-AI-Guide** | Verified against the shipped `libMIOpen.so`: only `MIOPEN_USER_DB_PATH` exists, so the user perf DB stays on `$HOME`. Two-line fix; complements E3 from the user side. |
| D1 | `expandable_segments` is a no-op on this platform; publish a practical per-GCD memory ceiling and the ~0.65 GiB-per-communicator overhead | LUMI docs | Confirmed verbatim at `c10/hip/HIPAllocatorConfig.h:40`. |
| D2 | `init_process_group(device_id=…)` is required; omitting it hangs multi-communicator jobs | LUMI docs / reference recipe | Note #28's own reproducer already passes `device_id`, so maintainers know — but the LUMI-facing recipe should state it. |
| D3 | Per-release artifacts (`*-release.md`, `*-tests.md`, known-issues label) exist but are undiscoverable | LUMI docs | The reporter never found them. Pointer, not new artifacts. |
| D4 | MI250X bf16 peak 383 TF/s is per module; 191.5 per GCD | LUMI hardware docs | Factor-of-two MFU trap. |
| P1 | `nid007xxx` nodes hang during RCCL init | LUMI service desk | 2/2 of our hangs there; 3/3 passes elsewhere. |
| P2 | Per-partition QOS submit / node-minute limits undocumented | LUMI docs | Slurm policy, out of container scope. |

---
