# Escalation: what to file, where, and what is already covered

Checked against all 38 issues in `lumi-ai-factory/laifs-container-recipes` (full repo, not just
the release label) on 2026-08-06. Several findings turned out to be already reported, and one
existing issue probably explains report §4.4.

---

## Drafts

Ready to paste, in [`docs/issues/`](issues/). Nothing has been filed — these are drafts only.

| Draft | Target repo |
| --- | --- |
| [`E1-fi_info-broken.md`](issues/E1-fi_info-broken.md) | `laifs-container-recipes` |
| [`E2-entrypoint-not-run-under-exec.md`](issues/E2-entrypoint-not-run-under-exec.md) | `laifs-container-recipes` |
| [`E3-jit-cache-defaults.md`](issues/E3-jit-cache-defaults.md) | `laifs-container-recipes` |
| [`E5-torch-predates-inductor-cache-fix.md`](issues/E5-torch-predates-inductor-cache-fix.md) | `laifs-container-recipes` |
| [`T1-add-alltoall-test.md`](issues/T1-add-alltoall-test.md) | `laifs-container-tests` |
| [`G2-guide-ch5-exec-snippet.md`](issues/G2-guide-ch5-exec-snippet.md) | `Lumi-supercomputer/LUMI-AI-Guide` |
| [`comments-on-existing-issues.md`](issues/comments-on-existing-issues.md) | comments on recipes #20, #28, #30, #39 and guide #81, #112 |

---

## File on the container repo — 3 new issues

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

### E3. Set safe JIT cache defaults in the image — hardening, with a reproduction

No existing issue. The image sets none of `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`TORCH_EXTENSIONS_DIR`, `MIOPEN_USER_DB_PATH`, so each falls back to its framework default.

**Correction to an earlier version of this section.** It said the defaults are "typically
`$HOME`". Measured inside the image, three of four are (`TRITON_CACHE_DIR` →
`/users/$USER/.triton/cache`, `TORCH_EXTENSIONS_DIR` → `/users/$USER/.cache/torch_extensions/…`,
`MIOPEN_USER_DB_PATH` → `~/.config/miopen/`) but **`TORCHINDUCTOR_CACHE_DIR` defaults to
`/tmp/torchinductor_$USER`, which is node-local**. Our failing arm set that variable explicitly, so
a user who changes nothing does not hit it.

They are steered into it instead. The LUMI-AI-Guide sets a cache block in 17 job scripts whose
stated purpose is "to avoid saving to home directory", redirecting MIOpen's kernel cache to a
node-local temp dir and `TORCH_HOME` to `/scratch` — while covering none of the three torch/Triton
JIT variables. Completing that pattern by pointing the missing three at `/scratch`, as the block
models for `TORCH_HOME`, builds exactly the failing configuration. The two the guide leaves alone
default to `$HOME`, shared across the job's nodes and under a 20 GB quota. Both outcomes are bad
and the documentation gives no basis for choosing.

Reproduced on `20260513_121430`, 128 ranks / 16 nodes, 24 forced compilations per rank,
identical except cache location:

| Cache | Failed ranks | Compile time | Result |
| --- | --- | --- | --- |
| Lustre | **1 of 128** + 80 recovered warnings across 9 ranks | 42–43 s | fail |
| per-node `/tmp` | none | 28.3–28.5 s | pass |

Lustre is also ~1.5× slower to compile even when nothing fails. Setting per-node defaults keyed
by container ID would remove the failure class for every user. The root cause is upstream — the
Inductor cache reader opens other processes' in-flight `.tmp` files (see U1) — but the container
is where the mitigation belongs, since it controls the defaults.

---

## Already filed — comment, do not duplicate

| Our finding | Existing issue | Action |
| --- | --- | --- |
| `NCCL_NET_GDR_LEVEL=PHB` hangs multi-node collectives | **#30** (open) — "Setting NCCL_NET_GDR_LEVEL may cause jobs to hang" | Already known. Our independent confirmation (job 19624583: indefinite hang → 24 s once removed) could be added as a comment. |
| Intermittent RCCL hangs, node-correlated | **#20** (open) — "RCCL communications sometimes hang with PyTorch DDP" | Add the `nid007xxx` correlation as a comment. #20 expected a fix in the ROCm 7 / PyTorch 2.10 release, which *is* the image we tested, so evidence that hangs persist there is directly useful. |

---

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

## Suggested order

1. **E3** (cache defaults) — largest user impact; carries its own reproduction.
2. **E1** (`fi_info`) — lowest severity of the three, but a one-line fix and zero risk, and it
   restores the first tool anyone reaches for when debugging the fabric.
3. **E5** (torch 2.10.0 predates `pytorch#172144`) — do **not** file upstream, it is already fixed
   there; the ask is that the LUMI torch build picks it up. Strongest of the container findings:
   a named upstream commit, a two-line diff, and our 128-rank reproduction as evidence.
4. **Guide #112 and #81 comments** — the guide-side half of E3, already tracked upstream. #112 asks
   for exactly the three JIT variables; the useful addition is that two of them want node-local
   storage rather than `/scratch`. #81 is closed but its implementation used `MIOPEN_USER_DB`, which
   MIOpen does not read.
5. **E2** (ENTRYPOINT under `exec`) — weakest of the three, and optional. Verified against
   LUMI-AI-Guide `main`: every GPU workload uses `run`, the two opt-in variables appear nowhere, and
   the guide binds from `LOCAL_RANK` in the application, so guide followers are unaffected. What
   remains is a dead feature and a documentation gap. Could be a note on the release rather than an
   issue.
6. **T1** (add an all-to-all test) — closes the validation gap the report actually identified.
7. Comments on **#20** and **#30**, and the reframing question on **#28**.
8. **G2** (guide chapter 5 `exec` snippet) — cosmetic, no functional effect today; file only if a
   one-word consistency fix is welcome.
9. Documentation items D1–D4, then P1/P2 to the service desk.
