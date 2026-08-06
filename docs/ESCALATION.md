# Escalation: what to file, where, and what is already covered

Checked against all 38 issues in `lumi-ai-factory/laifs-container-recipes` (full repo, not just
the release label) on 2026-08-06. Several findings turned out to be already reported, and one
existing issue probably explains report §4.4.

---

## File on the container repo — 3 new issues

### E1. `fi_info` is broken in every variant of the release — highest confidence

No existing issue. Unambiguous packaging defect, one-command reproducer, no interpretation
required.

```
$ singularity exec lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif fi_info --version
/opt/venv/bin/fi_info: line 34: /opt/mpi/libfabric/bin/fi_info: No such file or directory
rc=127
```

`fi_info` is on `PATH` as a wrapper that `exec`s `/opt/mpi/libfabric/bin/fi_info`, which is not
installed. Verified missing in **all four** variants of `20260513_121430`:

| Variant | Wrapper | Target |
| --- | --- | --- |
| `libfabric` | `/usr/bin/fi_info` | missing |
| `mpich` | `/usr/bin/fi_info` | missing |
| `torch` | `/usr/bin/fi_info` | missing |
| `full` | `/opt/venv/bin/fi_info` | missing |

The libfabric *library* is present and healthy (`libfabric.so.1.27.0`); only the tools are
absent from the path the wrappers expect. `/opt/cray` is not bind-mounted, so the host's tools
are not reachable either.

Why it matters beyond tidiness: `fi_info` is the first thing anyone runs when debugging a
Portals/CXI error. It looks installed and fails with a message naming neither libfabric nor the
real cause. Given issues #28, #30 and #20 are all fabric-adjacent, this is the diagnostic
users need most and cannot use.

### E2. `ENTRYPOINT` no longer takes effect under `apptainer exec` — regresses #6 and #13

No existing issue, and it is best framed as a regression against the *intent* of closed issues
**#6** ("Copy ROCR_VISIBLE_DEVICES to HIP_VISIBLE_DEVICES at container startup") and **#13**
("Environment variable HIP_VISIBLE_DEVICES set incorrectly").

The `20260513` release moved runtime variables from a SIF runscript into an OCI `ENTRYPOINT`
(`Containerfile:256–268`). `apptainer exec` does not run an ENTRYPOINT; only `run` does. The
LUMI AI Guide launch pattern, and the reporter's, is `exec`. Measured at one node, 8 ranks:

| Launcher binds | Mode | `ROCR_VISIBLE_DEVICES` seen | Devices/rank |
| --- | --- | --- | --- |
| yes | `exec` | `'0'` | 1 |
| no | `exec` | `'0,1,2,3,4,5,6,7'` | **8** |
| no | `run` | `'0'` | 1 |

`ROCR_USE_SLURM_LOCALID=1` and `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1` were confirmed present
*inside* the container in the middle row and the exports still did not happen, so the cause is
the ENTRYPOINT not executing rather than an unset variable.

So the fix delivered for #6 is inert for anyone launching with `exec`. Ask: either relocate the
logic somewhere `exec` honours, or state in the release notes that `run` is required for it.

### E3. Set safe JIT cache defaults in the image — hardening, with a reproduction

No existing issue. The image sets none of `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`TORCH_EXTENSIONS_DIR`, `MIOPEN_USER_DB_PATH`, so a user who does not set them gets the
framework default, typically `$HOME` — shared, and therefore unsafe at scale.

Reproduced on `20260513_121430`, 128 ranks / 16 nodes, 24 forced compilations per rank,
identical except cache location:

| Cache | Failed ranks | Compile time | Result |
| --- | --- | --- | --- |
| Lustre | **1 of 128** + 80 recovered warnings across 9 ranks | 42–43 s | fail |
| per-node `/tmp` | none | 28.3–28.5 s | pass |

Lustre is also ~1.5× slower to compile even when nothing fails. Setting per-node defaults keyed
by container ID would remove the failure class for every user. Note the root cause is upstream
(see U1) — but the container is where the mitigation belongs, since it controls the defaults.

---

## Already filed — comment, do not duplicate

| Our finding | Existing issue | Action |
| --- | --- | --- |
| `NCCL_NET_GDR_LEVEL=PHB` hangs multi-node collectives | **#30** (open) — "Setting NCCL_NET_GDR_LEVEL may cause jobs to hang" | Already known. Our independent confirmation (job 19624583: indefinite hang → 24 s once removed) could be added as a comment. |
| Intermittent RCCL hangs, node-correlated | **#20** (open) — "RCCL communications sometimes hang with PyTorch DDP" | Add the `nid007xxx` correlation as a comment. #20 expected a fix in the ROCm 7 / PyTorch 2.10 release, which *is* the image we tested, so evidence that hangs persist there is directly useful. |
| Multi-node init deadlock | **#28** (open) — "Multi-node `torch.distributed.init` fails." | See below — likely explains §4.4. |

### #28 probably explains report §4.4

#28 reports 2-node / 16-rank `init_process_group` hanging indefinitely on
`lumi-multitorch-torch-u24r70f21m50t210-**20260415**` — the **April** build — while the older
`20260319` build succeeds in ~35 s.

Report §4.4's EP=32 initialisation failure was also on the **April** build. So the reporter's
"32-rank all-to-all fails on the fabric" may be a manifestation of #28's April-build init
deadlock rather than anything specific to 32 ranks. That is consistent with our result that
EP=32 works on the May build, 2/2 at 128 ranks with four concurrent meshes.

Worth noting for the maintainers: **#28 and report §4.1 point in opposite directions.** #28
says April is broken and March works; §4.1 says April works and May is broken. Both cannot be
a simple monotonic regression, which suggests multi-node init stability varies per build *and*
per configuration — #30 (GDR level) and #20 (straggler rank) being two known configuration
causes. That reframing is more useful to file as a question on #28 than as a new bug.

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
| U1 | Inductor cache temp files named `.{pid}.{tid}.tmp` collide across nodes on a shared filesystem, failing at `torch/_inductor/codecache.py:1040` | **pytorch/pytorch** | Root cause of §4.7 and of E3. Three temp paths were each claimed by 2–4 distinct ranks. Genuinely upstream and not LUMI-specific. |
| T1 | Release test suite has no all-to-all test, and its inter-node test runs one process per node | **lumi-ai-factory/laifs-container-tests** | Separate repo. 18 tests, all collectives are allreduce or point-to-point, so endpoint-count-driven failures are unreachable by construction. |
| D1 | `expandable_segments` is a no-op on this platform; publish a practical per-GCD memory ceiling and the ~0.65 GiB-per-communicator overhead | LUMI docs | Confirmed verbatim at `c10/hip/HIPAllocatorConfig.h:40`. |
| D2 | `init_process_group(device_id=…)` is required; omitting it hangs multi-communicator jobs | LUMI docs / reference recipe | Note #28's own reproducer already passes `device_id`, so maintainers know — but the LUMI-facing recipe should state it. |
| D3 | Per-release artifacts (`*-release.md`, `*-tests.md`, known-issues label) exist but are undiscoverable | LUMI docs | The reporter never found them. Pointer, not new artifacts. |
| D4 | MI250X bf16 peak 383 TF/s is per module; 191.5 per GCD | LUMI hardware docs | Factor-of-two MFU trap. |
| P1 | `nid007xxx` nodes hang during RCCL init | LUMI service desk | 2/2 of our hangs there; 3/3 passes elsewhere. |
| P2 | Per-partition QOS submit / node-minute limits undocumented | LUMI docs | Slurm policy, out of container scope. |

---

## Suggested order

1. **E1** (`fi_info`) — trivial to verify, trivial to fix, unblocks everyone else's diagnosis.
2. **E3** (cache defaults) — largest user impact; carries its own reproduction.
3. **U1** (PyTorch upstream) — the root cause behind E3; file so the mitigation can eventually
   be dropped.
4. **E2** (ENTRYPOINT under `exec`) — needs a maintainer decision on relocate-vs-document.
5. **T1** (add an all-to-all test) — closes the validation gap the report actually identified.
6. Comments on **#20** and **#30**, and the reframing question on **#28**.
7. Documentation items D1–D4, then P1/P2 to the service desk.
