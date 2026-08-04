# Phase 1 results — one node, capability probe

**Image:** `lumi-multitorch-latest.sif` → `lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif`
**Digest:** `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
**Partition:** `dev-g`, 1 node, 8 ranks (except where noted)
**Date:** 2026-08-04
**Jobs:** 20671589 (aborted, see H4), then arms A/B/C and one corrected re-run

---

## Environment as actually loaded

Every value below is read from inside the running job, not from the image manifest.

| Component | Version | Matches report? |
| --- | --- | --- |
| Python | 3.12.3 | — |
| PyTorch | `2.10.0+rocm7.0_lumi_aif_20260513142306` | yes (`2.10.0+rocm7.0`) |
| HIP runtime | `7.0.51831` | yes, exactly |
| RCCL | `2.26.6` | not stated in report |
| Triton | `3.6.0` | yes |
| flash-attn | `2.8.4+lumi_aif_gfx90a_bbe25ba` | yes |
| apex | `1.10.0+lumi_aif_gfx90a_73423b4` | — |
| Megatron-Core | `0.15.0rc8+lumi_aif_c333868` | **yes** — confirms the container ships 0.15.0rc8 |
| transformer-engine | absent | — |
| Device | AMD Instinct MI250X, `gfx90a:sramecc+:xnack-`, 63.98 GiB | yes, exactly |
| aws-ofi-nccl | `1.19.1-git-206c02c` | confirms the May build's plugin |
| libfabric (library) | `1.27.0` at `/usr/lib/x86_64-linux-gnu/` | — |
| `RLIMIT_NOFILE` | 131072 soft and hard | — |

The `torch` build string carries the May 13 timestamp, confirming the rebuild noted in the
release diff. Megatron-Core `0.15.0rc8` confirms report §6: the shipped version was never
the version the reporter exercised, since they overlaid 0.16.1 on `PYTHONPATH`.

---

## H1 — CONFIRMED: the container's GPU binding is inert under `apptainer exec`

The hypothesis from `docs/PHASE0_FINDINGS.md` F3, tested in three arms.

| Arm | `USE_ROCR_VISIBLE_DEVICES` | `APPTAINER_MODE` | `ROCR_VISIBLE_DEVICES` seen | `HIP_VISIBLE_DEVICES` | devices/rank |
| --- | --- | --- | --- | --- | --- |
| A | 1 (launcher binds) | `exec` | `'0'` | `''` | **1** |
| B | 0 (nobody binds) | `exec` | `'0,1,2,3,4,5,6,7'` | `''` | **8** |
| C | 0 (nobody binds) | `run` | `'0'` | `'0'` | **1** |

Arm B is the decisive one. `ROCR_USE_SLURM_LOCALID=1` and
`MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1` were both confirmed present *inside* the container in
that arm — the probe's environment dump shows them — and binding still did not happen. So
the cause is the ENTRYPOINT never being executed, not an unset variable. Arm C shows the
same entrypoint working correctly when reached via `run`, and additionally setting
`HIP_VISIBLE_DEVICES`, which arm A does not.

Arm A's per-rank records confirm the launcher-side path binds all 8 ranks correctly:

```
rank0000 localid=0 rocr='0' count=1     rank0004 localid=4 rocr='4' count=1
rank0001 localid=1 rocr='1' count=1     rank0005 localid=5 rocr='5' count=1
rank0002 localid=2 rocr='2' count=1     rank0006 localid=6 rocr='6' count=1
rank0003 localid=3 rocr='3' count=1     rank0007 localid=7 rocr='7' count=1
```

### What this does and does not establish

**Established:** the container documents and implements a GPU-binding mechanism that is
silently inert under `exec`, which is the launch verb in the LUMI AI Guide pattern, in this
repo, and in the reporter's own scripts. The release notes describe the runscript→ENTRYPOINT
move but carry no caveat that `exec` bypasses an ENTRYPOINT.

**Consequence for a user:** with `srun --ntasks-per-node=8` + `exec` and no launcher-side
binding, every rank sees all 8 GCDs. A script that does not explicitly select a device
lands all 8 ranks on GCD 0, leaving 7 idle. This repo is insulated by accident —
`lumi_common.sh` sets `ROCR_VISIBLE_DEVICES` itself, and `distributed.local_cuda_index()`
falls back to `LOCAL_RANK % device_count` — but neither protection comes from the container.

**Not established — correcting my Phase 0 framing.** I wrote that the lost binding
"multiplies per-node fabric endpoints ~8×" and was therefore a candidate mechanism for
`PTLTE_NOT_FOUND`. That claim is weaker than I presented it. In the reporter's actual
pattern (`srun --ntasks-per-node=1` → `exec` → `torchrun --nproc_per_node=8`) `SLURM_LOCALID`
is 0 for the single task, so the entrypoint would not have bound devices even under `run`;
torchrun workers select their device from `LOCAL_RANK` instead, and seeing 8 devices is
normal and harmless there. RCCL allocates communicator resources for the device a rank
actually selects, not for every visible one. So H1 is a confirmed container defect on its
own terms, but it is **not** currently supported as the cause of the all-to-all regression.
That question is still open and belongs to Phase 2/3.

### Classification
Container defect + documentation defect. Confirmed, 3/3 arms reproduced as predicted.

---

## H2 — CONFIRMED: `expandable_segments` is unsupported, verbatim

Report §4.2 reproduced exactly on the latest container:

```
UserWarning: expandable_segments not supported on this platform
  (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:40.)
```

Obtained by running a subprocess with `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
actually set and allocating a 64 MiB tensor. The allocation itself succeeds — the option is
accepted and silently ignored, which is why the OOM message keeps recommending it.

This matters because it is the only lever that would let the allocator compact
fragmentation, and the reporter measured reserved memory running far above allocated
(34.8 GiB allocated vs 52.9 GiB reserved) with reserved being what determines the failure.
A practical per-GCD planning ceiling should be published; the reporter now plans against
~40 GB, not the 63.98 GiB nameplate.

**Method note.** My first implementation reported `supported: True`. It called the
deprecated `torch.cuda.memory._set_allocator_settings()` in-process, which never reaches
the config parser that emits the warning, so it produced a false negative. Fixed in
`13650c9`. Worth recording because the same mistake would make a release gate certify a
broken configuration as healthy.

### Classification
Upstream framework limitation, confirmed. Documentation defect for LAIF.

---

## H3 — NEW FINDING: `fi_info` is broken in every variant of the release

Not from the report. Found while resolving why the probe's CXI check failed.

```
$ singularity exec <full>.sif fi_info --version
/opt/venv/bin/fi_info: line 34: /opt/mpi/libfabric/bin/fi_info: No such file or directory
rc=127
```

`fi_info` is on `PATH` as a wrapper script that `exec`s
`/opt/mpi/libfabric/bin/fi_info`, and that target is not installed. Checked across all four
variants of the 2026-05-13 release:

| Variant | Wrapper location | Target present |
| --- | --- | --- |
| `libfabric` | `/usr/bin/fi_info` | **missing** |
| `mpich` | `/usr/bin/fi_info` | **missing** |
| `torch` | `/usr/bin/fi_info` | **missing** |
| `full` | `/opt/venv/bin/fi_info` | **missing** |

The libfabric *library* is present and healthy (`libfabric.so.1.27.0`); only the
command-line tools are missing from the path their wrappers expect. `/opt/cray` is not
bind-mounted, so the host's Cray libfabric and `libcxi` are not visible either — by design
for a self-contained LAIF image, but it means the container's own tooling is the only way
in.

**Why this compounds the reported problems.** `fi_info` is the first tool anyone reaches
for when debugging a Portals/CXI error such as `PTLTE_NOT_FOUND`. It appears installed,
then fails with a message about a missing file that names neither libfabric nor the real
problem. The reporter had no way to enumerate fabric providers from inside the container
while diagnosing §4.1 and §4.4.

Cheap to fix and easy to gate — the `fi_info_runs` gate now covers it.

### Classification
Container defect (packaging), confirmed across all variants. Reproduction 4/4.

---

## H4 — Harness finding: LUMI CPU bind masks require an exclusive full node

Job 20671589 aborted immediately:

```
srun: error: CPU binding outside of job step allocation,
      allocated CPUs are: 0x001E1E1E1E1E1E1E001E1E1E1E1E1E1E
srun: error: Unable to satisfy cpu bind request
```

The canonical LUMI GPU/CPU bind mask list assumes 7 usable cores per GPU group (`0xfe`);
the `dev-g` allocation granted 4 (`0x1E`). The masks are correct for an exclusive full node
and are rejected otherwise.

Fixed by defaulting `ENABLE_LUMI_CPU_MASKS=0` in `templates/probe.sh` (commit `0df2396`):
a probe measures device visibility, not throughput, so NUMA placement is irrelevant to it.
The performance templates keep masks on, and those should run on `standard-g` with a full
node.

---

## Incidental observations

- **Apex extension load costs ~66 s per rank on a cold node.** All 8 ranks reported
  `Time to load amp_C op: ~66 s` simultaneously. With `LAIF_CACHE_MODE=tmp` the cache is
  node-local, so this is paid again on every new node — relevant background to report §4.5
  on long startup, and worth measuring properly in Phase 4 alongside the cache work.
- **`/tmp` is `tmpfs` and node-local**, so the per-node cache default is sound.
- **All required mounts present:** `/scratch`, `/projappl`, `/project`, `/flash`, `/appl`,
  `/pfs`, `/tmp`. Report §4.10's `realpath`/`/projappl` concern does not reproduce with
  bindings module 1.0.1, which binds `/pfs` explicitly.
- **`nid005004` served one of these jobs** and is on this project's known-bad list for
  RCCL init hangs. It was harmless here because the probe never initialises a process
  group, but Phase 2 onward must set `EXCLUDE_NODES`.
- **`TORCH_HOME` is on Lustre** (set by `lumi_common.sh`). Flagged by the probe. Unlike the
  JIT caches this is a model-download cache where sharing is desirable and concurrent
  writes are rare, so it is being left as-is rather than moved.

---

## Gate status

| Gate | Result |
| --- | --- |
| `torch_present` | pass |
| `device_arch_is_gfx90a` | pass |
| `aws_ofi_nccl_recorded` | pass |
| `tmp_is_node_local` | pass |
| `triton_cache_off_lustre` | pass |
| `inductor_cache_off_lustre` | pass |
| `scratch_visible` / `projappl_visible` | pass |
| `device_count_matches_binding` | pass in arm A / **fail in arm B** (8) — as designed |
| `fi_info_runs` | **fail** (H3) |
| `cxi_provider_visible` | skipped — blocked by H3 |

The suite is doing its job: it fails on a real defect, and the one arm that should fail
does.

---

## Next

Phase 2, two nodes: all-to-all correctness and bandwidth at EP=8 (intra-node XGMI control)
then EP=16 (crosses Slingshot). `EXCLUDE_NODES` mandatory from here. The open question H1
did not settle — whether the all-to-all regression reproduces on the latest container at
all — is what Phase 2 and 3 exist to answer.
