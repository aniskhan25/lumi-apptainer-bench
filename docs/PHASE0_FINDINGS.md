# Phase 0 findings — desk analysis, no GPU hours

Analysis of the `lumi-multitorch` latest container against the ten findings in the
`project_465003047` experience report, from artifacts already published with the release.
Nothing here required an allocation.

**Image under test**

```
/appl/local/laifs/containers/lumi-multitorch-latest.sif
  -> lumi-multitorch-u24r70f21m50t210-20260513_121430/
       lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
```

As of 2026-08-03 the `-latest` symlink still points at the May 13 build — the build the
report identifies as regressed. Every user who follows the documented `-latest` path gets
this image.

**Artifacts available in the release directory.** Worth stating because the roadmap
assumed they did not exist:

| Artifact | Contents |
| --- | --- |
| `*.sha256` | Published digests for all six variant images |
| `*.yaml` (92 KB) | Full SBOM (APT + PyPI) |
| `*.Containerfile` | Build recipe |
| `*-release.md` | Release notes with a complete package diff vs the April build |
| `*-tests.md` | Results of the automated release test suite |
| `*.json`, `*.log` | Image config and build log |

The release also ships six layered variants — `rocm`, `torch`, `libfabric`, `mpich`,
`full`, `plus`. The `full` variant is the one in general use. Because `libfabric` and
`mpich` are separate layers of the *same* release, they give a layer-isolation axis that
does not require reaching for an older image.

---

## F1 — The release test suite has no all-to-all test

### Symptom
Report §4.1: "A single-node NCCL probe PASSED on the regressed build while real
multi-node training failed. A user validating a new container with a small smoke test
would not catch this."

### What the shipped suite actually does
`lumi-multitorch-u24r70f21m50t210-20260513_121430-tests.md`, suite
`lumi-ai-factory/laifs-container-tests` @ `13a3c42`, 18 tests, 79 PASS / 2 FAIL:

```
accelerate-big-model-inference      pytorch-ddp-singlenode-srun
bitsandbytes-inference-int8  [FAIL] pytorch-ddp-singlenode-torchrun
bitsandbytes-inference              pytorch-ds-multinode-srun
osu-inter-node-gcd2gcd-bw   (x32)   pytorch-ds-multinode-torchrun
osu-intra-node-gcd2gcd-bw   (x32)   pytorch-ds-singlenode-srun
peft-alora-finetuning               pytorch-ds-singlenode-torchrun
pytorch-ddp-multinode-srun          pytorch-singlegpu
pytorch-ddp-multinode-torchrun      transformers-inference      (x2)
                                    vllm-bench-full-node-gpt-oss-120b
                                    vllm-bench-single-gpu-llama31-8b
```

The report's characterisation is too generous to itself: the suite *does* test multi-node,
via `pytorch-ddp-multinode-{srun,torchrun}` and `osu-inter-node-gcd2gcd-bw`. The gap is
narrower and more specific:

1. **No all-to-all test anywhere.** Every collective test is an allreduce (the DDP and
   DeepSpeed tests) or a point-to-point transfer (OSU). All-to-all is a different traffic
   pattern with different fabric resource demands, and it is the collective that carries
   MoE expert dispatch.
2. **`osu-inter-node-gcd2gcd-bw` runs one process per node.** It therefore opens a
   handful of CXI endpoints. An 8-rank-per-node all-to-all opens far more. `PTLTE_NOT_FOUND`
   concerns Portals table entries — a finite per-node resource — so a test that keeps the
   endpoint count structurally low cannot reach the failure regardless of how many nodes
   it spans.
3. **Failing tests do not block release.** This image shipped with two FAILs
   (`bitsandbytes-inference-int8`: generate throughput 1.19 vs 50.0 samples/s required,
   prefill 0.02 vs 2.5 required).

### Classification
Container validation gap. This one is confirmed from the shipped artifacts and needs no
reproduction.

### Resolution
`bench/tests/alltoall.py` plus the `alltoall_intra` / `alltoall_cross` gate specs. The
`actually_crossed_nodes` and `has_zero_token_peers` gates exist specifically to stop a
future run from satisfying the suite without exercising the pattern.

---

## F2 — Candidate root cause for the all-to-all regression, from the published diff

### Why no bisection is needed
The roadmap proposed bisecting image changes. `*-release.md` already publishes the
complete package diff between the April and May builds. Both images carry the identical
version tag `u24r70f21m50t210`, so this was **not** a ROCm, PyTorch, flash-attn or Triton
version bump.

Comms-stack changes, complete:

| Package | April | May |
| --- | --- | --- |
| `aws-ofi-nccl` | `1.18.0-git-c1b89cc` | `1.19.1-git-206c02c` |
| `mpich` | `5.0.0` | `5.0.1` |
| `torch` | `2.10.0+rocm7.0.lumi.aif.20260415153022` | `2.10.0+rocm7.0.lumi.aif.20260513142306` |

Everything else in the APT diff is Ubuntu security patching (`curl`, `libpng`,
`python3.12`, `linux-libc-dev`, …).

`aws-ofi-nccl` is the RCCL↔libfabric/CXI plugin — the component that maps RCCL
collectives onto Slingshot, and the layer that would surface a Portals error. It is the
prime candidate. The `torch` rebuild is secondary: same upstream version, but a different
LUMI AIF build, so the bundled RCCL may differ.

### Classification
Unresolved pending measurement — this is a candidate, not a confirmed cause. Recorded
here so Phase 2/3 can test the hypothesis directly rather than rediscovering it.

### Falsification test
If a cross-node all-to-all on the latest image passes cleanly at EP=16 and EP=32, the
`aws-ofi-nccl` bump is not implicated and this record closes as not-reproduced.

---

## F3 — GPU binding moved into an ENTRYPOINT that `apptainer exec` never runs

### Symptom
Candidate mechanism for a multi-node-only, scale-dependent fabric failure.

### Evidence
`*-release.md`, under Major Updates:

> **Replace SIF runscript with OCI entrypoint** — Certain runtime variables like
> `FI_HMEM_DISABLE_P2P` were previously set using a runscript that was included in the
> container's SIF definition file. This is now done using an entrypoint script included in
> the OCI image.

`lumi-multitorch-full-...Containerfile:256–268`:

```bash
RUN printf '#!/usr/bin/env bash\n\
if [ "${SLURM_NNODES}" = 1 ] && [ -z "${SLURM_GPUS_ON_NODE}" ]; then\n\
    export FI_HMEM_DISABLE_P2P=1\n\
fi\n\
if [ "${ROCR_USE_SLURM_LOCALID}" = 1 ] && [ -n "${SLURM_LOCALID}" ]; then\n\
    export ROCR_VISIBLE_DEVICES="$SLURM_LOCALID"\n\
fi\n\
if [ "${MAP_HIP_TO_ROCR_VISIBLE_DEVICES}" = 1 ] && [ -n "${ROCR_VISIBLE_DEVICES}" ]; then\n\
    export HIP_VISIBLE_DEVICES=...\n\
fi\n\
exec "$@"\n' > /opt/oci-entrypoint.sh && chmod +x /opt/oci-entrypoint.sh

ENTRYPOINT ["/opt/oci-entrypoint.sh"]
```

`apptainer exec` runs the given command directly and does not invoke an image's
ENTRYPOINT; only `apptainer run` does. The report's launch pattern is
`sbatch → srun → singularity exec → torchrun`, and the LUMI AI Guide pattern this repo
follows is the same.

### Reasoning
`FI_HMEM_DISABLE_P2P` is gated to `SLURM_NNODES = 1`, so losing it cannot explain a
multi-node failure — that strand is discarded. The **GPU-binding** exports are the
relevant ones. Under `exec` they never fire, so with `--ntasks-per-node=8` every rank
sees all 8 GCDs rather than one. That multiplies per-node device contexts and fabric
endpoints, which is the kind of pressure that exhausts a finite per-node CXI resource —
and it would only bite once the collective leaves the node, matching the reported
symptom.

### Layer isolation
- Container: ENTRYPOINT is the wrong mechanism for a `exec`-launched image, or the docs
  must state that `run` is required.
- Platform integration: interaction between Apptainer's exec semantics and OCI metadata.

### Classification
Container defect (candidate) + documentation defect. **Not confirmed** — no run yet.

### Falsification test
Phase 1, one node, ~one node-minute. Three arms, comparing
`tests.probe.visibility.torch_device_count`:

| Arm | `USE_ROCR_VISIBLE_DEVICES` | `APPTAINER_MODE` | Expected if hypothesis holds |
| --- | --- | --- | --- |
| launcher binds | 1 | `exec` | 1 device per rank |
| nobody binds | 0 | `exec` | **8** devices per rank |
| entrypoint binds | 0 | `run` | 1 device per rank |

If arm 2 reports 1 device, the hypothesis is dead. Requires `ROCR_USE_SLURM_LOCALID=1` in
arms 2 and 3 for the entrypoint to act at all.

Note this repo is incidentally immune: `templates/lumi_common.sh` sets
`ROCR_VISIBLE_DEVICES` from its own per-task wrapper. That is what makes the three-arm
comparison cheap to run.

---

## F4 — A changelog already exists; the problem is discoverability

### Symptom
Report §4.1 suggestion: "a short changelog or a known-issues note for the
`lumi-multitorch` container series would help".

### Finding
All three already exist and ship with the release:

- `*-release.md` — release notes plus the complete APT/PyPI diff against the prior build.
- A recipe-level diff link:
  `github.com/lumi-ai-factory/laifs-container-recipes/compare/<april-tag>...<may-tag>`
- A known-issues query:
  `github.com/lumi-ai-factory/laifs-container-recipes/issues?q=label%3A<release-tag>`
- `*-tests.md` — the automated test results for the release.

The user, who diagnosed the regression in detail and pinned the April build in response,
never found any of it.

### Classification
Documentation defect — discoverability, not absence. The remedy is a pointer from the
LUMI AI Factory software-environment docs to the per-release artifacts sitting next to
each image, not new artifacts.

---

## F5 — `expandable_segments` is confirmed unavailable

### Symptom
Report §4.2: OOM near 57 GiB of 63.98 GiB nameplate; runs log
`expandable_segments not supported on this platform` at
`/pytorch/c10/hip/HIPAllocatorConfig.h:40`, while the OOM message itself recommends
enabling that very option.

### Finding
Already pinned to a source location by the reporter. No investigation is warranted; this
is a confirmation-and-document item, downgraded from the roadmap's experimental
sub-workstream.

`env_detect.expandable_segments_supported()` captures the warning at startup, and the
probe raises it as a run warning so users see it before an OOM rather than after.

### Classification
Upstream framework limitation (ROCm/PyTorch) + documentation defect. The allocator cannot
compact fragmentation, so a per-GCD planning ceiling — the report plans against ~40 GB,
not 57 and certainly not 64 — should be published rather than left to be rediscovered.

---

## F6 — Pre-flight: this repo's own multi-node templates set a known-hanging variable

Found while preparing the branch, not from the report.

`templates/multi_ng_8rpn.sh` and `templates/allreduce_sweep.sh` both defaulted
`ENABLE_LUMI_HSN=1`, which makes `lumi_common.sh` export:

```bash
NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
NCCL_NET_GDR_LEVEL=PHB
```

`NCCL_NET_GDR_LEVEL=PHB` forces GDR even where the CXI/GDR path is broken and is recorded
in this project as a confirmed cause of indefinite allreduce hangs — job 19624583 hung
until it was removed, then completed in 24 s. `lumi_common.sh`'s own default was already
`0`; the two wrappers overrode it back on.

Any all-to-all test added on top of this would have hung before testing anything. Both
defaults are now `0`, with `1` retained as an explicit opt-in for the Phase 3 fabric
sweep.

### Classification
Harness defect, fixed on this branch.

---

## Items explicitly reclassified or deferred

| Report item | Disposition |
| --- | --- |
| §4.4 EP=32 all-to-all fails on the fabric | **In scope, Phase 3.** The primary target. |
| §4.5 45-min bootstrap at 1024 ranks | **Cannot reproduce within a 16-node ceiling.** Phases instrumented and measured at 1–16 nodes; the 1024-rank figure will be marked not-reproduced. Note node/CXI warmth is a confound the roadmap missed — this project has measured 10–15 min cold starts at *2* nodes and nodes that hang indefinitely, so rank count alone does not explain the number. |
| §4.7 Lustre JIT cache corruption | **In scope, Phase 4.** Needs ≥64 ranks. `LAIF_CACHE_MODE=lustre` reproduces, `tmp` validates the fix. |
| §4.8 `torch_dist` checkpoint hang | **Deferred** — needs Megatron. Also demoted from release-blocking: a hang was observed, corruption never was. |
| §4.9 QOS limit discoverability | **Out of scope** — Slurm policy, not container. |
| §2.1 packed-sequence attention leakage | **Deferred and reclassified.** No leakage was observed; the reporter identified that micro-batch > 1 *would* leak and pinned mbs=1. This is a property of the Megatron/flash-attn varlen path, not a container defect. |
| §6 Megatron-Core observations | **Deferred** — Phase 1 is pure PyTorch. The container bundles 0.15.0rc8 while the report ran a 0.16.1 `PYTHONPATH` overlay, so the shipped version was never the version exercised. |
| §4.10 383 vs 191.5 TF/s per GCD | **In scope, cheap.** MI250X bf16 peak is per OAM module (two GCDs); per GCD it is 191.5. This repo reports per-GCD TFLOPS, so any MFU derived from it is exposed to the same factor-of-two error. |
