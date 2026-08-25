**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** The #6 / #13 GPU-binding fix is doubly opt-in and undocumented: inert under `apptainer exec`, and its two enabling variables are set nowhere

---

## Summary

The `20260513_121430` release moved runtime variables from the SIF runscript into an OCI
`ENTRYPOINT`. The GPU-binding logic added for #6 and #13 lives there, and it only takes effect if
**both** of the following hold:

1. the image is launched with `apptainer run`, because `exec` does not execute an `ENTRYPOINT`; and
2. the user has exported `ROCR_USE_SLURM_LOCALID=1` (and `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1`).

Neither condition is documented, and the image ships no default for either variable. The result is
that the delivered fix does not apply to any launch pattern in the official documentation, and
silently does nothing for users who switch to `exec`.

Measured under `exec` with binding left to the container: every rank sees all 8 GCDs instead of one.

## Background

`lumi-multitorch-full-…-20260513_121430.Containerfile`, lines 256–268:

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

The second and third conditionals are the mechanism delivered for:

- #6; "Copy ROCR_VISIBLE_DEVICES to HIP_VISIBLE_DEVICES at container startup" (closed)
- #13; "Environment variable HIP_VISIBLE_DEVICES set incorrectly" (closed)

The release notes describe the runscript → entrypoint move and cite `FI_HMEM_DISABLE_P2P` as an
example, but do not mention that `exec` bypasses an `ENTRYPOINT`, nor that GPU-binding variables
moved with it.

## Reproducer

One node, `--ntasks-per-node=8 --gpus-per-node=8`, with
`ROCR_USE_SLURM_LOCALID=1` and `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1` exported before `srun`, and
no launcher-side binding. Compare `ROCR_VISIBLE_DEVICES` and `torch.cuda.device_count()` per
rank:

| Launcher-side binding | Mode | `ROCR_VISIBLE_DEVICES` seen | `HIP_VISIBLE_DEVICES` | devices/rank |
| --- | --- | --- | --- | --- |
| yes (`export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`) | `exec` | `'0'` | `''` | 1 |
| none | `exec` | `'0,1,2,3,4,5,6,7'` | `''` | **8** |
| none | `run` | `'0'` | `'0'` | 1 |

Both `ROCR_USE_SLURM_LOCALID=1` and `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1` were confirmed present
*inside* the container in the middle row (dumped from `os.environ`), and the exports still did
not happen, so the cause is the `ENTRYPOINT` not executing, not an unset variable. The third
row shows the same entrypoint working correctly when reached via `run`, and additionally setting
`HIP_VISIBLE_DEVICES`, which the launcher-side path does not.

## Verified against LUMI-AI-Guide `main`

Checked on `main` (not a pinned commit), 27 scripts and 11 chapter READMEs:

| | count |
| --- | --- |
| `singularity run` | 27 |
| `singularity exec` | 3 all build helpers (`create_squashfs.sh`, `create_venv.sh`, `install_venv.sh`), no GPU workload |
| `ROCR_USE_SLURM_LOCALID` | **0** |
| `MAP_HIP_TO_ROCR_VISIBLE_DEVICES` | **0** |
| `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` | **0** |

So every GPU workload in the guide uses `run`, and the guide sets neither opt-in variable anywhere
in scripts or in prose. The `docs.lumi-supercomputer.eu` LAIF page likewise uses `run` throughout.
No guide issue covers this; the three binding-related issues (#95, #41, #46) are all about *CPU*
bindings and are closed.

**And the guide does not need this feature.** Its 8-task-per-node scripts bind in the application
instead:

```bash
srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity run $SIF bash -c \
  "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python ddp_visiontransformer.py"
```

The training script selects its device from `LOCAL_RANK`, so the container's binding logic
contributes nothing and its absence costs nothing. Guide followers are unaffected.

## Severity, stated honestly

This is **not a bug for anyone following the documented path**. It is a dead feature plus a
documentation gap:

- The mechanism delivered for #6 and #13 requires `run` **and** two variables that appear in no
  documentation, no release note, and no example. As shipped it activates for no documented
  workflow.
- Users outside the guide; the experience report's author, and our own harness; do use `exec`, and
  someone who assumes the container binds devices gets all 8 GCDs per rank.
- One small trap remains in the guide: `05-multi-gpu-and-node/README.md:297` shows
  `srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS,v singularity exec ...` as an abbreviated illustration of
  the `--cpu-bind=v` flag, five lines below the same command written with `run`. A reader copying that
  line switches launch verb without being told it changes container behaviour. That is a one-word fix
  in the guide, better raised there than here.

Of the container-side findings this is the weakest, and it may not warrant its own issue, the ask
below could equally be a note appended to whichever release documents the entrypoint.

## Note on the torchrun pattern

With `torchrun` under a single `srun` task, `SLURM_LOCALID` is 0 for that task, so the entrypoint
would not bind devices even under `run` with both variables set, and torchrun workers select their
device from `LOCAL_RANK`, where seeing 8 devices is normal and harmless. So the only pattern where
the feature would do anything is `--ntasks-per-node=8` with a script that does not select a device
itself.

## Request

Documentation, primarily:

1. State in the release notes and in the software-environment documentation that `apptainer run`
   is required for the entrypoint logic to apply, and that `exec` bypasses it entirely.
2. Document `ROCR_USE_SLURM_LOCALID` and `MAP_HIP_TO_ROCR_VISIBLE_DEVICES`; what they do, and
   that they must be exported by the user. Right now a feature exists that no documented workflow
   activates.
3. Consider whether `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1` should be an `ENV` default. `ENV`
   directives apply under both `exec` and `run`, so that half would then work for everyone. The
   `SLURM_LOCALID` conditional cannot be expressed as `ENV` and would still need `run`.

The current state is a silent no-op, which is the difficult failure mode: nothing errors.

## Environment

- Image: `lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif`
- Digest: `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
- LUMI `dev-g`, 1 node, 8 ranks
- Verb/variable survey: LUMI-AI-Guide `main` (27 scripts, 11 chapter READMEs) and the
  `docs.lumi-supercomputer.eu` search index, both checked 2026-08-20. Entrypoint conditionals and
  the opt-in variables are unchanged in `20260807_115122`.
