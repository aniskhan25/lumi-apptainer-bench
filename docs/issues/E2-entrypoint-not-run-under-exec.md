**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** OCI `ENTRYPOINT` does not run under `apptainer exec`, so the #6 / #13 GPU-binding fix is inert for the documented launch pattern

---

## Summary

The `20260513_121430` release moved runtime variables from the SIF runscript into an OCI
`ENTRYPOINT`. `apptainer exec` does not execute an image's `ENTRYPOINT` — only `apptainer run`
does. Since the documented LUMI launch pattern is `srun … singularity exec … torchrun`, the
GPU-binding logic added for #6 and #13 never runs for those users.

Measured: with binding left to the container under `exec`, every rank sees all 8 GCDs instead
of one.

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

- #6 — "Copy ROCR_VISIBLE_DEVICES to HIP_VISIBLE_DEVICES at container startup" (closed)
- #13 — "Environment variable HIP_VISIBLE_DEVICES set incorrectly" (closed)

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
not happen — so the cause is the `ENTRYPOINT` not executing, not an unset variable. The third
row shows the same entrypoint working correctly when reached via `run`, and additionally setting
`HIP_VISIBLE_DEVICES`, which the launcher-side path does not.

## Impact

A user following the documented `exec` pattern with 8 ranks per node and relying on the
container to bind devices gets all 8 GCDs visible to every rank. A script that does not select a
device explicitly then places all 8 ranks on GCD 0 and leaves 7 idle.

Scope note, to avoid overstating this: with `torchrun` under a single `srun` task,
`SLURM_LOCALID` is 0 for that task, so the entrypoint would not bind devices even under `run`,
and torchrun workers select their device from `LOCAL_RANK` — where seeing 8 devices is normal
and harmless. The impact is specific to launch patterns that use `--ntasks-per-node=8` and
expect the container to do the binding.

## Request

Either:

1. move the binding logic somewhere `exec` honours — `ENV` directives apply under both `exec`
   and `run`, though the conditional-on-`SLURM_LOCALID` part cannot be expressed that way; or
2. state explicitly in the release notes and the software-environment documentation that
   `apptainer run` is required for the entrypoint logic to apply, and that `exec` users must
   bind devices themselves.

Option 2 is cheap and would be enough. The current state is a silent no-op for the documented
pattern, which is the difficult failure mode: nothing errors.

## Environment

- Image: `lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif`
- Digest: `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
- LUMI `dev-g`, 1 node, 8 ranks
