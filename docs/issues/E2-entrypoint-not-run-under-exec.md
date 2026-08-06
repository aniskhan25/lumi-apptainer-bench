**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** The #6 / #13 GPU-binding fix is doubly opt-in and undocumented: inert under `apptainer exec`, and its two enabling variables are set nowhere

---

## Summary

The `20260513_121430` release moved runtime variables from the SIF runscript into an OCI
`ENTRYPOINT`. The GPU-binding logic added for #6 and #13 lives there, and it only takes effect if
**both** of the following hold:

1. the image is launched with `apptainer run` — `exec` does not execute an `ENTRYPOINT`; and
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

## Scope: which launch verb is documented, and whether the variables are ever set

I originally assumed `exec` was the documented pattern. It is not, for the LAIF images, so the
first condition affects fewer users than I thought — but the second condition then removes almost
everyone who is left.

**Launch verb.** Every runnable example for these images uses `run`:

- `docs.lumi-supercomputer.eu/laif/software/ai-environment/` — `singularity run $SIF …` in all
  three examples.
- LUMI-AI-Guide @ `3705c3c` — **20 of 20** launch commands across all ten chapters use
  `singularity run`.

`exec` appears in general LUMI container documentation
(`runjobs/scheduled-jobs/container-jobs/`, `runjobs/scheduled-jobs/python/`), which is not
LAIF-specific, and once in the AI Guide itself: `05-multi-gpu-and-node/README.md` line 297 shows

```bash
srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS,v singularity exec ...
```

as an abbreviated illustration of the `--cpu-bind=v` flag, five lines below the same command
written with `run`. That inconsistency is the point: the verb is being treated as interchangeable
in the guide's own prose, and nothing tells a reader it changes the container's runtime behaviour.

**The opt-in variables.** `singularity inspect --environment` on the `full` image sets neither
`ROCR_USE_SLURM_LOCALID` nor `MAP_HIP_TO_ROCR_VISIBLE_DEVICES` (nor `ROCR_VISIBLE_DEVICES` or
`HIP_VISIBLE_DEVICES`). Neither name appears anywhere in the release notes, in the LUMI
documentation search index, or in any LUMI-AI-Guide script.

Concretely: the guide's own `05-multi-gpu-and-node/run_ddp_srun_4.sh` runs 8 tasks per node under
`singularity run` and sets neither variable, so the binding branch does not fire there either.
That example works only because `ddp_visiontransformer.py` selects its device from `LOCAL_RANK` —
the container's binding logic contributes nothing.

So the mechanism delivered for #6 and #13 currently reaches only users who both use `run` and
independently discovered two undocumented variable names.

## Impact

Two distinct groups:

- **`exec` users who rely on the container to bind devices** get all 8 GCDs visible to every rank.
  A script that does not select a device explicitly then places all 8 ranks on GCD 0 and leaves 7
  idle. This is the pattern the user report we were investigating used.
- **Everyone else** gets no binding either, because the opt-in variables are unset — they are
  simply unaffected, because the documented examples bind from `LOCAL_RANK` in the application.

Scope note, to avoid overstating this: with `torchrun` under a single `srun` task, `SLURM_LOCALID`
is 0 for that task, so the entrypoint would not bind devices even under `run` with the variables
set, and torchrun workers select their device from `LOCAL_RANK` — where seeing 8 devices is normal
and harmless. The impact is specific to launch patterns that use `--ntasks-per-node=8` and expect
the container to do the binding.

## Request

Documentation, primarily:

1. State in the release notes and in the software-environment documentation that `apptainer run`
   is required for the entrypoint logic to apply, and that `exec` bypasses it entirely.
2. Document `ROCR_USE_SLURM_LOCALID` and `MAP_HIP_TO_ROCR_VISIBLE_DEVICES` — what they do, and
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
- Verb/variable survey: LUMI-AI-Guide @ `3705c3c9a3ec0fd7f9e73980ab3cd41d29170c48`,
  `docs.lumi-supercomputer.eu` search index as of 2026-08-06
