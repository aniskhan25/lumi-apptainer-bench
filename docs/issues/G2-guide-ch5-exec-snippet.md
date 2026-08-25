**Repo:** `Lumi-supercomputer/LUMI-AI-Guide`
**Title:** Chapter 5: one snippet uses `singularity exec` where the guide otherwise uses `run`

---

## What

`05-multi-gpu-and-node/README.md`, in the "CPU-GPU binding" section, illustrates the
`--cpu-bind=v` flag with:

```bash
srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS,v singularity exec ...
```

Five lines above, the same command is written with `run`:

```bash
srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity run $SIF bash -c "..."
```

Everywhere else the guide uses `run` for GPU workloads; 27 occurrences across `main`, with the
only other three `exec` uses being build helpers (`create_venv.sh`, `create_squashfs.sh`,
`install_venv.sh`) that run no GPU code.

## Why it is worth a one-word change

The two verbs are not equivalent for these images: `singularity run` executes the image's OCI
`ENTRYPOINT`, `exec` does not. The LAIF containers ship one (`/opt/oci-entrypoint.sh`), which
conditionally sets `FI_HMEM_DISABLE_P2P`, `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`.

**In fairness, this has no effect on the guide's examples today.** I checked, and for guide-shaped
jobs the entrypoint is a no-op under either verb:

- `FI_HMEM_DISABLE_P2P` requires `SLURM_NNODES=1` **and** `SLURM_GPUS_ON_NODE` empty. Guide scripts
  request GPUs, so `SLURM_GPUS_ON_NODE` is set, measured `SLURM_NNODES=[1]
  SLURM_GPUS_ON_NODE=[1] FI_HMEM_DISABLE_P2P=[<unset>]` under `run` with `--gpus-per-node=1`.
- The two GPU-binding branches require `ROCR_USE_SLURM_LOCALID=1` and
  `MAP_HIP_TO_ROCR_VISIBLE_DEVICES=1`, which appear nowhere in the guide (0 occurrences on `main`),
  and are not image defaults.

So this is a consistency fix, not a bug fix. The reason to make it anyway is that the guide is the
reference people copy from, the inconsistency is invisible to a reader, and if the entrypoint ever
gains a branch that does apply to guide-shaped jobs, the copied `exec` would silently stop getting
it.

## Suggested change

Either make the snippet consistent:

```diff
-srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS,v singularity exec ...
+srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS,v singularity run ...
```

or, if the abbreviation is deliberate, drop the verb entirely; `srun --cpu-bind=...,v <container
invocation>`; since the point of the line is the `,v` flag.

A sentence somewhere in chapter 5 noting that the guide uses `run` because it executes the image's
entrypoint, while `exec` bypasses it, would also be useful. That distinction is currently not stated
anywhere in the guide.

## Checked against

- `Lumi-supercomputer/LUMI-AI-Guide` `main`, 2026-08-20 (27 scripts, 11 chapter READMEs)
- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122`, entrypoint read from
  `/opt/oci-entrypoint.sh`
