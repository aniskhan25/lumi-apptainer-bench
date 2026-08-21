**Repo:** `Lumi-supercomputer/LUMI-AI-Guide`
**Title:** Chapter 5: pass `device_id` to `init_process_group` — the current pattern hangs when scaled to multiple process groups

---

## Summary

`ddp_visiontransformer.py` calls `init_process_group` without `device_id`, so PyTorch guesses which
GPU each rank owns. The guess is wrong on LUMI. It is harmless for the guide's single-group DDP
example, but the pattern deadlocks once a job creates more than one process group — which any
tensor/pipeline/expert-parallel job does.

## Reproduce the warning

One node, 2 ranks:

```bash
srun -N1 -n2 --gpus-per-node=2 singularity run "$SIF" bash -c \
  'export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NPROCS
   export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) MASTER_PORT=29511
   python3 -c "
import torch, torch.distributed as dist
dist.init_process_group(\"nccl\")
print(\"rank\", dist.get_rank(), \"ok\")"'
```

```
UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current
device set by the user. ... Guessing device ID based on global rank. This can cause a hang if
rank to GPU mapping is heterogeneous.
```

## Reason

The guide's scripts give each rank one GPU (`ROCR_VISIBLE_DEVICES`/`--gpus-per-node` with 8 tasks),
so **every rank sees its GPU as index 0**. PyTorch guesses `rank N → device N`, which is right only
for rank 0. That is the "heterogeneous rank to GPU mapping" the warning names.

Note the ordering in `ddp_visiontransformer.py:61-64`: `init_process_group()` is called *before*
`torch.cuda.set_device(local_rank)`, so there is no current device set for PyTorch to adopt at init
time.

One process group tolerates the wrong guess. More than one does not. Measured at 128 ranks:

| | no `device_id` | with `device_id` |
| --- | --- | --- |
| 1st process group | 13.9 s | 1.14 s |
| 2nd process group | **never returned** | 0.29 s |

Scope, honestly: the warning reproduces at 2 ranks, but the hang above was observed at 128 ranks
with more than one group. We did not establish the threshold in between, so treat the hang as
"happens at scale" rather than "happens at N ranks".

## Mitigation

Set the device first, then declare it:

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",
    device_id=torch.device("cuda", local_rank),
)
```

This silences the warning and removes the failure mode. It costs nothing for the guide's own
single-group example, and it is the difference between working and deadlocking for a reader who
extends it to model-parallel training.

`ds_visiontransformer.py` sets the device but leaves initialisation to DeepSpeed, so it is worth
checking whether the same argument can be threaded there.
