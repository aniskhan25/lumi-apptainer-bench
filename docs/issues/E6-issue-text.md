Paste-ready text for filing. The full record, with every run, timing and control, is in
[`E6-newgroup-first-collective-hang.md`](E6-newgroup-first-collective-hang.md).

---

**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** Creating a second process group intermittently hangs forever with no error

---

On 4 nodes / 32 ranks, the first collective on a newly created process group sometimes never
returns. `new_group()` succeeds, then the `all_reduce` that follows it, which is where RCCL
initialises the communicator, blocks on every rank indefinitely. There is no exception, no timeout
and no traceback. The job runs until Slurm kills it.

Hung in 13 of 17 attempts across four allocations. A single-group job in the same allocation passed
every time, so this affects tensor, pipeline and expert parallelism, not plain DDP.

## Reproduce

```python
# hang.py
import os, torch, torch.distributed as dist
LR = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(LR)
dist.init_process_group("nccl")
R = dist.get_rank()
for i in range(1, 9):
    g = dist.new_group()
    print(f"r{R} group {i}: created", flush=True)
    dist.all_reduce(torch.ones(1024, device=f"cuda:{LR}"), group=g)
    torch.cuda.synchronize()
    print(f"r{R} group {i}: collective done", flush=True)
```

```bash
#SBATCH --nodes=4 --gpus-per-node=8 --ntasks-per-node=8 --cpus-per-task=7 --mem-per-gpu=60G
module purge && module use /appl/local/laifs/modules && module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}"
export WORLD_SIZE=$SLURM_NPROCS
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

for i in 1 2 3 4 5; do
  timeout 240 srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS singularity run $SIF bash -c \
    "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python3 -u hang.py"
  echo "attempt $i -> exit $?   (124 = hung)"
done
```

A hung run prints `created` for group N on every rank and never prints `collective done`. Which
group it stalls on varies between runs, so no particular number of groups is the trigger.

We left one run alone for an hour. It was still blocked, and PyTorch's own timeout never fired.

## Environment

- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`, digest `d70ec87f...`
- PyTorch `2.10.0+rocm7.0`, RCCL `2.26.6`, `aws-ofi-nccl 1.20.0-git-a2a6d08`, ROCm 7.0
- `standard-g`, 4 nodes, 8 ranks/node. Also 3 of 3 hangs at 16 nodes.