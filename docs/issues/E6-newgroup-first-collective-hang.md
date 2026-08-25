**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** First collective on a new process group intermittently hangs forever, with no timeout and no diagnostics

---

## Summary

On 4 nodes / 32 ranks, the first collective on a newly created world-spanning process group
sometimes never returns. `new_group()` succeeds. The `all_reduce` after it, which is where RCCL
lazily initialises the communicator, blocks on **all** ranks indefinitely.

There is no timeout, no exception and no flight-recorder dump. The job hangs until Slurm kills it.

Hung in 13 of 17 attempts across four allocations. A single-communicator job in the same
allocation passed every time, so the trigger is the additional process groups, not the nodes.

## Reproduce

```python
# hang.py
import os, torch, torch.distributed as dist
LR = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(LR)
dist.init_process_group("nccl")
R = dist.get_rank()
for i in range(1, 9):
    g = dist.new_group()                                          # world-spanning
    print(f"r{R} group {i}: created", flush=True)
    dist.all_reduce(torch.ones(1024, device=f"cuda:{LR}"), group=g)
    torch.cuda.synchronize()
    print(f"r{R} group {i}: collective done", flush=True)         # never printed on a hang
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
  timeout 240 srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity run $SIF bash -c \
    "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python3 -u hang.py"
  echo "attempt $i -> exit $?   (124 = hung)"
done
```

## Observed

Every rank prints `created` for group N; **no** rank prints `collective done`. The group index at
which it stalls varies (we saw 2, 4, 5 and 8), and each earlier communicator comes up in ~1.5 s.
So it is not a ceiling or a limit at a particular count. All 32 ranks block at the same call, so
there is no straggler.

With `timeout=timedelta(seconds=90)` on the process group and a 240 s wall cap, four hangs produced
zero `Watchdog caught` messages, zero `DistBackendError`, and zero dumps under
`TORCH_FR_BUFFER_SIZE=2000` + `TORCH_NCCL_DUMP_ON_TIMEOUT=1`. Consistent with the block landing
before a `WorkNCCL` is enqueued: nothing to time out, nothing to record.

## Rates, and a control

| Job | Nodes | Attempts | Hung |
| --- | --- | --- | --- |
| 21436817 | `nid[007434,007455,007457,007461]` | 5 | 5 |
| 21499264 | `nid[007745-007748]` | 5 | 4 |
| 21492040 | `nid[006972-006975]` | 3 | 1 |
| 21516331 | `nid[005818-005821]` | 4 | 3 |

**A single-communicator job is not affected.** In job 21516331 we alternated two workloads inside one
allocation, on the same four nodes: a plain single-group DDP job (wrap a model in
`DistributedDataParallel`, one forward/backward, so one communicator) and the eight-group reproducer
above.

| Round | 1 group | 8 groups |
| --- | --- | --- |
| 1 | pass | hung at group 2 |
| 2 | pass | hung at group 4 |
| 3 | pass | hung at group 7 |
| 4 | pass | pass |

Single-group passed 4 of 4 while eight-group hung 3 of 4, in the same allocation, alternating. A
separate 5-attempt single-group run also passed 5 of 5, so 9 of 9 overall. The trigger is therefore
creating process groups beyond the default, not node health: the paired design holds nodes constant.

Passing `device_id` to `init_process_group` makes initialisation eager and moves the stall into that
call instead. Same failure, different call site. That variant hung 2 of 5 allocations at 4 nodes and
3 of 3 at 16 nodes.

## Who this affects

Multiple process groups are what model parallelism is built on, so the exposure is tensor, pipeline
and expert parallelism, not data-parallel training.

That is a workload this platform is built for but does not document. The `full` image ships
`megatron-core 0.15.0rc8`, and Megatron exists only for model-parallel training. Meanwhile
LUMI-AI-Guide chapter 5 covers DDP and DeepSpeed ZeRO stage 1, and never mentions Megatron or
tensor, pipeline, expert or model parallelism anywhere in its eleven chapters.

The combination is awkward for users. The documented paths are all single-communicator and pass
consistently in our testing, which is likely why this has not been reported. The failure sits in
undocumented territory, produces no error, no timeout and no traceback, and is therefore easy to
attribute to one's own code.

## Environment

- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`, digest `d70ec87f…`
- PyTorch `2.10.0+rocm7.0`, RCCL `2.26.6`, `aws-ofi-nccl 1.20.0-git-a2a6d08`, ROCm 7.0
- `standard-g`, 4 and 16 nodes, 8 ranks/node

## Notes

- `exit 124` is our wall cap. We did not test whether these would eventually return, and repeats
  within one allocation are not independent samples.
- Possibly the same root cause as #20, which is labelled for the `u24r64` generation. This is on
  `u24r70` with the failing call identified.
- Happy to run `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` on this reproducer if useful.
