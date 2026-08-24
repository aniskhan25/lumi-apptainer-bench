**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** First collective on a new process group intermittently hangs forever, with no timeout and no diagnostics

---

## Summary

On 4 nodes / 32 ranks, the first collective on a newly created world-spanning process group
sometimes never returns. `new_group()` succeeds; the `all_reduce` that follows it — where RCCL
lazily initialises the communicator — blocks indefinitely on **all** ranks.

PyTorch's watchdog does not detect it, so there is no timeout, no exception, and no flight-recorder
dump. The job simply hangs until Slurm kills it.

Reproduced 10 times in 13 attempts across three separate allocations, using the LUMI AI Guide's
Chapter 5 launch recipe unmodified.

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

for i in 1 2 3 4 5; do
  timeout 240 srun singularity run $SIF bash -c \
    "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python3 -u hang.py"
  echo "attempt $i -> exit $?   (124 = hung)"
done
```

## What we see

Every rank prints `created` for group N; **no** rank prints `collective done`. The group index at
which it stalls varies run to run — we saw 2, 4, 5 and 8 — and every earlier communicator comes up
in about 1.5 s. So this is not a ceiling or a resource limit at a particular count.

There is no straggler: all 32 ranks block at the same call.

## Why it is hard to diagnose

With `timeout=timedelta(seconds=90)` on the process group and a 240 s wall cap, across four hangs:

- zero `Watchdog caught collective operation timeout` messages
- zero `DistBackendError`
- zero dumps, with `TORCH_FR_BUFFER_SIZE=2000` and `TORCH_NCCL_DUMP_ON_TIMEOUT=1`

That is consistent with the block landing *before* a `WorkNCCL` is enqueued — there is no work item
to time out and nothing for the flight recorder to record. Users therefore get an indefinite hang
with no error message at all.

(Note `TORCH_NCCL_TRACE_BUFFER_SIZE` is deprecated in this build in favour of
`TORCH_FR_BUFFER_SIZE`; using the old name silently disables the recorder.)

## Rates

| Job | Nodes | Attempts | Hung |
| --- | --- | --- | --- |
| 21436817 | `nid[007434,007455,007457,007461]` | 5 | 5 |
| 21499264 | `nid[007745-007748]` | 5 | 4 |
| 21492040 | `nid[006972-006975]` | 3 | 1 |

With a variant that restricts visibility (`ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`) and passes
`device_id`, making initialisation eager, the stall moves into `init_process_group` instead — same
failure, different call site. That variant hung 2 of 5 allocations at 4 nodes and **3 of 3 at 16
nodes**.

The `nid007xxx` allocations fared much worse than the `nid006xxx` one (9/10 vs 1/3). We tested and
rejected a node-placement hypothesis on an earlier, different dataset, so we are not asserting one —
the node lists are above in case they match your own failures.

## Environment

- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`, digest
  `d70ec87fda17e97ff3b3241bcb34774365bba5f7b9172a22b8fda0897213bc81`
- PyTorch `2.10.0+rocm7.0`, RCCL `2.26.6`, `aws-ofi-nccl 1.20.0-git-a2a6d08`, ROCm 7.0
- `standard-g`, 4 and 16 nodes, 8 ranks/node

## Notes

- `exit 124` is our own wall cap. We know these do not finish within 240 s; we have not tested
  whether they would ever finish.
- Repeated attempts inside one allocation are not independent samples.
- Possibly the same root cause as #20 ("RCCL communications sometimes hang with PyTorch DDP"),
  which is labelled for the `u24r64` generation. This is on the current `u24r70` build with the
  failing call identified.
- Happy to run `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,NET` on this reproducer if the
  RCCL-internal view would help.
