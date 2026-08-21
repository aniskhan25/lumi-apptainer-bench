**Repo:** `Lumi-supercomputer/LUMI-AI-Guide`
**Title:** ~~Chapter 5: pass `device_id` to `init_process_group`~~ — **WITHDRAWN**

---

## Withdrawn 2026-08-21, before filing

The premise was wrong. This draft argued that `ddp_visiontransformer.py` omitting `device_id`, and
calling `init_process_group()` before `torch.cuda.set_device(local_rank)`, taught a pattern that
deadlocks at scale.

Two things are wrong with that:

1. **The ordering does not matter.** `init_process_group` is lazy; the NCCL communicator binds at the
   first collective. In the guide's script `set_device(local_rank)` runs at line 64 and the first
   collective is the parameter broadcast inside `DistributedDataParallel(...)` at line 79. So a
   correct current device is set well before anything communicates, and PyTorch adopts it rather than
   guessing.

2. **Our measurement did not isolate `device_id`.** The commit that fixed our hang (`2ecabcb`) added
   `torch.cuda.set_device(index)` *and* `device_id=` together, and the failing runs had neither. The
   attributable cause is "no device bound before the first collective" — which is exactly the thing
   the guide already does correctly.

So the guide's pattern is fine, and there is nothing here to file. `device_id` remains a reasonable
addition — it makes initialisation eager and silences the warning — but that is a preference, not a
defect, and not worth a maintainer's time.

Kept as a record of the correction rather than deleted.
