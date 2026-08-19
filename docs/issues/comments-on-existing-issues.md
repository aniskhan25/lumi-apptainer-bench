Comments to add to existing `laifs-container-recipes` issues rather than filing duplicates.

---

## Comment on #30 — "Setting `NCCL_NET_GDR_LEVEL` may cause jobs to hang"

> Independent confirmation on `u24r70f21m50t210-20260513_121430`, in case another data point is
> useful.
>
> Our multi-node benchmark templates had `NCCL_NET_GDR_LEVEL=PHB` and
> `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3` set together. A 2-node allreduce hung indefinitely;
> removing both made the same job complete in 24 s (job 19624583). We now default them off and
> keep them behind an explicit opt-in used only for deliberate fabric tuning.
>
> Matches your observation that performance is fine without forcing the GDR level — we see no
> measurable loss from leaving it unset.

---

## Comment on #20 — "RCCL communications sometimes hang with PyTorch DDP"

> Data point on whether this persists in the ROCm 7 / PyTorch 2.10 release you expected the fix
> in. We are on `full-u24r70f21m50t210-20260513_121430` (PyTorch `2.10.0+rocm7.0`, RCCL `2.26.6`)
> and still see intermittent hangs — but with a **node correlation** that may be worth checking
> against your own failures.
>
> Across a 5-run communicator-creation sweep (2/4/8/16 nodes, 8 ranks/node, up to 8 concurrent
> world-spanning communicators per rank), two runs hung and three passed:
>
> | Job | Nodes | Node list | Result |
> | --- | --- | --- | --- |
> | 20724372 | 4 | `nid[007038-007041]` | hang (killed at 10 min) |
> | 20711452 | 16 | `nid[007769-007784]` | hang (killed at 25 min) |
> | 20724354 | 2 | `nid[005556-005557]` | pass, 37 s |
> | 20724753 | 8 | `nid[006186-006193]` | pass, 45 s |
> | 20724970 | 16 | `nid[005724-005729,006186-006195]` | pass, 47 s |
>
> Both hangs on `nid007xxx`; all three passes on `nid005xxx`/`nid006xxx`. The result is
> non-monotonic in scale — 4 nodes hung while 8 and 16 passed — which argues against a
> rank-count or configuration cause and for node placement.
>
> Suggestion: if the `pytorch-ddp-multi-node` test failures are recorded with node lists, it may
> be worth checking whether they cluster the same way. If they do, part of this is a node-state
> problem rather than a container or PyTorch one, and `CUDA_LAUNCH_BLOCKING=1` may be masking a
> different cause than assumed.
>
> Separately, and possibly relevant to the straggler-rank theory: we found that omitting
> `device_id` from `init_process_group` reliably hung us at 128 ranks once a rank held more than
> one communicator (first communicator 13.9 s vs 1.14 s with `device_id`, second one never
> returning vs 0.29 s). Your #28 reproducer already passes `device_id`, so this is probably not
> your case — noting it because a single communicator per rank survives the wrong guess, so the
> symptom only appears in multi-communicator jobs.

---

## Comment on #28 — "Multi-node `torch.distributed.init` fails."

> One data point from validating the current release, in case it is useful for scoping.
>
> We could not reproduce a cross-node all-to-all or init failure at any group size (8/16/32) or
> node count (1/2/4/16), with rank-tagged correctness checking and uneven/zero-token dispatch —
> including four concurrent 32-rank meshes across 128 ranks, 2/2 runs, in ~33 s. Details and job
> IDs:
> https://github.com/aniskhan25/lumi-apptainer-bench/blob/feature/laif-container-validation/docs/FINDINGS.md
>
> Possibly relevant to the straggler-rank theory: omitting `device_id` from `init_process_group`
> reliably hung us at 128 ranks once a rank held more than one communicator (first communicator
> 13.9 s vs 1.14 s with `device_id`, second one never returning vs 0.29 s). Your reproducer already
> passes `device_id`, so this is probably not your case — noting it because a single communicator
> per rank survives the wrong guess, so the symptom only shows up in multi-communicator jobs.
