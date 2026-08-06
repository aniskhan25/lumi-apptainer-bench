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

## Comment / question on #28 — "Multi-node `torch.distributed.init` fails."

> Two observations from validating `20260513_121430` that may bear on this.
>
> **1. This may explain a separate user report.** A LUMI project reported a 32-rank expert
> all-to-all failing during initialisation with an NCCL "unhandled system error" at 16 nodes, on
> the **April** `20260415_130625` build — the same build this issue is about. We could not
> reproduce that on the **May** build: 32-rank all-to-all groups spanning 4 nodes initialise and
> exchange correctly, including 4 concurrent 32-rank meshes across 128 ranks, 2/2 runs, in ~33 s.
> So their "EP=32 fails on the fabric" may simply be this issue rather than anything specific to
> 32 ranks — worth considering if you are tracking user impact of #28.
>
> **2. The build-to-build direction is inconsistent, which may matter for diagnosis.** This issue
> reports April broken and March (`20260319_153422`) working. The same user report says April
> worked for them and **May** regressed inter-node all-to-all with `PTLTE_NOT_FOUND`, which is
> why they pinned April. Those two cannot both be a simple monotonic regression.
>
> The more likely reading is that multi-node init stability depends on configuration as much as
> on build — with #30 (`NCCL_NET_GDR_LEVEL`) and #20 (straggler rank / node state) being two
> known configuration-dependent causes. If that is right, "which build is broken" is the wrong
> question and the useful artefacts are the full environment plus node list from each failing
> run.
>
> **3. A third data point, build-independent.** On 2026-08-06 a run on the **March**
> `20260319_153422` build — the one this issue reports as working — succeeded at 2 nodes and failed
> at both 128 and 256 nodes with a collective timeout rather than an init failure:
>
> ```
> [PG ID 10 PG GUID 10 Rank 0] Watchdog caught collective operation timeout:
> WorkNCCL(SeqNum=4, OpType=BROADCAST, NumelIn=68585472, NumelOut=68585472,
> Timeout(ms)=600000) ran for 600025 milliseconds before timing out.
> ```
>
> (job 20766576, `standard-g`, 128 nodes / 1024 ranks; the step ran 13 min 23 s before the
> watchdog aborted rank 0). The same failure was then seen on the `rocm` variant, so it does not
> track the ROCm version either.
>
> Note this is on a **non-default process group** (PG ID 10) at **SeqNum 4** — so three collectives
> had already succeeded on that group and the fabric was working for it. That points at one rank
> diverging or stalling rather than at connection establishment, and it is the same shape as a
> failure we reproduced from an unrelated cause: a single rank dying in Inductor compilation while
> the other 127 wait at the next collective. So a timeout on rank 0 is not by itself evidence about
> the fabric.
>
> Taken with the two points above, "which build is broken" looks increasingly like the wrong axis:
> the same class of failure now appears on March, April and the `rocm` variant, and varies with
> scale and configuration instead.
>
> For what it is worth we could not reproduce a cross-node all-to-all failure on the May build at
> any group size (8/16/32) or node count (1/2/4/16), with rank-tagged correctness checking and
> uneven/zero-token dispatch. Details and job IDs:
> https://github.com/aniskhan25/lumi-apptainer-bench/blob/feature/laif-container-validation/docs/FINDINGS.md
