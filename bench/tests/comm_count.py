"""Concurrent communicator count stress.

Targets the remaining hypothesis for report §4.1. Every collective test so far has been
clean at group sizes 8, 16 and 32 across 1, 2, 4 and 16 nodes, so the group's *shape* is not
the problem. What none of those tests reproduce is Megatron's communicator *population*: a
real MoE job builds data-parallel, tensor-parallel, pipeline-parallel, expert-parallel and
expert-data-parallel groups at once, so each rank holds many live communicators
simultaneously rather than the one or four this suite has been creating.

`PTLTE_NOT_FOUND` concerns Portals List Table Entries, a finite per-NIC resource. Each
network-using communicator consumes CXI endpoint resources, and a LUMI-G node has 8 ranks
sharing 4 NICs, so the per-node demand is (communicators per rank x 8). That product, not the
group size, is what this test drives.

It also answers a second open question from report §4.2. The HIP context and the RCCL/CXI
communicator buffers are registered outside PyTorch's caching allocator, so they are
invisible to `torch.cuda.memory_allocated()`. Sampling `torch.cuda.mem_get_info()` after each
communicator measures that hidden cost directly, which is what the report could only
estimate.

Groups are created and *kept alive*. Destroying them would defeat the point.
"""

import json
import os
import time

from tests import distributed


DEFAULT_MAX_GROUPS = 32
# Small payload: this test is about communicator resources, not bandwidth. all_to_all is used
# rather than all_reduce because it establishes connections to every peer, so it forces the
# full endpoint set to be materialised.
PROBE_ELEMS = 1024


def _mem(torch):
    """Both views of memory: PyTorch's own accounting and the device's."""
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "device_free_mib": round(free / 1024**2, 1),
        "device_used_mib": round((total - free) / 1024**2, 1),
    }


def _exercise(torch, group, device):
    """Force the communicator to actually initialise.

    dist.new_group() does not create the RCCL communicator; that happens lazily on first
    use. Without this the test would count group objects rather than communicators and
    would consume no fabric resources at all.
    """
    send = torch.ones(PROBE_ELEMS, dtype=torch.float32, device=device)
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group)
    torch.cuda.synchronize()


def run_comm_count(max_groups=DEFAULT_MAX_GROUPS, results_dir=""):
    try:
        import torch
    except ImportError as exc:
        return {"error": f"torch not available: {exc}"}
    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    ok, err = distributed.init_process_group(torch)
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ranks_per_node = distributed.env_int(
        "BENCH_NTASKS_PER_NODE", distributed.env_int("SLURM_NTASKS_PER_NODE", 0)
    )

    rank_dir = os.path.join(results_dir or ".", "comm_count_ranks")
    rank_file = os.path.join(rank_dir, f"rank{rank:04d}.json")

    record = {
        "rank": rank,
        "world_size": world_size,
        "ranks_per_node": ranks_per_node,
        "hostname": os.environ.get("SLURMD_NODENAME", ""),
        "max_groups_requested": max_groups,
        "steps": [],
    }

    groups = []  # held deliberately: releasing them would release the resources under test
    try:
        device_index = distributed.local_cuda_index(torch)
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)

        record["baseline_memory"] = _mem(torch)

        for index in range(1, max_groups + 1):
            step = {"communicators": index}
            started = time.perf_counter()
            try:
                # A world-spanning group, so every communicator uses the network.
                group = torch.distributed.new_group()
                step["create_seconds"] = time.perf_counter() - started

                exercised = time.perf_counter()
                _exercise(torch, group, device)
                step["exercise_seconds"] = time.perf_counter() - exercised

                groups.append(group)
                step["ok"] = True
                step["memory"] = _mem(torch)
            except Exception as exc:  # noqa: BLE001 - the failure is the measurement
                step["ok"] = False
                step["seconds"] = time.perf_counter() - started
                step["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:1500],
                }
                record["steps"].append(step)
                break
            record["steps"].append(step)

        succeeded = [s for s in record["steps"] if s["ok"]]
        record["communicators_created"] = len(succeeded)
        record["reached_limit"] = len(succeeded) == max_groups
        failed = [s for s in record["steps"] if not s["ok"]]
        record["first_failure"] = failed[0] if failed else None
        record["ok"] = not failed

        if succeeded:
            base = record["baseline_memory"]["device_used_mib"]
            last = succeeded[-1]["memory"]["device_used_mib"]
            record["device_mib_per_communicator"] = round(
                (last - base) / len(succeeded), 2
            )
            record["device_mib_total_growth"] = round(last - base, 1)
    except Exception as exc:  # noqa: BLE001
        record["ok"] = False
        record["fatal"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}

    # Written before the barrier below: if a peer has already died or the fabric has run out
    # of endpoints, that barrier hangs and this file is the only surviving evidence.
    os.makedirs(rank_dir, exist_ok=True)
    with open(rank_file, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)

    barrier_ok = True
    barrier_error = ""
    try:
        torch.distributed.barrier()
    except Exception as exc:  # noqa: BLE001
        barrier_ok = False
        barrier_error = f"{type(exc).__name__}: {exc}"

    aggregate = None
    if rank == 0:
        aggregate = _aggregate(rank_dir, world_size, ranks_per_node, max_groups)
        aggregate["barrier_ok"] = barrier_ok
        aggregate["barrier_error"] = barrier_error

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return aggregate if rank == 0 else {"rank_record_written": rank_file}


def _aggregate(rank_dir, world_size, ranks_per_node, max_groups):
    records = []
    for rank in range(world_size):
        path = os.path.join(rank_dir, f"rank{rank:04d}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        except (OSError, ValueError):
            continue

    reported = {r["rank"] for r in records}
    missing = [r for r in range(world_size) if r not in reported]
    created = [r["communicators_created"] for r in records if "communicators_created" in r]
    failed = [r for r in records if not r.get("ok")]
    per_comm = [
        r["device_mib_per_communicator"]
        for r in records
        if r.get("device_mib_per_communicator") is not None
    ]

    return {
        "world_size": world_size,
        "ranks_per_node": ranks_per_node,
        "max_groups_requested": max_groups,
        "ranks_reporting": len(records),
        "ranks_missing": missing,
        "ranks_failed": sorted(r["rank"] for r in failed),
        # The headline: the smallest number any rank managed. A collective is only usable if
        # every participant created it, so the minimum is the real limit.
        "communicators_min": min(created) if created else 0,
        "communicators_max": max(created) if created else 0,
        "all_reached_limit": bool(created) and min(created) == max_groups,
        # Per-node endpoint demand, which is the quantity Portals table entries are consumed by.
        "communicators_per_node": (min(created) * ranks_per_node) if created else 0,
        "device_mib_per_communicator_avg": (
            round(sum(per_comm) / len(per_comm), 2) if per_comm else None
        ),
        "first_failures": [
            {"rank": r["rank"], **(r.get("first_failure") or {})}
            for r in failed
            if r.get("first_failure")
        ][:5],
        "fatals": [
            {"rank": r["rank"], **r["fatal"]} for r in records if r.get("fatal")
        ][:5],
        "passed": not missing and not failed,
    }
