"""All-to-all collective test.

This is the collective the shipped container test suite never exercises. Every
collective test in that suite is either an allreduce (the DDP tests) or a
point-to-point transfer (the OSU bandwidth tests), and the inter-node OSU test runs
one process per node. An 8-rank-per-node all-to-all opens far more fabric endpoints
than either, which is why a multi-node all-to-all failure can survive a suite that
otherwise passes.

`--group-size` selects the communicator topology, which is the axis that matters on
LUMI-G: a group of 8 stays inside one node on XGMI, while 16 or 32 crosses Slingshot.
Expert parallelism in a Mixture-of-Experts model is exactly this collective, so the
group size here corresponds directly to an EP degree.

Correctness is checked with rank-tagged payloads rather than a summed checksum -- a
sum cannot distinguish a correct exchange from a permuted one, and a bandwidth number
taken from a silently wrong exchange is worse than no number at all.
"""

import os
import time

from common import stats
from tests import distributed


# Chosen to span the range that matters for MoE expert dispatch: control-sized
# messages, activation-sized messages, and large dispatch buffers.
DEFAULT_MESSAGE_SIZES = [16384, 262144, 1048576, 4194304, 16777216]
TAG_SCALE = 1000


def _ranks_per_node():
    return distributed.env_int(
        "BENCH_NTASKS_PER_NODE", distributed.env_int("SLURM_NTASKS_PER_NODE", 0)
    )


def _build_group(torch, world_size, group_size):
    """Create every subgroup and return the one this rank belongs to.

    dist.new_group is collective over the whole world: every rank must call it for
    every group, even groups it is not a member of.
    """
    rank = torch.distributed.get_rank()
    my_group = None
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        group = torch.distributed.new_group(ranks=ranks)
        if rank in ranks:
            my_group = group
    return my_group


def _tagged_input(torch, group_rank, group_size, chunk_numel, device):
    """Chunk destined for peer j is filled with (my_group_rank * TAG_SCALE + j)."""
    values = torch.empty(group_size * chunk_numel, dtype=torch.float32, device=device)
    for dst in range(group_size):
        values[dst * chunk_numel : (dst + 1) * chunk_numel] = float(
            group_rank * TAG_SCALE + dst
        )
    return values


def _check_tagged_output(torch, output, group_rank, group_size, chunk_numel):
    """Chunk received from peer j must be (j * TAG_SCALE + my_group_rank)."""
    mismatches = 0
    first_bad = None
    for src in range(group_size):
        expected = float(src * TAG_SCALE + group_rank)
        chunk = output[src * chunk_numel : (src + 1) * chunk_numel]
        bad = int(torch.count_nonzero(chunk != expected).item())
        if bad and first_bad is None:
            first_bad = {
                "from_group_rank": src,
                "expected": expected,
                "observed": float(chunk.flatten()[0].item()),
            }
        mismatches += bad
    return mismatches, first_bad


def _uneven_case(torch, group, group_rank, group_size, device):
    """Uneven splits with zero-token peers, the shape MoE routing actually produces.

    Token counts per expert vary with routing, and some experts receive nothing at all.
    An implementation that only ever sees equal splits can pass every equal-split test
    and still fail here.
    """
    # Rank r sends (r + j) % 3 chunks to peer j, so some pairs exchange nothing.
    unit = 4096
    input_splits = [((group_rank + dst) % 3) * unit for dst in range(group_size)]

    # Every rank needs to know how much it will receive before it can size the output.
    send_counts = torch.tensor(input_splits, dtype=torch.int64, device=device)
    recv_counts = torch.empty_like(send_counts)
    torch.distributed.all_to_all_single(recv_counts, send_counts, group=group)
    output_splits = [int(v) for v in recv_counts.tolist()]

    send = torch.empty(sum(input_splits), dtype=torch.float32, device=device)
    offset = 0
    for dst, count in enumerate(input_splits):
        if count:
            send[offset : offset + count] = float(group_rank * TAG_SCALE + dst)
        offset += count

    recv = torch.empty(sum(output_splits), dtype=torch.float32, device=device)
    torch.distributed.all_to_all_single(
        recv, send, output_splits, input_splits, group=group
    )
    torch.cuda.synchronize()

    mismatches = 0
    offset = 0
    for src, count in enumerate(output_splits):
        if count:
            expected = float(src * TAG_SCALE + group_rank)
            chunk = recv[offset : offset + count]
            mismatches += int(torch.count_nonzero(chunk != expected).item())
        offset += count

    return {
        "input_splits": input_splits,
        "output_splits": output_splits,
        "zero_token_peers": sum(1 for c in input_splits if c == 0),
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }


def _group_churn(torch, world_size, group_size, rounds):
    """Repeated communicator create/destroy.

    A leak or a resource exhaustion in communicator setup shows up here and not in a
    single-shot test.
    """
    failures = []
    for round_index in range(rounds):
        try:
            group = _build_group(torch, world_size, group_size)
            tensor = torch.ones(group_size * 32, dtype=torch.float32, device="cuda")
            out = torch.empty_like(tensor)
            torch.distributed.all_to_all_single(out, tensor, group=group)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"round {round_index}: {type(exc).__name__}: {exc}")
    return {"rounds": rounds, "failures": failures, "passed": not failures}


def run_alltoall(message_sizes=None, group_size=8, iters=10, churn_rounds=3):
    try:
        import torch
    except ImportError as exc:
        return {"error": f"torch not available: {exc}"}
    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    ok, err = distributed.init_process_group(torch)
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    try:
        device_index = distributed.local_cuda_index(torch)
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)

        world_size = torch.distributed.get_world_size()
        if group_size > world_size:
            return {
                "error": (
                    f"group_size {group_size} exceeds world_size {world_size}; "
                    f"need at least {group_size} ranks"
                )
            }
        if world_size % group_size != 0:
            return {
                "error": (
                    f"world_size {world_size} is not a multiple of "
                    f"group_size {group_size}"
                )
            }

        sizes = message_sizes or DEFAULT_MESSAGE_SIZES
        ranks_per_node = _ranks_per_node()
        group = _build_group(torch, world_size, group_size)
        group_rank = torch.distributed.get_rank(group=group)

        result = {
            "world_size": world_size,
            "group_size": group_size,
            "num_groups": world_size // group_size,
            "ranks_per_node": ranks_per_node,
            # A group larger than one node's rank count must cross Slingshot; a group
            # that fits stays on XGMI. This is the EP=8 vs EP>8 distinction.
            "spans_nodes": bool(ranks_per_node and group_size > ranks_per_node),
            "message_sizes_bytes": [],
            "bandwidth_gbps": [],
            "latency_us": [],
        }

        correctness = {"passed": True, "mismatches": 0, "per_size": {}}
        for size in sizes:
            chunk_numel = max(size // 4 // group_size, 1)
            send = _tagged_input(torch, group_rank, group_size, chunk_numel, device)
            recv = torch.empty_like(send)

            torch.distributed.all_to_all_single(recv, send, group=group)
            torch.cuda.synchronize()
            mismatches, first_bad = _check_tagged_output(
                torch, recv, group_rank, group_size, chunk_numel
            )
            correctness["per_size"][str(size)] = {
                "mismatches": mismatches,
                "first_mismatch": first_bad,
            }
            correctness["mismatches"] += mismatches
            if mismatches:
                correctness["passed"] = False

            # Only measure once the exchange is known to be correct for this size.
            if mismatches:
                continue

            for _ in range(2):
                torch.distributed.all_to_all_single(recv, send, group=group)
            torch.cuda.synchronize()
            torch.distributed.barrier(group=group)

            durations = []
            for _ in range(max(iters, 1)):
                start = time.perf_counter()
                torch.distributed.all_to_all_single(recv, send, group=group)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)

            # Report the slowest rank in the group: an all-to-all completes only when
            # its last participant does, so a rank-0-only timing flatters the result.
            local = torch.tensor(durations, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(
                local, op=torch.distributed.ReduceOp.MAX, group=group
            )
            durations = [float(v) for v in local.tolist()]

            p50 = stats.percentile(durations, 50)
            # Bytes that actually leave the rank: everything except its own chunk.
            bytes_off_rank = chunk_numel * 4 * (group_size - 1)
            result["message_sizes_bytes"].append(size)
            result["bandwidth_gbps"].append((bytes_off_rank / p50) / 1.0e9)
            result["latency_us"].append(p50 * 1.0e6)

        result["correctness"] = correctness
        result["uneven"] = _uneven_case(torch, group, group_rank, group_size, device)
        result["group_churn"] = _group_churn(
            torch, world_size, group_size, churn_rounds
        )
        result["passed"] = bool(
            correctness["passed"]
            and result["uneven"]["passed"]
            and result["group_churn"]["passed"]
        )
        return result
    except Exception as exc:  # noqa: BLE001
        # A fabric failure here is the finding, not a crash to hide. PTLTE_NOT_FOUND and
        # "unhandled system error" both surface as exceptions out of the collective.
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "group_size": group_size,
            "hostname": os.environ.get("SLURMD_NODENAME", ""),
        }
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
