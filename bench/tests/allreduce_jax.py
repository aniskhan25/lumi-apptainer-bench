import os
import functools
import time

from tests import distributed

_initialized = False


def _init_distributed():
    global _initialized
    if _initialized:
        return True, ""

    rank = distributed.env_int("RANK", distributed.env_int("SLURM_PROCID", -1))
    world_size = distributed.env_int("WORLD_SIZE", distributed.env_int("SLURM_NTASKS", -1))
    local_rank = distributed.env_int("LOCAL_RANK", distributed.env_int("SLURM_LOCALID", 0))
    if rank < 0 or world_size < 1:
        return False, "missing rank/world_size"

    master_addr = distributed.master_addr_from_slurm()
    if not master_addr:
        master_addr = os.environ.get("MASTER_ADDR", "")
    if not master_addr:
        return False, "missing MASTER_ADDR"
    master_port = os.environ.get("MASTER_PORT", "29500")

    try:
        import jax
        jax.distributed.initialize(
            coordinator_address=f"{master_addr}:{master_port}",
            num_processes=world_size,
            process_id=rank,
            local_device_ids=[0],
        )
        _initialized = True
        return True, ""
    except Exception as exc:
        return False, str(exc)


def run_allreduce(message_sizes, iters=5):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return {"error": "jax not available"}

    ok, err = _init_distributed()
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    n_local = len(jax.local_devices())

    @functools.partial(jax.pmap, axis_name="devices")
    def allreduce_fn(x):
        return jax.lax.psum(x, axis_name="devices")

    results = {
        "message_sizes_bytes": [],
        "bandwidth_gbps": [],
        "latency_us": [],
        "checksum": "",
    }

    for size in message_sizes:
        numel = max(size // 4, 1)
        per_device = max(numel // n_local, 1)
        x = jnp.ones((n_local, per_device), dtype=jnp.float32)

        for _ in range(2):
            allreduce_fn(x).block_until_ready()

        start = time.perf_counter()
        for _ in range(max(iters, 1)):
            allreduce_fn(x).block_until_ready()
        end = time.perf_counter()

        avg_time = (end - start) / max(iters, 1)
        bandwidth = (size / avg_time) / 1.0e9 if avg_time > 0 else 0.0

        results["message_sizes_bytes"].append(size)
        results["bandwidth_gbps"].append(bandwidth)
        results["latency_us"].append(avg_time * 1.0e6)

    last = allreduce_fn(x)
    results["checksum"] = f"{float(last[0, 0]):.4f}"
    return results
