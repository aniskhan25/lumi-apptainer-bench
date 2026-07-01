import os
import time
import numpy as np
from functools import partial

from tests import distributed

_initialized = False


def _init_distributed():
    global _initialized
    if _initialized:
        return True, ""

    rank = distributed.env_int("RANK", distributed.env_int("SLURM_PROCID", -1))
    world_size = distributed.env_int("WORLD_SIZE", distributed.env_int("SLURM_NTASKS", -1))
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
        # No local_device_ids restriction — each process uses all visible GPUs
        jax.distributed.initialize(
            coordinator_address=f"{master_addr}:{master_port}",
            num_processes=world_size,
            process_id=rank,
        )
        _initialized = True
        return True, ""
    except Exception as exc:
        return False, str(exc)


def run_ddp_step(
    batch_size=64,
    input_size=4096,
    output_size=4096,
    warmup=3,
    iters=10,
    dtype_name="bfloat16",
):
    try:
        import jax
        import jax.numpy as jnp
        import jax.lax as lax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from jax.experimental.shard_map import shard_map
    except ImportError as e:
        return {"error": f"jax not available: {e}"}

    ok, err = _init_distributed()
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    n_total = len(jax.devices())
    n_local = len(jax.local_devices())
    if n_total < 2:
        return {"error": f"need at least 2 total devices, got {n_total}"}

    dtype = getattr(jnp, dtype_name, jnp.bfloat16)

    # 1D global mesh — all devices across all processes
    mesh = Mesh(np.array(jax.devices()), axis_names=("batch",))
    replicated = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P("batch"))

    # shard_map: each device runs local forward+backward; pmean allreduces grads
    @partial(shard_map, mesh=mesh, in_specs=(P(), P("batch")), out_specs=P())
    def compute_grads(w, x_shard):
        def loss_fn(w):
            return jnp.sum(jnp.dot(x_shard, w))
        _, grads = jax.value_and_grad(loss_fn)(w)
        return lax.pmean(grads, "batch")

    @jax.jit
    def step_fn(w, x):
        grads = compute_grads(w, x)
        return w - jnp.array(1e-3, dtype=dtype) * grads

    # Each process provides its local portion of the global arrays
    w = jax.make_array_from_process_local_data(
        replicated,
        jnp.ones((input_size, output_size), dtype=dtype),
    )
    local_x = jnp.ones((n_local * batch_size, input_size), dtype=dtype)
    x = jax.make_array_from_process_local_data(sharded, local_x)

    # Trigger compilation
    w = step_fn(w, x)
    jax.block_until_ready(w)

    for _ in range(max(warmup, 0)):
        w = step_fn(w, x)
    jax.block_until_ready(w)

    step_times = []
    for _ in range(max(iters, 1)):
        start = time.perf_counter()
        w = step_fn(w, x)
        jax.block_until_ready(w)
        end = time.perf_counter()
        step_times.append((end - start) * 1000.0)

    step_times_sorted = sorted(step_times)
    mid = len(step_times_sorted) // 2
    p50 = step_times_sorted[mid]
    p95 = step_times_sorted[int(0.95 * (len(step_times_sorted) - 1))]
    avg = sum(step_times) / len(step_times)

    global_batch = batch_size * n_total
    samples_per_sec = (global_batch / (avg / 1000.0)) if avg > 0 else 0.0

    return {
        "batch_size": batch_size,
        "input_size": input_size,
        "output_size": output_size,
        "dtype": dtype_name,
        "world_size": n_total,
        "step_time_ms_avg": avg,
        "step_time_ms_p50": p50,
        "step_time_ms_p95": p95,
        "samples_per_sec": samples_per_sec,
    }
