import time
import functools

import numpy as np
from tests import allreduce_jax


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
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    except ImportError:
        return {"error": "jax not available"}

    ok, err = allreduce_jax._init_distributed()
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    n_total = len(jax.devices())
    if n_total < 2:
        return {"error": f"need at least 2 total devices, got {n_total}"}

    dtype = getattr(jnp, dtype_name, jnp.bfloat16)
    local_device = jax.local_devices()[0]

    # 1D mesh over all 16 global devices (1 per process).
    # With 1 GPU per process, RCCL does a plain network allreduce — no GDR clique.
    mesh = Mesh(np.array(jax.devices()), ('d',))

    # w is replicated: every device holds the full weight matrix.
    # x is sharded: each device holds 1 batch slice.
    @functools.partial(
        shard_map, mesh=mesh,
        in_specs=(P(), P('d')),
        out_specs=P(),
    )
    def step_fn(w, x):
        # w: (input_size, output_size) — full weight, same on all devices
        # x: (1, batch_size, input_size) — local data shard
        x_local = x[0]
        _, grads = jax.value_and_grad(
            lambda w: jnp.sum(jnp.dot(x_local, w))
        )(w)
        avg_grads = jax.lax.pmean(grads, 'd')  # real cross-process allreduce
        return w - jnp.array(1e-3, dtype=dtype) * avg_grads

    # Build global arrays. w replicated, x sharded along the device axis.
    local_w = jax.device_put(jnp.ones((input_size, output_size), dtype=dtype), local_device)
    local_x = jax.device_put(jnp.ones((1, batch_size, input_size), dtype=dtype), local_device)

    w = jax.make_array_from_single_device_arrays(
        (input_size, output_size),
        NamedSharding(mesh, P()),
        [local_w],
    )
    x = jax.make_array_from_single_device_arrays(
        (n_total, batch_size, input_size),
        NamedSharding(mesh, P('d')),
        [local_x],
    )

    def step_and_sync():
        nonlocal w
        w = step_fn(w, x)
        jax.block_until_ready(w)

    # Trigger compilation.
    step_and_sync()

    for _ in range(max(warmup, 0)):
        step_and_sync()

    step_times = []
    for _ in range(max(iters, 1)):
        start = time.perf_counter()
        step_and_sync()
        step_times.append((time.perf_counter() - start) * 1000.0)

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
