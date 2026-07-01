import time
from functools import partial

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
    except ImportError:
        return {"error": "jax not available"}

    # Reuse the working distributed init: 1 GPU per process, local_device_ids=[0]
    ok, err = allreduce_jax._init_distributed()
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    n_total = len(jax.devices())
    n_local = len(jax.local_devices())
    if n_total < 2:
        return {"error": f"need at least 2 total devices, got {n_total}"}

    dtype = getattr(jnp, dtype_name, jnp.bfloat16)

    @partial(jax.pmap, axis_name="devices")
    def step_fn(w, x):
        def loss_fn(w):
            return jnp.sum(jnp.dot(x, w))
        _, grads = jax.value_and_grad(loss_fn)(w)
        grads = jax.lax.pmean(grads, axis_name="devices")
        return w - jnp.array(1e-3, dtype=dtype) * grads

    # Each process contributes n_local (=1) device slices
    w = jnp.ones((n_local, input_size, output_size), dtype=dtype)
    x = jnp.ones((n_local, batch_size, input_size), dtype=dtype)

    # Trigger compilation + clique init
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
