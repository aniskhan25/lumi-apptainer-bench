import functools
import time

from tests import allreduce_jax
from common import stats


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

    ok, err = allreduce_jax._init_distributed()
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    n_local = len(jax.local_devices())
    world_size = len(jax.devices())
    dtype = getattr(jnp, dtype_name, jnp.bfloat16)

    def loss_fn(w, x):
        return jnp.sum(jnp.dot(x, w))

    grad_fn = jax.value_and_grad(loss_fn)

    @functools.partial(jax.pmap, axis_name="devices")
    def step_fn(w, x):
        _, grads = grad_fn(w, x)
        grads = jax.lax.pmean(grads, axis_name="devices")
        w = w - jnp.array(1.0e-3, dtype=dtype) * grads
        return w

    w = jnp.ones((n_local, input_size, output_size), dtype=dtype)
    x = jnp.ones((n_local, batch_size, input_size), dtype=dtype)

    # trigger compilation
    step_fn(w, x)[0].block_until_ready()

    def _step():
        nonlocal w
        w = step_fn(w, x)
        w[0].block_until_ready()

    for _ in range(max(warmup, 0)):
        _step()

    step_times = []
    for _ in range(max(iters, 1)):
        start = time.perf_counter()
        _step()
        end = time.perf_counter()
        step_times.append((end - start) * 1000.0)

    step_times_sorted = sorted(step_times)
    mid = len(step_times_sorted) // 2
    p50 = step_times_sorted[mid]
    p95 = step_times_sorted[int(0.95 * (len(step_times_sorted) - 1))]
    avg = sum(step_times) / len(step_times)

    global_batch = batch_size * world_size
    samples_per_sec = (global_batch / (avg / 1000.0)) if avg > 0 else 0.0

    return {
        "batch_size": batch_size,
        "input_size": input_size,
        "output_size": output_size,
        "dtype": dtype_name,
        "world_size": world_size,
        "step_time_ms_avg": avg,
        "step_time_ms_p50": p50,
        "step_time_ms_p95": p95,
        "samples_per_sec": samples_per_sec,
    }
