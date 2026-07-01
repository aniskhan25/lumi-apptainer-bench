import time
import functools

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

    ok, err = allreduce_jax._init_distributed()
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    n_total = len(jax.devices())
    n_local = len(jax.local_devices())
    if n_total < 2:
        return {"error": f"need at least 2 total devices, got {n_total}"}

    dtype = getattr(jnp, dtype_name, jnp.bfloat16)

    # ------------------------------------------------------------------ #
    # Verification: each rank scales x by (rank+1) so every device gets a
    # distinct gradient. If pmean is a real cross-process allreduce, rank 0
    # sees the mean of (1..n_total); if it is a local no-op, rank 0 only
    # sees its own gradient (scale=1).
    #
    # grad_w for rank r = batch_size * (r+1) * ones(input, output)
    # correct avg        = batch_size * mean(1..n_total) * ones
    # rank-0 no-op value = batch_size * 1 * ones
    #
    # w_after[0,0] lets us distinguish:
    #   allreduce ok  -> 1.0 - 1e-3 * batch_size * (n_total+1)/2
    #   local no-op   -> 1.0 - 1e-3 * batch_size * 1
    # ------------------------------------------------------------------ #
    @functools.partial(jax.pmap, axis_name="devices")
    def verify_fn(w, x):
        rank = jax.lax.axis_index("devices")
        x_scaled = x * jnp.array(rank + 1, jnp.float32)
        _, grads = jax.value_and_grad(
            lambda w: jnp.sum(jnp.dot(x_scaled, w))
        )(w)
        avg_grads = jax.lax.pmean(grads, "devices")
        return w - jnp.array(1e-3, jnp.float32) * avg_grads

    w_f32 = jnp.ones((n_local, input_size, output_size), jnp.float32)
    x_f32 = jnp.ones((n_local, batch_size, input_size), jnp.float32)
    w_after_verify = verify_fn(w_f32, x_f32)
    actual_w0 = float(w_after_verify[0, 0, 0])

    expected_w0 = 1.0 - 1e-3 * batch_size * (n_total + 1) / 2.0
    noop_w0 = 1.0 - 1e-3 * batch_size * 1.0

    tol = 0.01
    if abs(actual_w0 - expected_w0) < tol:
        allreduce_verified = True
    elif abs(actual_w0 - noop_w0) < tol:
        allreduce_verified = False
    else:
        allreduce_verified = None  # unexpected — partial or wrong collective

    # ------------------------------------------------------------------ #
    # Main benchmark
    # ------------------------------------------------------------------ #
    @functools.partial(jax.pmap, axis_name="devices")
    def step_fn(w, x):
        _, grads = jax.value_and_grad(
            lambda w: jnp.sum(jnp.dot(x, w))
        )(w)
        avg_grads = jax.lax.pmean(grads, "devices")
        return w - jnp.array(1e-3, dtype=dtype) * avg_grads

    w = jnp.ones((n_local, input_size, output_size), dtype=dtype)
    x = jnp.ones((n_local, batch_size, input_size), dtype=dtype)

    def step_and_sync():
        nonlocal w
        w = step_fn(w, x)
        jax.block_until_ready(w)

    step_and_sync()  # compile

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
        "allreduce_verified": allreduce_verified,
        "verify_w0_actual": actual_w0,
        "verify_w0_expected_allreduce": expected_w0,
        "verify_w0_expected_noop": noop_w0,
        "step_time_ms_avg": avg,
        "step_time_ms_p50": p50,
        "step_time_ms_p95": p95,
        "samples_per_sec": samples_per_sec,
    }
