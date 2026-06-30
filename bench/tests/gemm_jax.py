from common import stats


def run_gemm(size, dtype_name=None, warmup=2, iters=5):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return {"error": "jax not available"}

    try:
        device = jax.local_devices()[0]
    except (RuntimeError, IndexError) as exc:
        return {"error": f"no JAX device: {exc}"}

    dtype = getattr(jnp, dtype_name or "bfloat16", jnp.bfloat16)
    a = jax.device_put(jnp.ones((size, size), dtype=dtype), device)
    b = jax.device_put(jnp.ones((size, size), dtype=dtype), device)

    dot_jit = jax.jit(jnp.dot)
    dot_jit(a, b).block_until_ready()  # trigger compilation

    def _op():
        dot_jit(a, b).block_until_ready()

    timings = stats.timeit(_op, warmup=warmup, iters=iters)
    p50 = timings["p50_s"]
    p95 = timings["p95_s"]
    tflops = (2 * size**3) / p50 / 1.0e12 if p50 else None

    return {
        "dtype": dtype_name or "bfloat16",
        "tflops": tflops,
        "latency_p50_ms": p50 * 1000 if p50 else None,
        "latency_p95_ms": p95 * 1000 if p95 else None,
        "size": size,
    }
