import os
import time

from common import stats


def _select_dtype(torch_mod, requested):
    if requested:
        return getattr(torch_mod, requested, None)
    if hasattr(torch_mod, "bfloat16") and torch_mod.cuda.is_bf16_supported():
        return torch_mod.bfloat16
    return torch_mod.float16


def run_gemm(size, dtype_name=None, warmup=2, iters=5):
    try:
        import torch
    except ImportError:
        return {"error": "torch not available"}

    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    dtype = _select_dtype(torch, dtype_name)
    if dtype is None:
        return {"error": f"unsupported dtype: {dtype_name}"}

    device = torch.device("cuda")
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    def _op():
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        return c

    # Warmup handled by stats.timeit
    timings = stats.timeit(_op, warmup=warmup, iters=iters)

    p50 = timings["p50_s"]
    p95 = timings["p95_s"]
    if p50:
        tflops = (2 * (size**3)) / p50 / 1.0e12
    else:
        tflops = None

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "tflops": tflops,
        "latency_p50_ms": p50 * 1000 if p50 else None,
        "latency_p95_ms": p95 * 1000 if p95 else None,
        "size": size,
    }
