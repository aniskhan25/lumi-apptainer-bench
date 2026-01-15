from common import stats


def run_kernel_mix(size, warmup=2, iters=5, softmax_fp32=True):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return {"error": "torch not available"}

    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    device = torch.device("cuda")
    hidden = max(size, 256)
    batch = max(hidden // 16, 16)

    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    x = torch.randn(batch, hidden, device=device, dtype=dtype)
    residual = torch.randn(batch, hidden, device=device, dtype=dtype)
    w = torch.randn(hidden, hidden, device=device, dtype=dtype)

    def _op():
        y = torch.matmul(x, w)
        y = F.layer_norm(y, (hidden,))
        if softmax_fp32:
            y = y.float()
            y = F.softmax(y, dim=-1)
            y = F.gelu(y)
            y = y + residual.float()
            y = torch.mean(y)
            y = y.to(dtype)
        else:
            y = F.softmax(y, dim=-1)
            y = F.gelu(y)
            y = y + residual
            y = torch.mean(y)
        torch.cuda.synchronize()
        return y

    timings = stats.timeit(_op, warmup=warmup, iters=iters)
    p50 = timings["p50_s"]
    p95 = timings["p95_s"]

    return {
        "latency_p50_ms": p50 * 1000 if p50 else None,
        "latency_p95_ms": p95 * 1000 if p95 else None,
        "size": size,
        "batch": batch,
        "hidden": hidden,
    }
