import time
from tests import distributed


def run_allreduce(message_sizes, iters=5):
    try:
        import torch
    except ImportError:
        return {"error": "torch not available"}

    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    ok, err = distributed.init_process_group(torch)
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    device = torch.device("cuda")
    results = {
        "message_sizes_bytes": [],
        "bandwidth_gbps": [],
        "latency_us": [],
        "checksum": "",
    }

    try:
        for size in message_sizes:
            numel = max(size // 4, 1)
            tensor = torch.ones(numel, device=device, dtype=torch.float32)
            torch.cuda.synchronize()

            # Warmup
            for _ in range(2):
                torch.distributed.all_reduce(tensor)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(max(iters, 1)):
                torch.distributed.all_reduce(tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()

            total_time = end - start
            avg_time = total_time / max(iters, 1)
            bandwidth = (size / avg_time) / 1.0e9 if avg_time > 0 else 0.0

            results["message_sizes_bytes"].append(size)
            results["bandwidth_gbps"].append(bandwidth)
            results["latency_us"].append(avg_time * 1.0e6)

        checksum = torch.sum(tensor).item()
        results["checksum"] = f"{checksum:.4f}"
        return results
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
