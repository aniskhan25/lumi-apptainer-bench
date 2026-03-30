import time
from tests import distributed


def run_ddp_step(
    batch_size=64,
    input_size=4096,
    output_size=4096,
    warmup=3,
    iters=10,
    dtype_name="bfloat16",
):
    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError:
        return {"error": "torch not available"}

    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    ok, err = distributed.init_process_group(torch)
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    device_index = distributed.local_cuda_index(torch)
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_name, torch.bfloat16)

    model = torch.nn.Linear(input_size, output_size, bias=False).to(
        device=device,
        dtype=dtype,
    )
    model = DDP(model, device_ids=[device_index])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    x = torch.randn(batch_size, input_size, device=device, dtype=dtype)

    def step():
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    try:
        for _ in range(max(warmup, 0)):
            step()

        step_times = []
        for _ in range(max(iters, 1)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            end = time.perf_counter()
            step_times.append((end - start) * 1000.0)

        step_times_sorted = sorted(step_times)
        mid = len(step_times_sorted) // 2
        p50 = step_times_sorted[mid]
        p95_index = int(0.95 * (len(step_times_sorted) - 1))
        p95 = step_times_sorted[p95_index]
        avg = sum(step_times) / len(step_times)

        world_size = dist.get_world_size()
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
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
