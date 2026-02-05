import os
import time
import shutil
import subprocess


def _env_int(name, default):
    value = os.environ.get(name, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env(name, default=""):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _backend():
    return os.environ.get("BENCH_DIST_BACKEND", "nccl")


def _first_host_from_nodelist(nodelist):
    if not nodelist:
        return None
    if "[" not in nodelist:
        return nodelist.split(",")[0]
    prefix = nodelist.split("[", 1)[0]
    inside = nodelist.split("[", 1)[1].split("]", 1)[0]
    first = inside.split(",", 1)[0].split("-", 1)[0]
    return f"{prefix}{first}"


def _master_addr_from_slurm():
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        return None
    if shutil.which("scontrol"):
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.stdout:
            return result.stdout.splitlines()[0].strip()
    return _first_host_from_nodelist(nodelist)


def _init_distributed(torch_mod):
    if torch_mod.distributed.is_initialized():
        return True, ""
    rank = _env_int("RANK", _env_int("SLURM_PROCID", -1))
    world = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", -1))
    if rank < 0 or world < 1:
        return False, "missing rank/world"
    if not os.environ.get("MASTER_ADDR"):
        master_addr = _master_addr_from_slurm()
        if not master_addr:
            return False, "missing MASTER_ADDR"
        os.environ["MASTER_ADDR"] = master_addr
    if not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = "29500"
    backend = _backend()
    try:
        torch_mod.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world
        )
    except Exception as exc:
        return False, str(exc)
    return True, ""


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

    ok, err = _init_distributed(torch)
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_name, torch.bfloat16)

    model = torch.nn.Linear(input_size, output_size, bias=False).to(device=device, dtype=dtype)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    x = torch.randn(batch_size, input_size, device=device, dtype=dtype)

    def step():
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
