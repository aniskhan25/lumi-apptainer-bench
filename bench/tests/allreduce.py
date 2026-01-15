import os
import shutil
import subprocess
import time


def _env_int(name, default):
    value = os.environ.get(name, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
        torch_mod.distributed.init_process_group(backend=backend, rank=rank, world_size=world)
    except Exception as exc:
        return False, str(exc)
    return True, ""


def run_allreduce(message_sizes, iters=5):
    try:
        import torch
    except ImportError:
        return {"error": "torch not available"}

    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    ok, err = _init_distributed(torch)
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    device = torch.device("cuda")
    results = {
        "message_sizes_bytes": [],
        "bandwidth_gbps": [],
        "latency_us": [],
        "checksum": "",
    }

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
