import os
import shutil
import subprocess


DEFAULT_MASTER_PORT = "29500"
DEFAULT_BACKEND = "nccl"


def env_int(name, default):
    value = os.environ.get(name, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def backend():
    return os.environ.get("BENCH_DIST_BACKEND", DEFAULT_BACKEND)


def first_host_from_nodelist(nodelist):
    if not nodelist:
        return None
    if "[" not in nodelist:
        return nodelist.split(",")[0]
    prefix = nodelist.split("[", 1)[0]
    inside = nodelist.split("[", 1)[1].split("]", 1)[0]
    first = inside.split(",", 1)[0].split("-", 1)[0]
    return f"{prefix}{first}"


def master_addr_from_slurm():
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
    return first_host_from_nodelist(nodelist)


def init_process_group(torch_mod):
    if torch_mod.distributed.is_initialized():
        return True, ""

    rank = env_int("RANK", env_int("SLURM_PROCID", -1))
    world_size = env_int("WORLD_SIZE", env_int("SLURM_NTASKS", -1))
    if rank < 0 or world_size < 1:
        return False, "missing rank/world"

    if not os.environ.get("MASTER_ADDR"):
        master_addr = master_addr_from_slurm()
        if not master_addr:
            return False, "missing MASTER_ADDR"
        os.environ["MASTER_ADDR"] = master_addr
    if not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = DEFAULT_MASTER_PORT

    try:
        torch_mod.distributed.init_process_group(
            backend=backend(),
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:
        return False, str(exc)
    return True, ""


def local_cuda_index(torch_mod):
    device_count = torch_mod.cuda.device_count()
    if device_count <= 1:
        return 0
    local_rank = env_int("LOCAL_RANK", env_int("SLURM_LOCALID", 0))
    return local_rank % device_count
