import os
import resource
import shutil
import socket
import subprocess
import sys


def hostname_list():
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        return [nodelist]
    return [socket.gethostname()]


def expanded_hostnames():
    """Expand SLURM_NODELIST into individual node names.

    hostname_list() returns the unexpanded range string ("nid[001001-001002]"),
    which is not enough to attribute a hang or a slow bootstrap to a specific node.
    """
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        return [socket.gethostname()]
    if shutil.which("scontrol"):
        code, out = run_cmd(["scontrol", "show", "hostnames", nodelist])
        if code == 0 and out:
            return out.splitlines()
    return [nodelist]


def rocm_version():
    for key in ("ROCM_VERSION", "ROCR_VERSION", "ROCM_VERSION_PATH"):
        value = os.environ.get(key)
        if value:
            return value
    return ""


def gpu_count_from_env():
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_GPUS"):
        value = os.environ.get(key)
        if value:
            try:
                return int(str(value).split("(")[0].split(",")[0])
            except ValueError:
                continue
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(key, "")
        if value:
            return len([v for v in value.split(",") if v.strip() != ""])
    return 0


def run_cmd(cmd):
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return completed.returncode, completed.stdout.strip()
    except FileNotFoundError:
        return 127, ""


def gpu_info():
    if shutil.which("rocminfo"):
        code, out = run_cmd(["rocminfo"])
        return "rocminfo", code, out
    if shutil.which("rocm-smi"):
        code, out = run_cmd(["rocm-smi", "-i"])
        return "rocm-smi", code, out
    if shutil.which("nvidia-smi"):
        code, out = run_cmd(["nvidia-smi", "-L"])
        return "nvidia-smi", code, out
    return "", 127, ""


# Environment variables that change collective, fabric, allocator or cache behaviour.
# Recorded verbatim so a run can be replayed, and so a failure can be attributed to a
# setting rather than to the image.
TRACKED_ENV_PREFIXES = ("FI_", "HSA_", "NCCL_", "RCCL_", "MIOPEN_", "TRITON_", "TORCH")
TRACKED_ENV_NAMES = (
    "GPU_MAX_HW_QUEUES",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_USE_SLURM_LOCALID",
    "MAP_HIP_TO_ROCR_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "SINGULARITY_BIND",
    "APPTAINER_BIND",
)


def tracked_env():
    out = {}
    for key, value in os.environ.items():
        if key.startswith(TRACKED_ENV_PREFIXES) or key in TRACKED_ENV_NAMES:
            out[key] = value
    return dict(sorted(out.items()))


def _module_version(name, dist=None):
    """Version of an importable module.

    Uses importlib.metadata first: several packages here (megatron-core among them) expose
    no __version__ on the module, and `__import__("a.b")` returns the top-level package
    rather than the submodule, so reading the attribute off it silently yields "".
    """
    import importlib
    import importlib.metadata

    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any import failure is a reportable absence
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"}

    version = ""
    for candidate in (dist, name.replace(".", "-"), name.split(".")[0]):
        if not candidate:
            continue
        try:
            version = importlib.metadata.version(candidate)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    if not version:
        module = importlib.import_module(name)
        version = str(getattr(module, "__version__", "") or "")
    return {"present": True, "version": version}


def package_versions():
    """Versions of what the job actually loaded, not what the image was meant to hold."""
    versions = {
        "python": sys.version.split()[0],
        "torch": _module_version("torch"),
        "triton": _module_version("triton"),
        "flash_attn": _module_version("flash_attn"),
        "megatron_core": _module_version("megatron.core", dist="megatron-core"),
        "transformer_engine": _module_version("transformer_engine"),
        "apex": _module_version("apex"),
    }
    try:
        import torch

        versions["torch_build"] = {
            "version": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda": getattr(torch.version, "cuda", None),
            "git_version": getattr(torch.version, "git_version", ""),
        }
        try:
            # On ROCm builds this reports the bundled RCCL version.
            versions["rccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
        except Exception as exc:  # noqa: BLE001
            versions["rccl"] = f"unavailable: {type(exc).__name__}: {exc}"
        versions["device_count"] = torch.cuda.device_count()
        if torch.cuda.device_count() > 0:
            versions["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            versions["gcn_arch"] = getattr(props, "gcnArchName", "")
            versions["device_total_mem_gib"] = round(props.total_memory / 1024**3, 2)
    except Exception as exc:  # noqa: BLE001
        versions["torch_build"] = {"error": f"{type(exc).__name__}: {exc}"}
    return versions


def fabric_info():
    """libfabric / CXI provider visibility.

    The CXI provider is what carries RCCL traffic over Slingshot. If `fi_info -p cxi`
    reports nothing, no amount of RCCL tuning will help.
    """
    info = {}
    # Distinguish "the tool is not installed" from "the provider is not there". The full
    # image does not ship fi_info, so a 127 says nothing about the fabric -- and fi_info is
    # exactly the tool a user would reach for when debugging a Portals/CXI error.
    info["fi_info_present"] = bool(shutil.which("fi_info"))
    if info["fi_info_present"]:
        code, out = run_cmd(["fi_info", "-p", "cxi"])
        info["fi_info_cxi_exit_code"] = code
        info["fi_info_cxi_snippet"] = "\n".join(out.splitlines()[:40])
        code, out = run_cmd(["fi_info", "--version"])
        info["libfabric_version"] = out.splitlines()[0] if code == 0 and out else ""
    else:
        info["fi_info_cxi_exit_code"] = None
        info["fi_info_cxi_snippet"] = ""
        info["libfabric_version"] = ""

    # Fall back to the shared library, which is present even when the tool is not.
    libs = []
    for root in ("/opt/cray/libfabric", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"):
        code, out = run_cmd(["find", root, "-maxdepth", "3", "-name", "libfabric.so*"])
        if code == 0 and out:
            libs.extend(out.splitlines())
    info["libfabric_libs"] = libs[:10]
    code, out = run_cmd(["find", "/opt/cray/lib64", "-maxdepth", "2", "-name", "libcxi*"])
    info["cxi_libs"] = out.splitlines()[:10] if code == 0 and out else []
    # aws-ofi-nccl is the RCCL<->libfabric plugin and the only comms-stack package that
    # changed between the April and May 2026 container builds.
    if shutil.which("dpkg-query"):
        code, out = run_cmd(["dpkg-query", "-W", "-f=${Version}", "aws-ofi-nccl"])
        info["aws_ofi_nccl_version"] = out if code == 0 else ""
    else:
        info["aws_ofi_nccl_version"] = ""
    return info


def cache_paths():
    keys = (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
        "TORCH_HOME",
    )
    out = {}
    for key in keys:
        path = os.environ.get(key, "")
        entry = {"path": path, "set": bool(path)}
        if path:
            entry["on_lustre"] = _is_lustre(path)
        out[key] = entry
    return out


def _is_lustre(path):
    """Whether a path lands on Lustre.

    Lustre-backed JIT caches lose atomic-write races at 64+ ranks; per-node /tmp does not.
    """
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not probe:
        return None
    code, out = run_cmd(["stat", "-f", "-c", "%T", probe])
    if code != 0:
        return None
    return out.strip() == "lustre"


def tmp_is_node_local():
    """Per-node /tmp is the safe home for JIT caches; a shared /tmp would defeat that."""
    code, out = run_cmd(["stat", "-f", "-c", "%T", "/tmp"])
    if code != 0:
        return {"filesystem": "", "node_local": None}
    fstype = out.strip()
    return {"filesystem": fstype, "node_local": fstype not in ("lustre", "nfs")}


def limits():
    out = {}
    for name in ("RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_MEMLOCK", "RLIMIT_STACK"):
        try:
            soft, hard = resource.getrlimit(getattr(resource, name))
        except (AttributeError, OSError):
            continue
        out[name] = {"soft": soft, "hard": hard}
    return out


def mount_points():
    """Whether the paths a job needs are actually visible inside the container."""
    checks = {}
    for path in ("/scratch", "/projappl", "/project", "/flash", "/appl", "/tmp", "/pfs"):
        checks[path] = os.path.isdir(path)
    return checks


EXPANDABLE_SEGMENTS_PROBE = (
    "import torch, sys\n"
    "t = torch.empty(1 << 26, dtype=torch.uint8, device='cuda')\n"
    "del t\n"
    "torch.cuda.empty_cache()\n"
    "sys.stderr.write('PROBE_OK\\n')\n"
)


def expandable_segments_supported():
    """Whether PYTORCH_HIP_ALLOC_CONF=expandable_segments actually does anything here.

    The OOM message PyTorch prints recommends enabling this, but on ROCm the allocator
    warns "expandable_segments not supported on this platform"
    (c10/hip/HIPAllocatorConfig.h:40) and cannot compact fragmentation, so users following
    the error text are sent down a dead end.

    Run in a subprocess with the environment variable actually set, because that is the
    only path that parses the allocator config and emits the warning. Calling the
    (deprecated) _set_allocator_settings API in-process does not reach it, and reading its
    return value gives a false "supported".
    """
    env = dict(os.environ)
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", EXPANDABLE_SEGMENTS_PROBE],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"supported": None, "reason": f"{type(exc).__name__}: {exc}"}

    stderr = completed.stderr or ""
    unsupported = "not supported" in stderr
    allocated = "PROBE_OK" in stderr
    if not allocated and completed.returncode != 0:
        return {
            "supported": None,
            "reason": "probe allocation failed",
            "exit_code": completed.returncode,
            "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        }
    return {
        "supported": not unsupported,
        "requested": "expandable_segments:True",
        "warning_seen": unsupported,
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
    }
