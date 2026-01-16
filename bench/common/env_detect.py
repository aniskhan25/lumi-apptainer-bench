import os
import shutil
import socket
import subprocess


def hostname_list():
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        return [nodelist]
    return [socket.gethostname()]


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
