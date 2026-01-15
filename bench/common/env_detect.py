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
    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    if rocr_visible:
        return len([v for v in rocr_visible.split(",") if v.strip() != ""])
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


def rocm_info():
    if shutil.which("rocminfo"):
        return run_cmd(["rocminfo"])
    if shutil.which("rocm-smi"):
        return run_cmd(["rocm-smi", "-i"])
    return 127, ""
