"""Startup capability probe.

Reports what the job actually loaded rather than what the image was intended to
contain. Two things make this worth a dedicated test:

- A container release ships a package manifest, but the manifest describes the image,
  not the process. Bind mounts, module loads and the launcher all change what a rank
  really sees.
- `apptainer exec` does not run an image's ENTRYPOINT, only `apptainer run` does. The
  lumi-multitorch entrypoint is where ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES get
  set, so under `exec` a rank can silently see all 8 GCDs instead of one. The
  visible-device counts below are what make that observable.
"""

import os
import socket

from common import env_detect


def _visibility():
    """Per-rank device visibility -- the signal for the exec-vs-run entrypoint question."""
    view = {
        "hostname": socket.gethostname(),
        "rank": os.environ.get("SLURM_PROCID", os.environ.get("RANK", "")),
        "local_id": os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "")),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
    }
    try:
        import torch

        view["torch_device_count"] = torch.cuda.device_count()
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "arch": getattr(props, "gcnArchName", ""),
                    "total_mem_gib": round(props.total_memory / 1024**3, 2),
                }
            )
        view["devices"] = devices
    except Exception as exc:  # noqa: BLE001
        view["error"] = f"{type(exc).__name__}: {exc}"
    return view


def run_probe():
    tool, code, out = env_detect.gpu_info()
    return {
        "packages": env_detect.package_versions(),
        "fabric": env_detect.fabric_info(),
        "visibility": _visibility(),
        "cache_paths": env_detect.cache_paths(),
        "tmp": env_detect.tmp_is_node_local(),
        "mounts": env_detect.mount_points(),
        "limits": env_detect.limits(),
        "allocator": {
            "expandable_segments": env_detect.expandable_segments_supported(),
            "conf": os.environ.get("PYTORCH_HIP_ALLOC_CONF", ""),
        },
        "environment": env_detect.tracked_env(),
        "hostnames": env_detect.expanded_hostnames(),
        "gpu_info_tool": tool,
        "gpu_info_exit_code": code,
        "gpu_info_snippet": "\n".join(out.splitlines()[:20]),
    }
