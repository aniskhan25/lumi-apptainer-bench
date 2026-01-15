import os
import tempfile

from common import env_detect


def run_check(cache_root):
    details = {}
    status = "pass"

    code, out = env_detect.rocm_info()
    details["rocm_info_exit_code"] = code
    if out:
        details["rocm_info_snippet"] = out.splitlines()[:20]
    if code != 0:
        status = "fail"
        details["rocm_info_error"] = "rocminfo/rocm-smi not available"

    if cache_root:
        try:
            os.makedirs(cache_root, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_root, delete=True):
                pass
            details["cache_root_writable"] = True
        except OSError as exc:
            status = "fail"
            details["cache_root_writable"] = False
            details["cache_root_error"] = str(exc)

    details["rocm_version"] = env_detect.rocm_version()
    details["gpu_count_env"] = env_detect.gpu_count_from_env()

    return {"status": status, "details": details}
