#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from common import env_detect, json_schema
from tests import allreduce, check_rocm, gemm_torch, kernel_mix


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _env(name, default=""):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _int_env(name, default=0):
    value = _env(name, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _gpu_count():
    return env_detect.gpu_count_from_env()


def _hostname_list():
    return env_detect.hostname_list()


def _rocm_version():
    return env_detect.rocm_version()


def _run_id():
    return _env("RUN_ID", _utc_now().replace(":", "").replace("-", ""))


def _base_payload():
    return {
        "schema_version": json_schema.SCHEMA_VERSION,
        "run_id": _run_id(),
        "timestamp_utc": _utc_now(),
        "container": {
            "image_path": _env("BENCH_CONTAINER_IMAGE", _env("CONTAINER_IMAGE", "")),
            "image_digest": _env("BENCH_CONTAINER_DIGEST", ""),
        },
        "slurm": {
            "job_id": _env("SLURM_JOB_ID", ""),
            "nodes": _int_env("BENCH_NODES", _int_env("SLURM_NNODES", 0)),
            "ntasks": _int_env("SLURM_NTASKS", 0),
            "ntasks_per_node": _int_env("BENCH_NTASKS_PER_NODE", 0),
            "gpus_per_node": _int_env("BENCH_GPUS_PER_NODE", _int_env("SLURM_GPUS_PER_NODE", 0)),
            "cpus_per_task": _int_env("BENCH_CPUS_PER_TASK", _int_env("SLURM_CPUS_PER_TASK", 0)),
            "distribution": _env("BENCH_DIST", ""),
            "cpu_bind": _env("BENCH_CPU_BIND", ""),
            "mpi_mode": _env("BENCH_MPI_MODE", ""),
        },
        "system": {
            "hostname_list": _hostname_list(),
            "partition": _env("BENCH_PARTITION", _env("SLURM_JOB_PARTITION", "")),
            "rocm_version": _rocm_version(),
            "gpu_count": _gpu_count(),
        },
        "paths": {
            "cache_root": _env("BENCH_CACHE_ROOT", ""),
            "results_dir": _env("BENCH_RESULTS_DIR", ""),
        },
    }


def _write_json(path, payload):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _add_warning(payload, message):
    optional = payload.setdefault("optional", {})
    warnings = optional.setdefault("warnings", [])
    warnings.append(message)


def cmd_check(args):
    payload = _base_payload()
    cache_root = _env("BENCH_CACHE_ROOT", "")
    check = check_rocm.run_check(cache_root)
    payload["tests"] = {"check": check}
    if check.get("status") != "pass":
        _add_warning(payload, "check: failures detected")
    _write_json(args.out, payload)
    return 0 if check.get("status") == "pass" else 1


def cmd_single(args):
    payload = _base_payload()
    gemm = gemm_torch.run_gemm(
        size=args.gemm_size,
        dtype_name=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
    )
    if "error" in gemm:
        _add_warning(payload, f"single: {gemm['error']}")

    mix = kernel_mix.run_kernel_mix(
        size=args.kernel_mix_size,
        warmup=args.warmup,
        iters=args.iters,
        softmax_fp32=args.softmax_fp32,
    )
    if "error" in mix:
        _add_warning(payload, f"single: kernel_mix {mix['error']}")

    payload["tests"] = {
        "single": {
            "gemm": {
                "dtype": gemm.get("dtype", "unknown"),
                "tflops": gemm.get("tflops"),
                "latency_p50_ms": gemm.get("latency_p50_ms"),
                "latency_p95_ms": gemm.get("latency_p95_ms"),
            },
            "kernel_mix": {
                "latency_p50_ms": mix.get("latency_p50_ms"),
                "latency_p95_ms": mix.get("latency_p95_ms"),
            },
        }
    }
    _write_json(args.out, payload)
    return 0


def _parse_sizes(value):
    if not value:
        return []
    sizes = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            sizes.append(int(token))
        except ValueError:
            continue
    return sizes


def cmd_multi(args):
    payload = _base_payload()
    sizes = _parse_sizes(args.message_sizes) or [1024, 4096, 16384, 65536, 262144, 1048576]
    allreduce_result = allreduce.run_allreduce(sizes, iters=args.iters)
    if "error" in allreduce_result:
        _add_warning(payload, f"multi: {allreduce_result['error']}")
        allreduce_payload = {
            "message_sizes_bytes": [],
            "bandwidth_gbps": [],
            "latency_us": [],
            "checksum": "",
        }
    else:
        allreduce_payload = allreduce_result

    payload["tests"] = {"multi": {"allreduce": allreduce_payload}}
    _write_json(args.out, payload)
    return 0


def cmd_compare(args):
    compare_path = os.path.join(os.path.dirname(__file__), "compare.sh")
    cmd = [compare_path] + args.args
    return subprocess.call(cmd)


def build_parser():
    parser = argparse.ArgumentParser(description="LUMI container benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="check benchmark")
    check.add_argument("--out", required=True, help="Output JSON path")

    single = subparsers.add_parser("single", help="single benchmark")
    single.add_argument("--out", required=True, help="Output JSON path")
    single.add_argument("--gemm-size", type=int, default=int(_env("BENCH_GEMM_SIZE", "4096")))
    single.add_argument("--dtype", default=_env("BENCH_GEMM_DTYPE", ""))
    single.add_argument(
        "--kernel-mix-size",
        type=int,
        default=int(_env("BENCH_KERNEL_MIX_SIZE", "2048")),
    )
    softmax_default = _env("BENCH_KERNEL_MIX_SOFTMAX_FP32", "1") != "0"
    softmax_group = single.add_mutually_exclusive_group()
    softmax_group.add_argument(
        "--softmax-fp32",
        dest="softmax_fp32",
        action="store_true",
        help="Run softmax and GELU in fp32 for stability.",
    )
    softmax_group.add_argument(
        "--no-softmax-fp32",
        dest="softmax_fp32",
        action="store_false",
        help="Keep softmax and GELU in the model dtype.",
    )
    single.set_defaults(softmax_fp32=softmax_default)
    single.add_argument("--warmup", type=int, default=int(_env("BENCH_WARMUP", "2")))
    single.add_argument("--iters", type=int, default=int(_env("BENCH_ITERS", "5")))

    multi = subparsers.add_parser("multi", help="multi benchmark")
    multi.add_argument("--out", required=True, help="Output JSON path")
    multi.add_argument("--message-sizes", default=_env("BENCH_ALLREDUCE_SIZES", ""))
    multi.add_argument("--iters", type=int, default=int(_env("BENCH_ITERS", "5")))
    multi.add_argument(
        "--allreduce",
        action="store_true",
        help="Run all-reduce sweep (default behavior).",
    )

    compare = subparsers.add_parser("compare", help="A/B comparison")
    compare.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return cmd_check(args)
    if args.command == "single":
        return cmd_single(args)
    if args.command == "multi":
        return cmd_multi(args)
    if args.command == "compare":
        return cmd_compare(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
