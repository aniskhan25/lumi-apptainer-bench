#!/usr/bin/env python3
import json
import sys

from pathlib import Path


RESULT_FILES = {
    "single": "lumi_single.json",
    "ddp": "lumi_ddp.json",
    "single_16r": "lumi_single_16r.json",
    "allreduce": "lumi_allreduce.json",
    "multi": "lumi_multi.json",
    "ddp_2n": "lumi_ddp_2n.json",
    "check": "lumi_check.json",
}

COMPARE_FILES = {
    "check": "lumi_check_compare/delta.json",
    "ddp": "lumi_ddp_compare/delta.json",
    "multi": "lumi_multi_compare/delta.json",
    "ddp_2n": "lumi_ddp_2n_compare/delta.json",
    "single_16r": "lumi_single_16r_compare/delta.json",
}


def load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def avg(values):
    if not values:
        return None
    return sum(values) / len(values)


def single_row(run_name, payload):
    tests = payload.get("tests", {}).get("single", {})
    gemm = tests.get("gemm", {})
    kernel_mix = tests.get("kernel_mix", {})
    slurm = payload.get("slurm", {})
    return {
        "Run": run_name,
        "Nodes": slurm.get("nodes"),
        "Tasks": slurm.get("ntasks"),
        "GPU/node": slurm.get("gpus_per_node"),
        "GEMM TFLOPS": gemm.get("tflops"),
        "GEMM p50 ms": gemm.get("latency_p50_ms"),
        "KernelMix p50 ms": kernel_mix.get("latency_p50_ms"),
        "Timestamp UTC": payload.get("timestamp_utc"),
    }


def ddp_row(run_name, payload):
    tests = payload.get("tests", {}).get("ddp_step", {})
    slurm = payload.get("slurm", {})
    return {
        "Run": run_name,
        "Nodes": slurm.get("nodes"),
        "Tasks": slurm.get("ntasks"),
        "GPU/node": slurm.get("gpus_per_node"),
        "Samples/sec": tests.get("samples_per_sec"),
        "Step avg ms": tests.get("step_time_ms_avg"),
        "Step p95 ms": tests.get("step_time_ms_p95"),
        "Timestamp UTC": payload.get("timestamp_utc"),
    }


def multi_row(run_name, payload):
    allreduce = payload.get("tests", {}).get("multi", {}).get("allreduce", {})
    slurm = payload.get("slurm", {})
    return {
        "Run": run_name,
        "Nodes": slurm.get("nodes"),
        "Tasks": slurm.get("ntasks"),
        "GPU/node": slurm.get("gpus_per_node"),
        "Avg BW (GB/s)": avg(allreduce.get("bandwidth_gbps", [])),
        "Avg Lat (us)": avg(allreduce.get("latency_us", [])),
        "Timestamp UTC": payload.get("timestamp_utc"),
    }


def compare_row(run_name, payload):
    metrics = payload.get("metrics", {})
    return {
        "Run": run_name,
        "GEMM TFLOPS Δ%": metrics.get("single_gemm_tflops", {}).get("delta_pct"),
        "KernelMix p50 Δ%": metrics.get("single_kernel_mix_p50_ms", {}).get("delta_pct"),
        "Allreduce BW Δ%": metrics.get("multi_allreduce_bw_avg_gbps", {}).get("delta_pct"),
        "Allreduce Lat Δ%": metrics.get("multi_allreduce_lat_avg_us", {}).get("delta_pct"),
        "DDP samples Δ%": metrics.get("ddp_samples_per_sec", {}).get("delta_pct"),
        "DDP step Δ%": metrics.get("ddp_step_time_ms_avg", {}).get("delta_pct"),
        "Regression count": payload.get("regression_count"),
        "Timestamp UTC": payload.get("timestamp_utc"),
    }


def print_table(title, columns, rows):
    if not rows:
        return
    print(f"**{title}**")
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        values = [fmt(row.get(column)) for column in columns]
        print("| " + " | ".join(values) + " |")
    print()


def summarize(results_dir):
    base = Path(results_dir)

    single_rows = []
    for key in ("single", "single_16r"):
        payload = load_json(base / RESULT_FILES[key])
        if payload:
            single_rows.append(single_row(RESULT_FILES[key], payload))

    ddp_rows = []
    for key in ("ddp", "ddp_2n"):
        payload = load_json(base / RESULT_FILES[key])
        if payload:
            ddp_rows.append(ddp_row(RESULT_FILES[key], payload))

    multi_rows = []
    for key in ("allreduce", "multi"):
        payload = load_json(base / RESULT_FILES[key])
        if payload:
            multi_rows.append(multi_row(RESULT_FILES[key], payload))

    compare_rows = []
    for key, relative_path in COMPARE_FILES.items():
        payload = load_json(base / relative_path)
        if payload:
            compare_rows.append(compare_row(relative_path, payload))

    print_table(
        "Single-node tests",
        ["Run", "Nodes", "Tasks", "GPU/node", "GEMM TFLOPS", "GEMM p50 ms", "KernelMix p50 ms", "Timestamp UTC"],
        single_rows,
    )
    print_table(
        "DDP tests",
        ["Run", "Nodes", "Tasks", "GPU/node", "Samples/sec", "Step avg ms", "Step p95 ms", "Timestamp UTC"],
        ddp_rows,
    )
    print_table(
        "Multi-node allreduce tests",
        ["Run", "Nodes", "Tasks", "GPU/node", "Avg BW (GB/s)", "Avg Lat (us)", "Timestamp UTC"],
        multi_rows,
    )
    print_table(
        "Comparison deltas",
        [
            "Run",
            "GEMM TFLOPS Δ%",
            "KernelMix p50 Δ%",
            "Allreduce BW Δ%",
            "Allreduce Lat Δ%",
            "DDP samples Δ%",
            "DDP step Δ%",
            "Regression count",
            "Timestamp UTC",
        ],
        compare_rows,
    )


def main(argv):
    results_dir = argv[1] if len(argv) > 1 else "bench_results"
    summarize(results_dir)


if __name__ == "__main__":
    main(sys.argv)
