#!/usr/bin/env python3
import json
import os
import sys

from datetime import datetime, timezone


METRICS = (
    {
        "name": "single_gemm_tflops",
        "path": ("tests", "single", "gemm", "tflops"),
        "threshold_env": "BENCH_REGRESS_GEMM_PCT",
        "default_threshold": 10.0,
        "regression_mode": "drop",
        "threshold_label": "gemm_drop_pct",
    },
    {
        "name": "single_kernel_mix_p50_ms",
        "path": ("tests", "single", "kernel_mix", "latency_p50_ms"),
        "threshold_env": "BENCH_REGRESS_LATENCY_PCT",
        "default_threshold": 15.0,
        "regression_mode": "increase",
        "threshold_label": "latency_increase_pct",
    },
    {
        "name": "multi_allreduce_bw_avg_gbps",
        "path": ("tests", "multi", "allreduce", "bandwidth_gbps"),
        "reducer": "avg",
        "threshold_env": "BENCH_REGRESS_ALLREDUCE_BW_PCT",
        "default_threshold": 10.0,
        "regression_mode": "drop",
        "threshold_label": "allreduce_bw_drop_pct",
    },
    {
        "name": "multi_allreduce_lat_avg_us",
        "path": ("tests", "multi", "allreduce", "latency_us"),
        "reducer": "avg",
        "threshold_env": "BENCH_REGRESS_LATENCY_PCT",
        "default_threshold": 15.0,
        "regression_mode": "increase",
        "threshold_label": "latency_increase_pct",
    },
    {
        "name": "ddp_samples_per_sec",
        "path": ("tests", "ddp_step", "samples_per_sec"),
        "threshold_env": "BENCH_REGRESS_DDP_SAMPLES_PCT",
        "default_threshold": 10.0,
        "regression_mode": "drop",
        "threshold_label": "ddp_samples_drop_pct",
    },
    {
        "name": "ddp_step_time_ms_avg",
        "path": ("tests", "ddp_step", "step_time_ms_avg"),
        "threshold_env": "BENCH_REGRESS_DDP_LATENCY_PCT",
        "default_threshold": 15.0,
        "regression_mode": "increase",
        "threshold_label": "ddp_latency_increase_pct",
    },
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_value(payload, path):
    value = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def avg(values):
    if not values:
        return None
    return sum(values) / len(values)


def normalize_value(payload, metric):
    value = get_value(payload, metric["path"])
    if metric.get("reducer") == "avg":
        return avg(value)
    return value


def pct_delta(old, new):
    if old is None or new is None:
        return None
    try:
        old_value = float(old)
        new_value = float(new)
    except (TypeError, ValueError):
        return None
    if old_value == 0:
        return None
    return (new_value - old_value) / old_value * 100.0


def threshold(metric):
    return float(
        os.environ.get(metric["threshold_env"], str(metric["default_threshold"]))
    )


def is_regression(delta_pct, metric):
    if delta_pct is None:
        return None
    limit = threshold(metric)
    if metric["regression_mode"] == "drop":
        return delta_pct < -limit
    return delta_pct > limit


def compare_metric(old_payload, new_payload, metric):
    old_value = normalize_value(old_payload, metric)
    new_value = normalize_value(new_payload, metric)
    delta_pct = pct_delta(old_value, new_value)
    return {
        "old": old_value,
        "new": new_value,
        "delta_pct": delta_pct,
        "regression": is_regression(delta_pct, metric),
    }


def compare_results(old_path, new_path):
    old_payload = load_json(old_path)
    new_payload = load_json(new_path)
    metrics = {
        metric["name"]: compare_metric(old_payload, new_payload, metric)
        for metric in METRICS
    }
    regressions = [
        name for name, result in metrics.items() if result.get("regression") is True
    ]
    thresholds = {
        metric["threshold_label"]: threshold(metric)
        for metric in METRICS
    }
    return {
        "run_id": old_payload.get("run_id", ""),
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "old_results": old_path,
        "new_results": new_path,
        "metrics": metrics,
        "regressions": regressions,
        "regression_count": len(regressions),
        "thresholds": thresholds,
    }


def main(argv):
    if len(argv) != 4:
        raise SystemExit(
            "usage: compare_results.py <old_results.json> <new_results.json> <delta.json>"
        )
    old_path, new_path, out_path = argv[1:4]
    payload = compare_results(old_path, new_path)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv)
