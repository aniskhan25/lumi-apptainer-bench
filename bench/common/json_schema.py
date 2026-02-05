SCHEMA_VERSION = "1.0"

# Minimal schema definition used by the bench output writer.
SCHEMA = {
    "schema_version": "string",
    "run_id": "string",
    "timestamp_utc": "string (RFC3339)",
    "container": {
        "image_path": "string",
        "image_digest": "string (optional)",
    },
    "slurm": {
        "job_id": "string",
        "nodes": "int",
        "ntasks": "int",
        "ntasks_per_node": "int",
        "gpus_per_node": "int",
        "cpus_per_task": "int",
        "distribution": "string",
        "cpu_bind": "string",
        "mpi_mode": "host|container",
    },
    "system": {
        "hostname_list": "string[]",
        "partition": "string",
        "rocm_version": "string",
        "gpu_count": "int",
    },
    "paths": {
        "cache_root": "string",
        "results_dir": "string",
    },
    "tests": {
        "check": {
            "status": "pass|fail",
            "details": "object",
        },
        "single": {
            "gemm": {
                "dtype": "string",
                "tflops": "float",
                "latency_p50_ms": "float",
                "latency_p95_ms": "float",
            },
            "kernel_mix": {
                "latency_p50_ms": "float",
                "latency_p95_ms": "float",
            },
        },
        "multi": {
            "allreduce": {
                "message_sizes_bytes": "int[]",
                "bandwidth_gbps": "float[]",
                "latency_us": "float[]",
                "checksum": "string",
            }
        },
        "ddp_step": {
            "batch_size": "int",
            "input_size": "int",
            "output_size": "int",
            "dtype": "string",
            "world_size": "int",
            "step_time_ms_avg": "float",
            "step_time_ms_p50": "float",
            "step_time_ms_p95": "float",
            "samples_per_sec": "float",
        },
    },
    "optional": {
        "git_rev": "string",
        "template_name": "string",
        "template_version": "string",
        "warnings": "string[]",
        "notes": "string",
    },
}
