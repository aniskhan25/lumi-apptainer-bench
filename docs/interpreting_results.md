# Interpreting results

## Schema summary

The authoritative schema is defined in `bench/common/json_schema.py`.

| Section | Required fields | Notes |
| --- | --- | --- |
| Top-level | `schema_version`, `run_id`, `timestamp_utc` | Strings, `timestamp_utc` is RFC3339. |
| container | `image_path`, `image_digest` | `image_digest` if available. |
| slurm | `job_id`, `nodes`, `ntasks`, `ntasks_per_node`, `gpus_per_node`, `cpus_per_task`, `distribution`, `cpu_bind`, `mpi_mode` | `mpi_mode` is `host` or `container`. |
| system | `hostname_list`, `partition`, `rocm_version`, `gpu_count` | `hostname_list` is array of strings. |
| paths | `cache_root`, `results_dir` | Absolute paths. |
| tests.check | `status`, `details` | Required if test executed. |
| tests.single.gemm | `dtype`, `tflops`, `latency_p50_ms`, `latency_p95_ms` | Required if test executed. |
| tests.single.kernel_mix | `latency_p50_ms`, `latency_p95_ms` | Required if test executed. |
| tests.multi.allreduce | `message_sizes_bytes`, `bandwidth_gbps`, `latency_us`, `checksum` | Required if test executed. |
| optional | `git_rev`, `template_name`, `template_version`, `warnings`, `notes` | Optional metadata. |

For a concrete example, see the sample JSON in `docs/how_to_run.md`.
