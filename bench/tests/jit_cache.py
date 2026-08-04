"""Concurrent JIT cache stress.

Reproduces the failure in report §4.7: with Triton and TorchInductor caches on Lustre, at
64+ ranks the concurrent writes lose races, a rank reads a partially written entry and
raises JSONDecodeError, that rank dies, and the remaining ranks hang at the next collective.

Two things shape the design:

- **Evidence is written before any collective.** The failure mode is one rank dying and the
  rest hanging, so if this test only reported through a barrier or through rank 0 it would
  produce nothing in exactly the case it exists to catch. Every rank writes its own result
  file to the results directory as soon as it finishes compiling, and the aggregate is
  assembled from those files afterwards. A hang then shows up as per-rank files present with
  no aggregate, which is itself the diagnosis.

- **The cache has to be cold.** A warm cache turns the test into a read-only workload and
  no race is possible. Rank 0 clears the cache directories and every rank waits at a barrier
  before compiling, so all ranks enter compilation simultaneously against an empty cache.

Cache placement comes from LAIF_CACHE_MODE in templates/lumi_common.sh: `lustre` to
reproduce, `tmp` to validate the fix.
"""

import json
import os
import shutil
import time

from tests import distributed


# Distinct shapes so Inductor generates a separate kernel, and therefore a separate cache
# entry, for each one -- more entries means more concurrent writes to race over.
DEFAULT_SHAPES = [256, 384, 512, 640, 768, 896]

CACHE_ENV_KEYS = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "MIOPEN_USER_DB_PATH",
)

# Error classes that indicate a corrupt or half-written cache entry rather than a genuine
# compilation problem. JSONDecodeError is the one named in the report.
CORRUPTION_MARKERS = (
    "JSONDecodeError",
    "Expecting value",
    "Unterminated string",
    "unexpected end of data",
    "TruncatedFileError",
    "BadZipFile",
    "UnpicklingError",
    "EOFError",
    "corrupt",
)


def _cache_dirs():
    return {key: os.environ.get(key, "") for key in CACHE_ENV_KEYS}


def _cache_stats(path):
    if not path or not os.path.isdir(path):
        return {"files": 0, "bytes": 0, "exists": False}
    files = 0
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            files += 1
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return {"files": files, "bytes": total, "exists": True}


def _classify(exc):
    text = f"{type(exc).__name__}: {exc}"
    corrupt = any(marker.lower() in text.lower() for marker in CORRUPTION_MARKERS)
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:600],
        "cache_corruption": corrupt,
    }


# A shape that takes at least this long is treated as having actually compiled rather than
# hit an in-process cache. Real compiles here run 1-10 s; cache hits run ~0.00 s.
COMPILE_SECONDS_THRESHOLD = 0.4


def _compile_shapes(torch, shapes, device):
    """Compile the same function at several shapes, recording per-shape outcome.

    Two settings are load-bearing, and without them this test silently does almost nothing:

    - `dynamic=False` and `automatic_dynamic_shapes=False`. By default Dynamo notices a
      changing dimension after the second recompile and produces a single dynamic kernel
      that serves every later shape. Measured directly: with 24 shapes only the first two
      compiled (9.9 s, 1.8 s) and shapes 3-24 took 0.00 s, so a 24-shape run generated
      exactly as many cache entries as a 6-shape one.
    - `torch._dynamo.reset()` before each shape, which drops the in-process code cache and
      forces the on-disk cache to be re-read. That read is where a partially written entry
      surfaces as JSONDecodeError, so without the reset the test exercises the write path
      but never the read path.

    Together these approximate the reported workload, where a fused activation recompiled on
    every step because tokens-per-expert varied with routing.
    """
    import torch._dynamo

    torch._dynamo.config.automatic_dynamic_shapes = False

    def fn(x, y):
        return torch.nn.functional.gelu(x @ y) + x

    results = []
    for size in shapes:
        entry = {"size": size}
        started = time.perf_counter()
        try:
            # Fresh compile per shape, re-reading the shared on-disk cache each time.
            torch._dynamo.reset()
            compiled = torch.compile(fn, dynamic=False)
            x = torch.randn(size, size, device=device, dtype=torch.bfloat16)
            y = torch.randn(size, size, device=device, dtype=torch.bfloat16)
            out = compiled(x, y)
            torch.cuda.synchronize()
            entry["ok"] = True
            entry["checksum"] = float(out.float().abs().sum().item())
        except Exception as exc:  # noqa: BLE001 - the exception is the measurement
            entry["ok"] = False
            entry["error"] = _classify(exc)
        entry["seconds"] = time.perf_counter() - started
        entry["compiled"] = entry["seconds"] >= COMPILE_SECONDS_THRESHOLD
        results.append(entry)
    return results


def run_jit_cache(shapes=None, clear_cache=True, results_dir=""):
    try:
        import torch
    except ImportError as exc:
        return {"error": f"torch not available: {exc}"}
    if not torch.cuda.is_available():
        return {"error": "cuda/rocm not available"}

    ok, err = distributed.init_process_group(torch)
    if not ok:
        return {"error": f"distributed init failed: {err}"}

    shapes = shapes or DEFAULT_SHAPES
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    dirs = _cache_dirs()

    rank_dir = os.path.join(results_dir or ".", "jit_cache_ranks")
    rank_file = os.path.join(rank_dir, f"rank{rank:04d}.json")

    record = {
        "rank": rank,
        "world_size": world_size,
        "hostname": os.environ.get("SLURMD_NODENAME", ""),
        "cache_mode": os.environ.get("LAIF_CACHE_MODE", ""),
        "cache_dirs": dirs,
    }

    try:
        device_index = distributed.local_cuda_index(torch)
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)

        # Cold cache: rank 0 clears, everyone waits, then all ranks compile at once.
        if clear_cache:
            if rank == 0:
                for path in dirs.values():
                    if path and os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    if path:
                        os.makedirs(path, exist_ok=True)
            torch.distributed.barrier()

        record["cache_before"] = {k: _cache_stats(v) for k, v in dirs.items()}
        started = time.perf_counter()
        record["shapes"] = _compile_shapes(torch, shapes, device)
        record["compile_seconds"] = time.perf_counter() - started
        record["cache_after"] = {k: _cache_stats(v) for k, v in dirs.items()}
        # How many shapes genuinely compiled. If this is far below len(shapes), Dynamo
        # collapsed them into one kernel and the run did not apply the pressure intended.
        record["compilations"] = sum(1 for s in record["shapes"] if s.get("compiled"))
        record["shapes_requested"] = len(record["shapes"])
        record["failed_shapes"] = [s for s in record["shapes"] if not s["ok"]]
        record["corruption_errors"] = [
            s["error"] for s in record["shapes"]
            if not s["ok"] and s["error"].get("cache_corruption")
        ]
        record["ok"] = not record["failed_shapes"]
    except Exception as exc:  # noqa: BLE001
        record["ok"] = False
        record["fatal"] = _classify(exc)

    # Write before the barrier. If a peer has already died, the barrier below will hang and
    # this file is the only evidence that will survive.
    os.makedirs(rank_dir, exist_ok=True)
    with open(rank_file, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)

    # Now the collective that the report says hangs when a rank has been lost.
    barrier_ok = True
    barrier_error = ""
    try:
        torch.distributed.barrier()
    except Exception as exc:  # noqa: BLE001
        barrier_ok = False
        barrier_error = f"{type(exc).__name__}: {exc}"

    aggregate = None
    if rank == 0:
        aggregate = _aggregate(rank_dir, world_size, dirs)
        aggregate["barrier_after_compile_ok"] = barrier_ok
        aggregate["barrier_error"] = barrier_error

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return aggregate if rank == 0 else {"rank_record_written": rank_file}


def _aggregate(rank_dir, world_size, dirs):
    """Assemble the per-rank files. Missing files are the signal, not an inconvenience."""
    records = []
    for rank in range(world_size):
        path = os.path.join(rank_dir, f"rank{rank:04d}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        except (OSError, ValueError):
            continue

    reported = {r["rank"] for r in records}
    missing = [r for r in range(world_size) if r not in reported]
    failed = [r for r in records if not r.get("ok")]
    corrupt = [r for r in records if r.get("corruption_errors")]

    total_compile = [
        r["compile_seconds"] for r in records if "compile_seconds" in r
    ]
    return {
        "world_size": world_size,
        "cache_mode": os.environ.get("LAIF_CACHE_MODE", ""),
        "cache_dirs": dirs,
        "ranks_reporting": len(records),
        "ranks_missing": missing,
        "ranks_failed": sorted(r["rank"] for r in failed),
        "ranks_with_cache_corruption": sorted(r["rank"] for r in corrupt),
        "corruption_examples": [
            e for r in corrupt for e in r["corruption_errors"]
        ][:5],
        "failure_examples": [
            s["error"] for r in failed for s in r.get("failed_shapes", [])
        ][:5],
        "compile_seconds_min": min(total_compile) if total_compile else None,
        "compile_seconds_max": max(total_compile) if total_compile else None,
        # Guards the test: the minimum number of real compilations any rank performed.
        "compilations_min": min(
            (r["compilations"] for r in records if "compilations" in r), default=0
        ),
        "shapes_requested": max(
            (r.get("shapes_requested", 0) for r in records), default=0
        ),
        "cache_files_after": (
            max(
                (
                    r["cache_after"]["TRITON_CACHE_DIR"]["files"]
                    for r in records
                    if "cache_after" in r
                ),
                default=0,
            )
        ),
        # The headline: every rank compiled cleanly and every rank reported.
        "passed": not missing and not failed and not corrupt,
    }
