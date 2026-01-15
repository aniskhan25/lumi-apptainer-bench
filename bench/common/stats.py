import statistics
import time


def _percentile(values, pct):
    if not values:
        return None
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def timeit(fn, warmup=2, iters=5):
    for _ in range(max(warmup, 0)):
        fn()
    durations = []
    for _ in range(max(iters, 1)):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        durations.append(end - start)
    return {
        "durations_s": durations,
        "p50_s": _percentile(durations, 50),
        "p95_s": _percentile(durations, 95),
        "mean_s": statistics.mean(durations) if durations else None,
    }
