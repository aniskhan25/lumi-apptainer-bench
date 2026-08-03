#!/usr/bin/env python3
"""Evaluate one results.json against absolute release gates.

This replaces the old-vs-new delta model for latest-container validation. With a single
container under test there is no baseline run, so a percentage regression has nothing to
be relative to; the question becomes whether the measured value clears a declared limit.

Two deliberate differences from compare_results.py:

- A missing value counts as a failure, not as a neutral result. compare_results.py
  reports `null` when a metric is absent and excludes it from the regression list, so a
  test that never ran looks the same as a test that passed. Mark a gate `optional` if it
  is genuinely allowed to be absent.
- Bandwidth gates reduce with `last` (the largest message size) rather than `avg`.
  Averaging bandwidth across a size sweep that starts at 16 KB lets latency-bound small
  messages dominate the mean, which makes the average useless as a bandwidth threshold.

Usage:
  eval_gates.py <results.json> <gates.json> <gates_out.json>
"""

import json
import sys

from datetime import datetime, timezone

# get_value and avg are shared with the A/B comparator rather than reimplemented.
from compare_results import avg, get_value, load_json


def _reduce(value, reducer):
    if reducer is None:
        return value
    if not isinstance(value, list) or not value:
        return None
    if reducer == "avg":
        return avg(value)
    if reducer == "last":
        return value[-1]
    if reducer == "max":
        return max(value)
    if reducer == "min":
        return min(value)
    if reducer == "len":
        return len(value)
    raise ValueError(f"unknown reducer: {reducer}")


def _compare(value, gate):
    """Return (status, detail). status is one of pass, fail, missing."""
    kind = gate["type"]

    if kind in ("must_be_true", "must_be_false"):
        if value is None:
            return "missing", "value absent"
        want = kind == "must_be_true"
        if bool(value) is want:
            return "pass", None
        return "fail", f"expected {want}, got {value!r}"

    if kind in ("equals", "contains"):
        if value is None:
            return "missing", "value absent"
        expected = gate["expect"]
        if kind == "equals":
            ok = str(value) == str(expected)
        else:
            ok = str(expected) in str(value)
        if ok:
            return "pass", None
        return "fail", f"expected {kind} {expected!r}, got {value!r}"

    if value is None:
        return "missing", "value absent"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "fail", f"value {value!r} is not numeric"

    limit = float(gate["limit"])
    if kind == "min":
        if numeric >= limit:
            return "pass", None
        return "fail", f"{numeric:.4g} below minimum {limit:.4g}"
    if kind == "max":
        if numeric <= limit:
            return "pass", None
        return "fail", f"{numeric:.4g} above maximum {limit:.4g}"
    raise ValueError(f"unknown gate type: {kind}")


def evaluate(results, spec):
    gates = {}
    for gate in spec["gates"]:
        raw = get_value(results, tuple(gate["path"]))
        value = _reduce(raw, gate.get("reducer"))
        status, detail = _compare(value, gate)
        if status == "missing" and gate.get("optional"):
            status = "skipped"
        gates[gate["name"]] = {
            "status": status,
            "value": value,
            "type": gate["type"],
            "limit": gate.get("limit"),
            "reducer": gate.get("reducer"),
            "path": list(gate["path"]),
            "detail": detail,
            "note": gate.get("note", ""),
        }

    failures = [n for n, g in gates.items() if g["status"] == "fail"]
    missing = [n for n, g in gates.items() if g["status"] == "missing"]
    return {
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "run_id": results.get("run_id", ""),
        "container": results.get("container", {}),
        "slurm": results.get("slurm", {}),
        "spec_name": spec.get("name", ""),
        "gates": gates,
        "failures": failures,
        "missing": missing,
        # Missing values are counted as not-passing: an absent measurement is not
        # evidence of a healthy container.
        "failure_count": len(failures) + len(missing),
        "passed": not failures and not missing,
    }


def main(argv):
    if len(argv) != 4:
        raise SystemExit(
            "usage: eval_gates.py <results.json> <gates.json> <gates_out.json>"
        )
    results_path, spec_path, out_path = argv[1:4]
    payload = evaluate(load_json(results_path), load_json(spec_path))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    status = "PASS" if payload["passed"] else "FAIL"
    print(f"[{status}] {payload['spec_name']} -> {out_path}")
    for name in payload["failures"]:
        print(f"  FAIL    {name}: {payload['gates'][name]['detail']}")
    for name in payload["missing"]:
        print(f"  MISSING {name}: {payload['gates'][name]['path']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
