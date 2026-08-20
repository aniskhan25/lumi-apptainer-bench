**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** Shipped torch 2.10.0 predates the upstream fix for the Inductor cache temp-file race (pytorch#172144)

---

## Summary

The image ships PyTorch `2.10.0`. Upstream fixed a race in the Inductor FX graph cache in
`pytorch#172144` (merged January 2026), and that fix is **not** in the 2.10 line — it is in 2.11.
The unfixed code is what causes a failure we reproduced at 128 ranks.

## The fix, and where it is

`GuardedCache.iterate_over_candidates` lists the cache subdirectory and `open()`s every entry,
including the `.{pid}.{tid}.tmp` files that `write_atomic` creates in that same directory. Upstream's
fix skips them:

```python
for path in sorted(os.listdir(subdir)):
    if path.startswith("."):
        continue  # Skip temp files from concurrent write_atomic() calls
```

| Ref | Has the fix? |
| --- | --- |
| `v2.10.0` | **no** |
| `release/2.10` | **no** |
| `v2.11.0` | yes |
| `release/2.11` | yes |
| `main` | yes |

## Verify in the shipped image

```bash
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif

singularity exec "$SIF" python3 -c "import torch; print(torch.__version__, torch.version.git_version[:12])"
singularity exec "$SIF" grep -c 'Skip temp files from concurrent write_atomic' \
  /opt/venv/lib/python3.12/site-packages/torch/_inductor/codecache.py
```

```
2.10.0+rocm7.0 449b17684101
0                                  <- fix absent
```

## What it costs

128 ranks (16 nodes x 8), 24 forced Inductor compilations per rank, cache on a shared Lustre
directory:

- 80 `FileNotFoundError` warnings across 9 ranks, all naming another rank's `.tmp` file
- **1 rank of 128** escalated to a hard `InductorError` and did not recover, which in a distributed
  job hangs the remaining ranks at the next collective
- ~1.5x slower compile than the node-local control (42.0-43.2 s vs 28.3-28.5 s per rank)

Jobs 20684269 / 20684441, measured on `20260513_121430`. Upstream's own report of this race came
from 512-GPU distributed training, so the scale dependence matches.

**A single-node reproduction does not work.** 8 ranks on one node, 12 forced compilations each,
shared Lustre cache, cold: all 8 ranks finished clean with zero warnings. The race needs many
concurrent writers, which is consistent with upstream seeing it at 512 GPUs. So the cheap check is
the one-command grep above, not a reproduction — the race itself is already established upstream and
does not need re-proving.

Note the symptom differs slightly from upstream's: they saw `pickle data was truncated` from reading
a partially written temp file, we saw `FileNotFoundError` from the writer renaming it away between
`listdir` and `open`. Same race, different point in the writer's sequence, same fix.

## Request

Pick up `pytorch#172144` in the LUMI torch build — it is a two-line change — or state that it lands
when the images move to 2.11.

Until then the practical mitigation is cache placement, which is the subject of the separate issue on
JIT cache defaults: with the cache on node-local storage the directory is shared by 8 ranks instead of
128 and the write-to-rename window is far narrower. That issue's urgency drops once this fix is in.

## Environment

- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`
  (digest `d70ec87fda17e97ff3b3241bcb34774365bba5f7b9172a22b8fda0897213bc81`)
- PyTorch `2.10.0+rocm7.0`, git `449b17684101`, Triton 3.6.0
- Reproduction measured on `20260513_121430`, LUMI `standard-g`, 16 nodes, 8 ranks/node
