**Repo:** `pytorch/pytorch`
**Title:** Inductor FX graph cache reader opens other processes' in-flight `.tmp` files, breaking on shared filesystems

---

## Summary

`GuardedCache.iterate_over_candidates` lists the on-disk cache subdirectory and `open()`s every
entry it finds, including the `.{pid}.{thread_id}.tmp` files that `write_atomic` creates in that
same directory. When the writing process completes its rename between the reader's `listdir` and
its `open`, the reader raises `FileNotFoundError`.

On a local filesystem the window is narrow enough that this is rare. On a shared filesystem with
many processes — a multi-node training job with a shared `TORCHINDUCTOR_CACHE_DIR` — the window
is wide and the participant count high, so it happens routinely. Most occurrences are logged and
recovered by recompiling, but we have observed one escalate to a hard `InductorError` that killed
the rank.

## The two code paths

`torch/_inductor/codecache.py`, writer at line 454 — the temp file is created in the target's own
directory:

```python
tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
```

Reader at approximately lines 1031–1044 — no filtering of temp files:

```python
if local:
    subdir = cls._get_tmp_dir_for_key(key)
    if os.path.exists(subdir):
        for path in sorted(os.listdir(subdir)):
            try:
                with open(os.path.join(subdir, path), "rb") as f:
                    content = f.read()
                    yield pickle.loads(content), content
            except Exception:
                log.warning(
                    "fx graph cache unable to load compiled graph",
                    exc_info=True,
                )
```

Any entry beginning with `.` and ending in `.tmp` is by construction an in-flight write by some
process, and should not be a load candidate. The broad `except Exception` means the failure is
usually invisible apart from a warning, which also masks genuine cache corruption.

## Observed

128 processes (16 nodes × 8), 24 distinct `torch.compile` shapes each with
`dynamic=False` and `torch._dynamo.reset()` between shapes, against a cold shared cache on a
Lustre filesystem:

```
[rank81] W torch/_inductor/codecache.py:1040] FileNotFoundError: [Errno 2] No such file or directory:
  '<shared>/inductor/fxgraph/qz/fqzsuavdiy…/.45092.22875271438464.tmp'
[rank35] W torch/_inductor/codecache.py:1040] FileNotFoundError: [Errno 2] No such file or directory:
  '<shared>/inductor/fxgraph/qz/fqzsuavdiy…/.45092.22875271438464.tmp'
```

80 such warnings across 9 distinct processes in one run, distributed over
`inductor/codecache` (70), `inductor/aotautograd` (6) and `inductor/fxgraph` (4).

That several *different* processes report the *same* temp filename is the signature of one
writer's file being seen by many readers — each reader reports the writer's `pid.tid`, not its
own.

With the cache on node-local storage instead (8 processes per directory, `tmpfs`): **zero**
warnings, same workload, same process count overall.

One process escalated beyond the warning to
`InductorError: SubprocException` from the Triton compile worker
(`triton_heuristics.py precompile → _precompile_worker`) and did not recover, which in a
distributed job means the remaining ranks hang at the next collective.

Cold-cache compile time was also ~1.5× higher on the shared filesystem (42–43 s vs 28.3–28.5 s
per process for the same 24 kernels), consistent with a wider write→rename window.

## Environment

- PyTorch `2.10.0+rocm7.0` (vendor build), Triton `3.6.0`, Python 3.12.3
- ROCm 7.0 / HIP `7.0.51831`, AMD MI250X (`gfx90a`)
- Shared cache on Lustre; control on per-node `tmpfs`

Not ROCm-specific as far as I can tell — the paths involved are filesystem and process
behaviour, not backend behaviour.

## Suggested fix

Skip in-flight temp files when enumerating cache candidates, e.g. ignore entries matching the
`write_atomic` temp pattern (leading `.`, trailing `.tmp`) in `iterate_over_candidates`. That is
a small change and removes the race entirely rather than relying on timing.

Two adjacent improvements, if of interest:

- Writing temp files to a sibling directory rather than the candidate directory would make the
  reader immune regardless of filtering.
- Narrowing the `except Exception` — or at least logging `FileNotFoundError` on a `.tmp` entry
  differently from a genuine deserialisation failure — would stop this masking real corruption.

Happy to test a patch on the multi-node setup above.
