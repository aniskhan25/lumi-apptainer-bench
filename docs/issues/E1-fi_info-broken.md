**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** `fi_info` is a broken wrapper in all variants of `u24r70f21m50t210-20260513_121430`

---

## Summary

`fi_info` is present on `PATH` in every variant of the `20260513_121430` release, but it is a
wrapper script that `exec`s `/opt/mpi/libfabric/bin/fi_info`, which is not installed in the
image. Every invocation fails with exit 127 and a message that names neither libfabric nor the
real problem.

The libfabric *library* is present and healthy — only the command-line tools are missing from
the path their wrappers expect.

## Reproducer

```bash
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

D=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430
singularity exec $D/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
  fi_info --version
```

Observed:

```
/opt/venv/bin/fi_info: line 34: /opt/mpi/libfabric/bin/fi_info: No such file or directory
rc=127
```

## Affected variants

All four checked; the wrapper location differs but the missing target is the same
(`/opt/mpi/libfabric/bin/fi_info`):

| Variant | Wrapper on `PATH` | Target present |
| --- | --- | --- |
| `libfabric` | `/usr/bin/fi_info` | no |
| `mpich` | `/usr/bin/fi_info` | no |
| `torch` | `/usr/bin/fi_info` | no |
| `full` | `/opt/venv/bin/fi_info` | no |

## What *is* present

```
/usr/lib/x86_64-linux-gnu/libfabric.so
/usr/lib/x86_64-linux-gnu/libfabric.so.1
/usr/lib/x86_64-linux-gnu/libfabric.so.1.27.0
```

`/opt/cray` is not bind-mounted by `lumi-aif-singularity-bindings` (it binds
`/var/spool/slurmd,/pfs,/scratch,/projappl,/project,/flash,/appl`), so the host's Cray
libfabric tools are not reachable as a fallback either. The container's own tooling is the only
route in.

## Why this matters

`fi_info` is the first tool anyone reaches for when diagnosing a fabric problem — enumerating
providers, confirming the `cxi` provider is visible, checking the libfabric version actually in
use. Issues #20, #28 and #30 are all fabric-adjacent, so this is the diagnostic users most need
and currently cannot run. Because the wrapper exists on `PATH`, `command -v fi_info` succeeds
and users reasonably conclude the tool is available.

It also affects automated validation: a capability probe that checks for the tool's presence
rather than its exit status will report a healthy fabric stack.

## Environment

- Image: `lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif`
- Digest: `f0de72f48d1213e1a1a96523382896a4e0b0807c55155fdecd91de29529358d4`
- System: LUMI, `standard-g` / `dev-g`
- Also reproduced in the `libfabric`, `mpich` and `torch` variants of the same release

## Suggested fix

Install the libfabric utilities at the path the wrappers expect, or drop the wrappers if the
tools are intentionally excluded — failing loudly at build time would be better than a wrapper
pointing at nothing. A build-time smoke check (`fi_info --version` returning 0) would catch
this class of packaging error.
