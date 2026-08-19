**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** `fi_info` and `fi_pingpong` are broken in the `full` and `plus` images — Intel MPI shims shadow the working binaries

---

## Summary

`fi_info` looks uninstalled in `full` and `plus`. It isn't — `/usr/bin/fi_info` works. A wrapper
from the pip package `impi-rt` sits in `/opt/venv/bin`, which is first on `PATH`, and shadows it.

## Reproduce

```bash
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif

singularity exec "$SIF" fi_info --version            # broken
singularity exec "$SIF" /usr/bin/fi_info --version   # works
```

```
/opt/venv/bin/fi_info: line 34: /opt/mpi/libfabric/bin/fi_info: No such file or directory
/usr/bin/fi_info: 2.1.0
libfabric: 2.1.0
```

## Cause

`/opt/venv/bin/fi_info` ends with `exec "$I_MPI_ROOT/opt/mpi/libfabric/bin/$bin_name" "$@"`.
`I_MPI_ROOT` is unset, so it resolves to a path that does not exist in this image.

The wrapper comes from `impi-rt` 2021.18.1, pulled in transitively by `oneccl` 2022.1.1
(`oneccl requires impi-rt>=2021.18`).

## Scope

`full` and `plus` only — the two variants with the venv. `libfabric`, `mpich` and `torch` all
return `libfabric: 2.1.0`. `fi_pingpong` is shadowed the same way.

## Fix

`rm -f /opt/venv/bin/fi_info /opt/venv/bin/fi_pingpong` at build time, or drop `oneccl` if it is
not needed. Workaround for users: call `/usr/bin/fi_info`.

Note `impi-rt` also causes a second, more serious problem — `mpi4py` running on Intel MPI instead
of MPICH, with no `cxi` provider. Filed separately; same root cause.
