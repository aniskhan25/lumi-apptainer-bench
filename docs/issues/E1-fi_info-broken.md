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

## Secondary: `impi-rt` also ships a duplicate `libmpi.so.12`

Worth mentioning in the same fix, though low severity. `impi-rt` installs Intel MPI's
`libmpi.so.12` into `/opt/venv/lib`, alongside the system MPICH 5.0.1 at
`/usr/lib/x86_64-linux-gnu/libmpi.so.12`. They share the soname, so import order decides which one
a process gets.

In normal use this is harmless: `import torch` loads the system MPICH first, and `mpi4py` then
binds to MPICH 5.0.1 correctly (verified on a GPU node). No shipped package triggers the other
order — `megatron`, `vllm`, `transformer_engine` and `apex` never reference mpi4py, and `deepspeed`
and `lightning` import torch before they reach it.

It only bites if `mpi4py` is imported before `torch`, in which case Intel's library wins and torch
then fails to load:

```console
$ singularity exec "$SIF" python3 -c "from mpi4py import MPI; import torch"
OSError: /usr/lib/x86_64-linux-gnu/libmpicxx.so.12: undefined symbol: MPIX_Win_create_errhandler_x
```

That ordering is unusual on LUMI, where rank discovery comes from Slurm and collectives go through
RCCL, so this is a latent wart rather than a live problem. Removing `oneccl`/`impi-rt` clears it for
free alongside the `fi_info` fix.
