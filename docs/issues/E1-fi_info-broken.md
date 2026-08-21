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

## What the shadowing costs

The hidden binary is not merely present, it is fully functional. Inside the container on a GPU node
it enumerates every CXI NIC:

```console
$ srun -N1 -n1 --gpus-per-node=1 singularity exec "$SIF" /usr/bin/fi_info -p cxi
provider: cxi
    fabric: cxi
    domain: cxi0
... (cxi0, cxi1, cxi2, cxi3 — all 4 found)

$ srun ... singularity exec "$SIF" fi_info -p cxi          # what a user actually gets
/opt/venv/bin/fi_info: line 34: /opt/mpi/libfabric/bin/fi_info: No such file or directory
```

**Severity, stated honestly:** this is not critical. No workload fails, and there is no performance
effect — `fi_info` is a diagnostic, so it only matters once something else has gone wrong. The
argument for fixing it is the ratio: the fix is one line, the risk is nil, and the failure falls on
the first command anyone runs when investigating the fabric — the same subject as open issues #20,
#28 and #30. A user hitting it reasonably concludes the libfabric tools were not shipped, and then
has no way to enumerate providers from inside the container.

## It also disables automated fabric checks

Not just a human inconvenience. Running our validation gates against
`20260807_115122` produced one failure and one silent gap:

```
FAIL     fi_info_runs           expected True, got False
skipped  cxi_provider_visible   None
```

The provider-enumeration check gets its data from `fi_info -p cxi`, so when the shim shadows the
working binary that gate cannot run at all. A suite in this state reports one failure and quietly
loses fabric coverage. Any release test that enumerates providers this way has the same blind spot
on `full` and `plus`.

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
