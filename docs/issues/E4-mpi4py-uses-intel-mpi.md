**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** `import mpi4py` before `import torch` breaks PyTorch — two MPI libraries with the same soname

---

## Summary

In `full` and `plus`, importing `mpi4py` before `torch` makes PyTorch fail to import. Each import
works fine on its own. Only that order fails, and it fails every time.

## Reproduce

```bash
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif

singularity exec "$SIF" python3 -c "import torch"                        # ok
singularity exec "$SIF" python3 -c "from mpi4py import MPI"              # ok
singularity exec "$SIF" python3 -c "from mpi4py import MPI; import torch"  # fails
```

```
OSError: /usr/lib/x86_64-linux-gnu/libmpicxx.so.12: undefined symbol: MPIX_Win_create_errhandler_x
```

Reproduced 3/3.

## Cause

The image contains two MPI implementations that share the `libmpi.so.12` soname:

| | Path | Version |
| --- | --- | --- |
| Intended | `/usr/lib/x86_64-linux-gnu/libmpi.so.12` | MPICH 5.0.1 |
| Unintended | `/opt/venv/lib/libmpi.so.12` | Intel MPI 2021.18.1 |

`mpi4py` loads the Intel one. `torch` then dlopens the system `libmpicxx.so.12` (MPICH's C++
wrapper), which needs `MPIX_Win_create_errhandler_x` — a symbol MPICH's `libmpi` exports and
Intel's does not. Since Intel's library already occupies the soname, the symbol is unresolvable and
the load fails.

Intel MPI is not requested by the recipe. It arrives transitively:

```
oneccl 2022.1.1  requires: impi-rt (>=2021.18)
```

`impi-rt` installs Intel's `libmpi.so.12` and `libfabric.so` into `/opt/venv/lib`, and its
`fi_info` / `fi_pingpong` wrappers into `/opt/venv/bin` (the latter is the separate `fi_info` issue).

## Why this is worth fixing

- **It is a hard import failure, not a slowdown.** Nothing degrades gracefully.
- **The trigger is import order, which is arbitrary.** `mpi4py` sorts before `torch`
  alphabetically, so running `isort` or `ruff --fix` on a working script can break it.
- **The error message is undiagnosable.** It names `libmpicxx.so.12` and a symbol; nothing points
  at mpi4py, at Intel MPI, or at `oneccl`. There is no realistic path from that message to the
  cause.
- **Nothing wants the Intel library.** It is collateral from one transitive dependency.

Secondary, same root cause: because `mpi4py` runs on Intel MPI, it also gets Intel's bundled
libfabric, which ships `efa/mlx/psm3/psmx2/rxm/shm/tcp/verbs` and **no `cxi`** — so mpi4py cannot
use LUMI's Slingshot interconnect. The system libfabric does have `cxi`.

## Scope, stated plainly

This does **not** affect a pure PyTorch job. Verified: `import torch` alone loads the system MPICH
and the system libfabric, correctly; and neither `torch` nor `deepspeed` imports `mpi4py` on its
own. The trigger requires the user to import `mpi4py` themselves.

So: severe consequence, narrow trigger. `full` and `plus` only — `mpich` and `torch` ship no
mpi4py.

## Fix

Drop `oneccl` if it is not needed on this platform, which removes `impi-rt` and with it both this
and the `fi_info` shadowing. Otherwise keep Intel MPI's `libmpi.so*` out of a directory that is
searched ahead of the system MPICH.

## Environment

- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`
  (digest `d70ec87fda17e97ff3b3241bcb34774365bba5f7b9172a22b8fda0897213bc81`)
- mpi4py 4.1.2, `impi-rt`/`impi-devel` 2021.18.1, `oneccl` 2022.1.1, MPICH 5.0.1, libfabric 2.1.0
- PyTorch 2.10.0+rocm7.0
