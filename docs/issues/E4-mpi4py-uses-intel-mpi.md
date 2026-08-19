**Repo:** `lumi-ai-factory/laifs-container-recipes`
**Title:** `mpi4py` in `full`/`plus` runs on Intel MPI, whose libfabric has no `cxi` provider

---

## Summary

`mpi4py` is built for MPICH but loads Intel MPI at runtime. Intel's bundled libfabric has no `cxi`
provider, so mpi4py cannot use LUMI's Slingshot interconnect.

## Reproduce

```bash
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif

# which MPI does mpi4py actually load?
singularity exec "$SIF" python3 -c "
import mpi4py.MPI as M
print(M.Get_library_version().splitlines()[0])
print([l.split()[-1] for l in open('/proc/self/maps') if 'libmpi' in l][0])"

# providers in each libfabric
singularity exec "$SIF" /usr/bin/fi_info -l | grep -E '^\w+:'      # system
singularity exec "$SIF" ls /opt/venv/lib/prov 2>/dev/null || \
  singularity exec "$SIF" find /opt/venv -name 'lib*-fi.so' -printf '%f\n'   # Intel's
```

```
Intel(R) MPI Library 2021.18.1 for Linux* OS
/opt/venv/lib/libmpi.so

cxi:  ofi_rxm:  ofi_rxd:  shm:  sm2:  lnx:  ...          <- system libfabric 2.1.0, has cxi
libefa-fi.so libmlx-fi.so libpsm3-fi.so libpsmx2-fi.so
librxm-fi.so libshm-fi.so libtcp-fi.so libverbs-*-fi.so  <- Intel's, no cxi
```

## Cause

`oneccl 2022.1.1` requires `impi-rt>=2021.18`, which installs Intel MPI's `libmpi.so.12` into
`/opt/venv/lib`. Intel MPI is MPICH-ABI-compatible, so it shares the `libmpi.so.12` soname and
satisfies mpi4py's link at runtime.

The extension itself is the MPICH build — `mpi4py/MPI.mpich.cpython-312-x86_64-linux-gnu.so`, with
no RPATH, and `ldd` resolves `libmpi.so.12` to `/lib/x86_64-linux-gnu/libmpi.so.12` (MPICH 5.0.1).
Intel's copy still wins in-process, and setting `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` does
not change it.

## Scope

`full` and `plus`. `mpich` and `torch` ship no mpi4py.

## Fix

Drop `oneccl` (and with it `impi-rt`) if it is not needed on this platform — that also fixes the
`fi_info` shadowing filed separately. Otherwise, keep Intel MPI's `libmpi.so*` out of a directory
that gets searched ahead of the system MPICH.

## Not yet verified

I confirmed the loaded library and the provider lists, but have **not** run a multi-node mpi4py job
to show it failing or falling back to TCP. The impact statement above is an inference from the
missing `cxi` provider. A 2-node `mpi4py` bandwidth test with `FI_LOG_LEVEL=info` would confirm
which provider it selects.

## Environment

- `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`
  (digest `d70ec87fda17e97ff3b3241bcb34774365bba5f7b9172a22b8fda0897213bc81`)
- mpi4py 4.1.2, `impi-rt`/`impi-devel` 2021.18.1, `oneccl` 2022.1.1, MPICH 5.0.1, libfabric 2.1.0
