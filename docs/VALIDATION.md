# Validating a container against the release gates

Checks one container; no old/new pairing. With a single image under test there is no
baseline, so metrics are compared against declared limits in `manifests/gates/`.

## Run it

```bash
export PROJECT_NAME=project_462000131

# Some nodes hang indefinitely during RCCL init. Always exclude the known-bad list.
export EXCLUDE_NODES=nid005003,nid005004,nid005161,nid005162,nid006315,nid007812,\
nid007679,nid007680,nid007405,nid007406,nid005519,nid005805,nid005761,nid005783,nid005784

./scripts/run_validation.sh              # Phase 1 (1 node) + Phase 2 (2 nodes)
PHASE=1 ./scripts/run_validation.sh      # capability probe only, ~minutes
PHASE=3 ./scripts/run_validation.sh      # 4-node EP=32
```

Exit code is the number of failed gate groups. Results and per-gate verdicts land in
`$RESULTS_ROOT/*.json` and `$RESULTS_ROOT/*.gates.json`.

## Phases

| Phase | Nodes | What it establishes |
| --- | --- | --- |
| 1 | 1 | Capability probe: versions, fabric visibility, cache placement, per-rank GPU visibility, allocator support |
| 2 | 2 | All-to-all at EP=8 (intra-node XGMI control) and EP=16 (crosses Slingshot) |
| 3 | 4 | All-to-all at EP=32 the configuration reported to fail during init |
| 4 | 8–16 | JIT cache under rank pressure, `torch.compile` stress, sustained stability |

Order matters. The EP=8 intra-node case is the control: a cross-node number means nothing
until the intra-node exchange is known correct.

## Individual tests

```bash
# Capability probe
./templates/probe.sh /appl/local/laifs/containers/lumi-multitorch-latest.sif

# All-to-all, one node, EP=8
NODES=1 GROUP_SIZE=8 ./templates/alltoall_sweep.sh <container.sif>

# All-to-all, four nodes, EP=32
NODES=4 GROUP_SIZE=32 ./templates/alltoall_sweep.sh <container.sif>

# Evaluate any results file against a gate spec
python3 bench/scripts/eval_gates.py results.json manifests/gates/probe.json gates.json
```

## The three launch arms

`apptainer exec` does not run an image's ENTRYPOINT; only `apptainer run` does. The
lumi-multitorch entrypoint is where `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` get
set, so the launch mode changes what a rank can see. Compare
`tests.probe.visibility.torch_device_count`:

```bash
# launcher binds GPUs (this repo's default)
USE_ROCR_VISIBLE_DEVICES=1 APPTAINER_MODE=exec ./templates/probe.sh <container.sif>

# nobody binds -- expect 8 devices per rank if the hypothesis in F3 holds
USE_ROCR_VISIBLE_DEVICES=0 APPTAINER_MODE=exec ROCR_USE_SLURM_LOCALID=1 \
  ./templates/probe.sh <container.sif>

# container entrypoint binds
USE_ROCR_VISIBLE_DEVICES=0 APPTAINER_MODE=run ROCR_USE_SLURM_LOCALID=1 \
  ./templates/probe.sh <container.sif>
```

## Cache placement

`LAIF_CACHE_MODE` controls where Triton, Inductor, torch-extension and MIOpen caches live.
Paths are keyed by container so an incompatible image cannot reuse another's artifacts.

| Mode | Location | Use |
| --- | --- | --- |
| `tmp` (default) | `/tmp/laif-cache-$USER/<container-id>/` | Safe at any scale |
| `lustre` | `$SCRATCH/$USER/laif-cache/<container-id>/` | Deliberately reproduce the 64+-rank corruption |

Cache writes are not atomic on Lustre. At 64+ ranks a rank reads a half-written entry,
raises `JSONDecodeError`, exits, and the remaining ranks hang at the next collective, a
failure whose message never names its cause.

## Reading the gate output

```json
{
  "passed": false,
  "failures": ["actually_crossed_nodes"],
  "missing": [],
  "failure_count": 1,
  "gates": {
    "actually_crossed_nodes": {
      "status": "fail",
      "value": false,
      "detail": "expected True, got False",
      "note": "Guards the test itself..."
    }
  }
}
```

Two things to know about the semantics:

- **A missing value counts as a failure**, not as a neutral result. An absent measurement
  is not evidence of a healthy container. Mark a gate `optional` in the spec if it is
  genuinely allowed to be absent. This differs from `compare_results.py`, which reports
  `null` for missing metrics and excludes them from the regression list.
- **Bandwidth gates reduce with `last`** (the largest message size), not `avg`. Averaging
  across a size sweep lets latency-bound small messages dominate the mean.

Gates whose `note` says `PROVISIONAL` are placeholders pending a measured baseline. The
correctness gates are enforceable now because they need no calibration; the throughput
limits do not, and should be recalibrated once Phase 2 and Phase 4 have produced numbers.

## Guard gates

Some gates check the test rather than the container, `actually_crossed_nodes`,
`stayed_intra_node`, `has_zero_token_peers`. They exist because the failure this suite is
built to catch is a run that *looks* like it passed: the reported regression survived
validation precisely because a small test can complete successfully without ever
exercising the pattern that breaks. A gate suite that cannot fail has not been shown to
detect anything.
