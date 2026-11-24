# Die-affinity scheduling (percounter)

Measures the cost of spreading one run's clusters across both A5 dies instead of
keeping them on one, at a fixed cluster count K.

## What it runs

`kernels/orchestration/die_rotation_orch.cpp` builds a rotated `add_scalar`
pipeline over one buffer of `(stages+1) * tiles` tiles (each a 128×128 float32
block, the shape the shared `kernel_add_scalar` kernel is fixed to). Stage `s`
writes region `s+1` tile `t` from region `s` tile `(t+1) % tiles` plus one. The
rotation makes a tile produced on one cluster feed a consumer on another, so the
die a consumer runs on matters; within a stage the `tiles` outputs are disjoint,
so they fill the clusters in parallel. After the run, region `r` tile `t` equals
region 0 tile `((t+r) % tiles) + r`.

The scheduling shape comes from `CallConfig`:

- `die_sched_policy`: 0 = all launched clusters, 1 = `DIE_LOCAL` (K clusters on
  die 0), 2 = `CROSS_DIE` (K/2 clusters per die; K must be even).
- `aic_cluster_count`: K, the number of clusters that draw work.

## Tests

- `test_die_local_rotation_matches_whole_device` (a5, a5sim): the pipeline result
  under `DIE_LOCAL` at K=4 matches the whole-device (`policy=0`) result. Runs on
  sim because a single die can host `DIE_LOCAL`.
- `test_cross_die_penalty_benchmark` (manual, a5 only): runs `DIE_LOCAL` then
  `CROSS_DIE` at the same K, parses the `chip.run.runner_run.aicore_wall` marker
  from the host log, and prints `penalty = (cross_die − die_local) / die_local`.
  `CROSS_DIE` needs two physical dies, so this is hardware only.

## Running on A5

Correctness:

```
pytest tests/st/a5/percounter/die_affinity/test_die_affinity.py \
  --platform a5 --device <id>
```

Benchmark (manual), wrapped for the A5 box:

```
task-submit --device <id> --run \
  "pytest tests/st/a5/percounter/die_affinity/test_die_affinity.py \
     -k benchmark --platform a5 --device <id> --manual include -s"
```

`-s` shows the printed penalty line. Edit `stages`, `tiles`, `k`, `repeats` in
`test_cross_die_penalty_benchmark` to change the workload; a larger `stages`
lengthens the dependency chain and amplifies the cross-die cost.
