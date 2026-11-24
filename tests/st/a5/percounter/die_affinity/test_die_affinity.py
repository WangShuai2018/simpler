#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Die-affinity scheduling for the A5 percounter runtime.

The correctness test runs the rotated add_scalar pipeline (die_rotation_orch)
under DieSchedPolicy.DIE_LOCAL at a small cluster count and checks the result is
unchanged from the whole-device run: narrowing which clusters draw work must not
change what the graph computes.

The benchmark (manual, a5 only) runs the same pipeline under DIE_LOCAL and
CROSS_DIE at the same K and reports the cross-die penalty from the on-chip
aicore_wall marker. It needs two physical dies, so it never runs on sim.
"""

import statistics
from pathlib import Path

import pytest
import torch

import simpler.task_interface as task_interface
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker
from simpler_setup.scene_test import compile_chip_callable_spec
from simpler_setup.tools.strace_timing import parse_spans

pytestmark = [pytest.mark.platforms(["a5", "a5sim"]), pytest.mark.runtime("percounter"), pytest.mark.device_count(1)]

# PercounterControl.die_sched_policy values; mirror CallConfig::DieSchedPolicy.
_DIE_SCHED_ALL = 0
_DIE_SCHED_DIE_LOCAL = 1
_DIE_SCHED_CROSS_DIE = 2

# The add_scalar kernel is fixed to one 128x128 float32 tile.
_TILE = 128 * 128

_AICORE_WALL = "chip.run.runner_run.aicore_wall"


def _build_callable(platform):
    vector_kernels = Path(__file__).resolve().parents[2] / "host_build_graph/vector_example/kernels"
    spec = {
        "orchestration": {
            "source": str(Path(__file__).resolve().parent / "kernels/orchestration/die_rotation_orch.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],  # buf tensor, then scalars stages, tiles
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(vector_kernels / "aiv/kernel_add_scalar.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }
    return compile_chip_callable_spec(spec, platform, "percounter", (__file__, platform, "percounter"))


def _rotation_input(stages, tiles):
    """Buffer of (stages+1)*tiles tiles. Region 0 tile t is filled with t; the
    rest with -1 so an unwritten tile is visible."""
    host = torch.full(((stages + 1) * tiles * _TILE,), -1.0, dtype=torch.float32)
    for t in range(tiles):
        host[t * _TILE : (t + 1) * _TILE] = float(t)
    return host


def _assert_rotation_golden(out, stages, tiles):
    for r in range(1, stages + 1):
        for t in range(tiles):
            expected = float(((t + r) % tiles) + r)
            base = (r * tiles + t) * _TILE
            assert torch.all(out[base : base + _TILE] == expected), f"region {r} tile {t} != {expected}"


def _run_rotation(worker, handle, buf, host, stages, tiles, *, policy, k, output_prefix=None):
    worker.copy_to(buf, host)
    regions = stages + 1
    args = TaskArgs()
    args.add_tensor(buf.tensor(shapes=(regions * tiles, _TILE), dtype=DataType.FLOAT32), TensorArgType.INOUT)
    args.add_scalar(stages)
    args.add_scalar(tiles)
    config = CallConfig()
    config.die_sched_policy = policy
    config.aic_cluster_count = k
    if output_prefix is not None:
        config.output_prefix = str(output_prefix)
    worker.run(handle, args, config)
    assert worker.scheduler_runtime_mode == 5
    out = torch.empty_like(host)
    worker.copy_from(out, buf)
    return out


@pytest.mark.parametrize(
    "policy,k",
    [(_DIE_SCHED_ALL, 0), (_DIE_SCHED_DIE_LOCAL, 4)],
    ids=["all", "die_local_k4"],
)
def test_die_local_rotation_matches_whole_device(st_platform, st_device_ids, tmp_path, policy, k):
    stages, tiles = 4, 8
    worker = Worker(level=2, platform=st_platform, runtime="percounter", device_id=int(st_device_ids[0]))
    buf = None
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        host = _rotation_input(stages, tiles)
        buf = worker.malloc(host.nbytes)
        out = _run_rotation(worker, handle, buf, host, stages, tiles, policy=policy, k=k)
        _assert_rotation_golden(out, stages, tiles)
    finally:
        if buf is not None:
            worker.free(buf)
        worker.close()


def _aicore_walls(log_dir):
    """aicore_wall durations (ns), ordered by invocation id."""
    samples = []
    for log_file in sorted(Path(log_dir).glob("host.*.log")):
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for span in parse_spans(lines):
            if span.name == _AICORE_WALL:
                samples.append((span.inv, span.dur))
    samples.sort()
    return [dur for _, dur in samples]


@pytest.mark.manual(["a5"])
def test_cross_die_penalty_benchmark(st_platform, st_device_ids, tmp_path):
    if st_platform != "a5":
        pytest.skip("cross-die penalty needs two physical dies; sim is one die")
    stages, tiles, k, repeats = 8, 16, 8, 20  # k even for CROSS_DIE
    worker = Worker(level=2, platform=st_platform, runtime="percounter", device_id=int(st_device_ids[0]))
    buf = None
    prefix = tmp_path / "artifacts"
    prefix.mkdir()
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        host = _rotation_input(stages, tiles)
        buf = worker.malloc(host.nbytes)

        def run_batch(policy):
            for _ in range(repeats):
                out = _run_rotation(
                    worker, handle, buf, host, stages, tiles, policy=policy, k=k, output_prefix=prefix
                )
                _assert_rotation_golden(out, stages, tiles)

        run_batch(_DIE_SCHED_DIE_LOCAL)
        run_batch(_DIE_SCHED_CROSS_DIE)
        task_interface._native_flush_host_log()

        log_dir = Path(str(task_interface._native_host_log_directory()))
        walls = _aicore_walls(log_dir)
        assert len(walls) >= 2 * repeats, f"found only {len(walls)} aicore_wall markers"
        walls = walls[-2 * repeats :]
        die_local, cross_die = walls[:repeats], walls[repeats:]
        median_local = statistics.median(die_local)
        median_cross = statistics.median(cross_die)
        penalty = (median_cross - median_local) / median_local
        print(
            f"\ndie-affinity K={k}: DIE_LOCAL median aicore_wall={median_local / 1000:.1f}us "
            f"CROSS_DIE={median_cross / 1000:.1f}us penalty={penalty:.2%} "
            f"(stages={stages} tiles={tiles} repeats={repeats})"
        )
        assert median_local > 0 and median_cross > 0
    finally:
        if buf is not None:
            worker.free(buf)
        worker.close()
