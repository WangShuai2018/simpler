#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render a die-coloured swimlane SVG from an L1 chip_swimlane_records.json.

The capture names the launch worker a kernel ran on; the AICPU device log names
each launch cluster's physical core and die. Joining the two gives a picture in
which die membership is visible rather than inferred.
"""

import argparse
import glob
import json
import re
from pathlib import Path

CORES_PER_BLOCKDIM = 3
AIV_PER_CLUSTER = 2
DIE_COLOR = {0: "#2563eb", 1: "#dc2626"}
DIE_FILL = {0: "#dbeafe", 1: "#fee2e2"}


def die_map(run_dir):
    """cluster -> (physical_core, die, slot) from the AICPU's own report."""
    rows = {}
    pattern = re.compile(r"cluster (\d+) physical_core (-?\d+) die (\d+) slot (-?\d+)")
    for path in glob.glob(str(run_dir / "ascend/**/*.log"), recursive=True):
        for line in Path(path).read_text(errors="replace").splitlines():
            found = pattern.search(line)
            if found:
                cluster, core, die, slot = map(int, found.groups())
                rows[cluster] = (core, die, slot)
    if not rows:
        raise SystemExit(f"No percounter die map in {run_dir}/ascend; re-run with DSV41_LOG_LEVEL=info")
    return rows


def worker_to_cluster(worker, clusters):
    """Workers are laid out as `clusters` AICs followed by their AIVs in pairs."""
    if worker < clusters:
        return worker, "C"
    offset = worker - clusters
    return offset // AIV_PER_CLUSTER, f"V{offset % AIV_PER_CLUSTER}"


def render(capture, lanes, title, output):
    records = json.loads(Path(capture).read_text())
    rows = records["aicore_tasks"]
    if not rows:
        raise SystemExit(f"No kernel executions in {capture}")
    clusters = max(lanes) + 1
    origin = min(row[3] for row in rows)
    scale = max(row[4] for row in rows) - origin
    busy = {}
    for row in rows:
        cluster, role = worker_to_cluster(row[0], clusters)
        busy.setdefault((cluster, role), []).append((row[3] - origin, row[4] - origin))

    roles = ["C", "V0", "V1"]
    lane_h, gap, left, top, width = 7, 2, 196, 54, 1180
    order = sorted(lanes, key=lambda c: (lanes[c][1], c))
    y, placements, rules = top, [], []
    previous_die = None
    for cluster in order:
        core, die, slot = lanes[cluster]
        if previous_die is not None and die != previous_die:
            rules.append(y - gap)
        previous_die = die
        for role in roles:
            placements.append((y, cluster, role, die, slot, core))
            y += lane_h
        y += gap
    height = y + 34

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 24}" height="{height}" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="9">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="12" y="20" font-size="14" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="12" y="36" fill="#6b7280">'
        f'{sum(1 for c in lanes if lanes[c][2] >= 0)} of {clusters} clusters active &#183; '
        f'span {scale / records["metadata"]["clock_freq_hz"] * 1e6:.1f} us &#183; '
        f'blue = die 0, red = die 1, grey = idle (no kernel ran)</text>',
    ]
    for rule in rules:
        out.append(
            f'<line x1="12" y1="{rule}" x2="{left + width}" y2="{rule}" stroke="#9ca3af" stroke-dasharray="4 3"/>'
        )
    for y, cluster, role, die, slot, core in placements:
        idle = slot < 0
        colour = "#e5e7eb" if idle else DIE_FILL[die]
        out.append(f'<rect x="{left}" y="{y}" width="{width}" height="{lane_h - 1}" fill="{colour}"/>')
        if role == "C":
            label = f"die{die} cluster{cluster:02d} core{core:02d}"
            label += "  IDLE" if idle else f"  slot{slot:02d}"
            out.append(
                f'<text x="12" y="{y + lane_h + 4}" fill="{"#9ca3af" if idle else DIE_COLOR[die]}">{label}</text>'
            )
        for start, end in busy.get((cluster, role), []):
            x = left + start / scale * width
            w = max((end - start) / scale * width, 0.7)
            out.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{lane_h - 1}" fill="{DIE_COLOR[die]}"/>')
    out.append(
        f'<text x="{left}" y="{height - 12}" fill="#6b7280">0 us</text>'
        f'<text x="{left + width - 40}" y="{height - 12}" fill="#6b7280">'
        f'{scale / records["metadata"]["clock_freq_hz"] * 1e6:.1f} us</text></svg>'
    )
    Path(output).write_text("\n".join(out))
    active = sorted(c for c in lanes if lanes[c][2] >= 0)
    touched = sorted({lanes[c][1] for c in active})
    observed = sorted({worker_to_cluster(row[0], clusters)[0] for row in rows})
    return {
        "capture": str(capture),
        "svg": str(output),
        "clusters_launched": clusters,
        "clusters_active_per_aicpu": active,
        "clusters_observed_in_capture": observed,
        "consistent": active == observed,
        "dies_touched": touched,
        "per_die_active": {die: sum(1 for c in active if lanes[c][1] == die) for die in touched},
        "kernel_executions": len(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="rank directory holding ascend/ and run*/")
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--capture", default="", help="run subdirectory (default: the last one)")
    args = parser.parse_args()
    lanes = die_map(args.run)
    captures = sorted(
        args.run.glob("run*/chip_swimlane_records.json"), key=lambda p: int(re.sub(r"\D", "", p.parent.name))
    )
    chosen = args.run / args.capture / "chip_swimlane_records.json" if args.capture else captures[-1]
    report = render(chosen, lanes, args.title or str(args.run), args.output)
    print(json.dumps(report, indent=2))
