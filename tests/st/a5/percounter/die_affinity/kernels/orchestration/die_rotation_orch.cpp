/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * A rotated add_scalar pipeline that spreads work across clusters and makes a
 * tile produced on one cluster feed a consumer on another.
 *
 * One external INOUT buffer holds (S+1)*T tiles of 128x128 float32, row r =
 * region*T + tile. Region 0 is the input; stage s (0..S-1) writes region s+1
 * tile t from region s tile (t+1)%T plus one. The rotation (reading tile
 * (t+1)%T, writing tile t) crosses tiles, so a stage's output lands on a
 * different cluster than the producer it read -- which is where the die the
 * consumer runs on starts to matter. Within a stage the T output tiles are
 * disjoint byte ranges, so the T tasks are independent and fill the clusters.
 *
 * Arg layout: [buf (INOUT)], then scalars S and T.
 *
 * After the run, region r tile t == region 0 tile ((t+r)%T) + r.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAddScalar = 0;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};  // 1 tensor + 2 scalars
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto &buf = orch_args.tensor(0).ref();
    const uint32_t stages = static_cast<uint32_t>(orch_args.scalar<int64_t>(0));
    const uint32_t tiles = static_cast<uint32_t>(orch_args.scalar<int64_t>(1));

    union {
        float f32;
        uint64_t u64;
    } one;
    one.f32 = 1.0f;

    for (uint32_t s = 0; s < stages; ++s) {
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t in_row = s * tiles + (t + 1) % tiles;
            const uint32_t out_row = (s + 1) * tiles + t;
            const auto in_tile = buf.slice(0, in_row, in_row + 1);
            const auto out_tile = buf.slice(0, out_row, out_row + 1);
            CoreTaskArgs task;
            task.add_input(in_tile);
            task.add_output(out_tile);
            task.add_scalar(one.u64);
            rt_submit_aiv_task(kAddScalar, task);
        }
    }
}

}  // extern "C"
