/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once
#include <cstdint>

namespace GauXC {
namespace cuda  {

static constexpr uint32_t warp_size = 32;
static constexpr uint32_t max_threads_per_thread_block = 1024;
static constexpr uint32_t max_warps_per_thread_block = 
  max_threads_per_thread_block / warp_size;

// A batch mixes tasks of very different sizes. The V-variable and Z-matrix
// kernels grid-stride over basis functions and points, so their per-task grid
// is sized for a typical task (128 functions x 512 points), not the largest.
static constexpr uint32_t max_task_blocks_basis_functions = 4;
static constexpr uint32_t max_task_blocks_points          = 16;

}

}
