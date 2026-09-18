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

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>

#include "device/device_queue.hpp"

namespace GauXC {

/// D = alpha (A B^T + B A^T) + beta C over the full nbe x nbe block, one thread
/// block per problem, nbe <= syr2k_tiny_max_nbe. The arguments and layouts are
/// cutlass_syr2k's without its host problem sizes; C is read before D is
/// written, so the two may name the same memory.
void syr2k_tiny(
  const int problem_count,
  const cutlass::gemm::GemmCoord* problem_sizes_device,
  double ** ptr_A,
  double ** ptr_B,
  double ** ptr_C,
  double ** ptr_D,
  int64_t* lda,
  int64_t* ldb,
  int64_t* ldc,
  int64_t* ldd,
  const double alpha,
  const double beta,
  device_queue queue
);

}
