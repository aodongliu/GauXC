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

/** Size classes of the multiparticle syr2k, V <- alpha (Phi Z^T + Z Phi^T),
 *  by a task's nbe. The species-task list is ordered in work ranges of one class and
 *  the CUDA syr2k dispatch runs one executor per class:
 *    TINY    nbe <= 8   syr2k_tiny
 *    SMALL   nbe <= 32  cutlass_syr2k_32x32 (threadblock 32x32x16)
 *    GENERAL otherwise  cutlass_syr2k       (threadblock 64x64x16)
 */

namespace GauXC {

enum class syr2k_size_class : std::uint32_t {
  TINY    = 0,
  SMALL   = 1,
  GENERAL = 2
};

/// Number of size classes; sized for array indexing by class.
inline constexpr std::uint32_t syr2k_size_class_count = 3;

/// Largest nbe served by the custom TINY kernel.
inline constexpr std::int64_t syr2k_tiny_max_nbe = 8;

/// Largest nbe served by the 32x32 CUTLASS tile.
inline constexpr std::int64_t syr2k_small_max_nbe = 32;

/// The class of a task with nbe basis functions; nbe <= 0 is TINY.
constexpr syr2k_size_class syr2k_size_class_of( std::int64_t nbe ) noexcept {
  if( nbe <= syr2k_tiny_max_nbe  ) return syr2k_size_class::TINY;
  if( nbe <= syr2k_small_max_nbe ) return syr2k_size_class::SMALL;
  return syr2k_size_class::GENERAL;
}

}
