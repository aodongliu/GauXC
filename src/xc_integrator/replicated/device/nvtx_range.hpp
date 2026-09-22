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

#ifdef GAUXC_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace GauXC {
namespace detail {

/// NVTX range over consecutive stages: next() closes the open range and opens a new one.
/// Does nothing unless GAUXC_ENABLE_NVTX
class nvtx_range {
#ifdef GAUXC_ENABLE_NVTX
  bool open_ = false;
public:
  nvtx_range() = default;
  nvtx_range( const nvtx_range& ) = delete;
  nvtx_range& operator=( const nvtx_range& ) = delete;
  inline void next( const char* name ) { end(); nvtxRangePushA( name ); open_ = true; }
  inline void end() { if( open_ ) nvtxRangePop(); open_ = false; }
  ~nvtx_range() noexcept { end(); }
#else
public:
  inline void next( const char* ) {}
  inline void end() {}
#endif
};

}
}
