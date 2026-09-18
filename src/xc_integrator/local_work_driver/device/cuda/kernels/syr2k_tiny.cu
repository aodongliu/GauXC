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
#include "device/cuda/kernels/syr2k_tiny.hpp"
#include "device/common/syr2k_size_classes.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include "device_specific/cuda_util.hpp"

namespace GauXC {

namespace {

constexpr int tiny_max_nbe    = static_cast<int>(syr2k_tiny_max_nbe);
constexpr int tiny_pair_count = tiny_max_nbe * (tiny_max_nbe + 1) / 2;
constexpr int tiny_block_size = 128;
constexpr int tiny_warp_count = tiny_block_size / static_cast<int>(cuda::warp_size);

// One thread of the block finalizes one entry of the output triangle
static_assert( tiny_pair_count <= tiny_block_size,
  "The TINY syr2k needs one thread per lower-triangle entry" );

/// Where the (row, col) entry of the lower triangle sits in a packed array of
/// tiny_pair_count entries, stored column by column
__host__ __device__ constexpr int tiny_pair_index( int col, int row ) noexcept {
  return col * tiny_max_nbe - (col * (col - 1)) / 2 + (row - col);
}

/// The (row, col) of a packed lower-triangle entry: the inverse of tiny_pair_index
__device__ void tiny_pair_position( int pair, int& col, int& row ) {
  col = 0;
  while( pair >= tiny_max_nbe - col ) { pair -= tiny_max_nbe - col; ++col; }
  row = col + pair;
}

/// One thread block per problem, its threads striding over npts. Rows past nbe
/// read as zero, so every subscript of the accumulation is a constant.
__global__ __launch_bounds__(tiny_block_size) void syr2k_tiny_kernel(
  const cutlass::gemm::GemmCoord* problem_sizes,
  double** ptr_A, double** ptr_B, double** ptr_C, double** ptr_D,
  const int64_t* lda, const int64_t* ldb, const int64_t* ldc, const int64_t* ldd,
  double alpha, double beta ) {

  const int problem = blockIdx.x;
  const int nbe     = problem_sizes[problem].m();
  const int npts    = problem_sizes[problem].k();
  if( nbe <= 0 ) return;
  // Uniform over the block: a larger nbe is a dispatch error, not a datum
  if( nbe > tiny_max_nbe ) __trap();

  const double* A = ptr_A[problem];
  const double* B = ptr_B[problem];
  const int64_t ld_A = lda[problem], ld_B = ldb[problem];

  double acc[tiny_pair_count];
  #pragma unroll
  for( int p = 0; p < tiny_pair_count; ++p ) acc[p] = 0.0;

  for( int t = threadIdx.x; t < npts; t += tiny_block_size ) {
    double av[tiny_max_nbe], bv[tiny_max_nbe];
    #pragma unroll
    for( int i = 0; i < tiny_max_nbe; ++i ) {
      av[i] = i < nbe ? A[i * ld_A + t] : 0.0;
      bv[i] = i < nbe ? B[i * ld_B + t] : 0.0;
    }
    #pragma unroll
    for( int col = 0; col < tiny_max_nbe; ++col ) {
      #pragma unroll
      for( int row = col; row < tiny_max_nbe; ++row ) {
        acc[ tiny_pair_index(col,row) ] += av[row] * bv[col] + bv[row] * av[col];
      }
    }
  }

  // A shuffle tree within each warp, then one thread per triangle entry summing
  // the warps: both stages run in an order fixed by the block size alone
  __shared__ double warp_partial[tiny_warp_count][tiny_pair_count];
  const int lane = threadIdx.x % cuda::warp_size;
  const int warp = threadIdx.x / cuda::warp_size;
  #pragma unroll
  for( int p = 0; p < tiny_pair_count; ++p ) {
    double v = acc[p];
    #pragma unroll
    for( uint32_t offset = cuda::warp_size / 2; offset > 0; offset >>= 1 )
      v += __shfl_down_sync( 0xffffffffu, v, offset );
    if( lane == 0 ) warp_partial[warp][p] = v;
  }
  __syncthreads();

  const int pair = threadIdx.x;
  int col = 0, row = 0;
  double lower = 0.0, upper = 0.0;
  bool active = false;
  if( pair < tiny_pair_count ) {
    tiny_pair_position( pair, col, row );
    active = row < nbe;                // col <= row, so this covers both
  }
  if( active ) {
    double total = warp_partial[0][pair];
    for( int w = 1; w < tiny_warp_count; ++w ) total += warp_partial[w][pair];
    lower = upper = alpha * total;
    if( beta != 0.0 ) {
      const double* C = ptr_C[problem];
      const int64_t ld_C = ldc[problem];
      lower += beta * C[ row + col * ld_C ];
      if( row != col ) upper += beta * C[ col + row * ld_C ];
    }
  }
  // Every C read is behind this barrier, so C and D may name the same memory
  __syncthreads();
  if( active ) {
    double* D = ptr_D[problem];
    const int64_t ld_D = ldd[problem];
    D[ row + col * ld_D ] = lower;
    if( row != col ) D[ col + row * ld_D ] = upper;
  }
}

}

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
) {
  if( problem_count <= 0 ) return;
  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  syr2k_tiny_kernel<<< problem_count, tiny_block_size, 0, stream >>>(
    problem_sizes_device, ptr_A, ptr_B, ptr_C, ptr_D,
    lda, ldb, ldc, ldd, alpha, beta );
}

}
