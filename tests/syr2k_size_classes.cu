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
#include "device/cuda/kernels/cutlass_wrapper.hpp"
#include "device/cuda/kernels/syr2k_tiny.hpp"
#include "device/common/syr2k_size_classes.hpp"
#include "device_specific/cuda_util.hpp"

#include <catch2/catch.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

namespace GauXC {
namespace {

constexpr double syr2k_alpha = 0.375;

/// The value in every entry the executor must not write
constexpr double sentinel = -1234.5;

/// A device allocation, filled from a host vector and freed with itself
template <typename T>
class device_array {

public:
  explicit device_array( const std::vector<T>& host ) : count_(host.size()) {
    REQUIRE( cudaMalloc( (void**)&data_, count_*sizeof(T) ) == cudaSuccess );
    REQUIRE( cudaMemcpy( data_, host.data(), count_*sizeof(T),
      cudaMemcpyHostToDevice ) == cudaSuccess );
  }
  device_array( device_array&& other ) noexcept
    : data_(other.data_), count_(other.count_) { other.data_ = nullptr; }
  device_array( const device_array& ) = delete;
  ~device_array() noexcept { if( data_ ) cudaFree(data_); }

  T* data() const noexcept { return data_; }
  void receive( std::vector<T>& host ) const {
    REQUIRE( cudaMemcpy( host.data(), data_, count_*sizeof(T),
      cudaMemcpyDeviceToHost ) == cudaSuccess );
  }

private:
  T*     data_  = nullptr;
  size_t count_ = 0;
};

/// One problem in the grouped syr2k's layouts: A, B are nbe x npts row major
/// with row strides lda, ldb; C, D are nbe x nbe column major with column
/// strides ldc, ldd. Every stride carries padding the executor must not touch.
struct syr2k_problem {
  int     nbe  = 0;
  int     npts = 0;
  int64_t lda = 0, ldb = 0, ldc = 0, ldd = 0;
  std::vector<double> A, B, C, D;
};

/// A random problem of the given nbe, D and the padding at the sentinel. The
/// beta term's C is deliberately non-symmetric, and holds the entry (row, col)
/// the full-block contract reads there.
syr2k_problem random_problem( int nbe, double beta, bool alias,
  std::mt19937& generator ) {

  std::uniform_int_distribution<int>     npts_of( 16, 3000 );
  std::uniform_int_distribution<int64_t> padding( 0, 5 );
  std::uniform_real_distribution<double> entry( -1.0, 1.0 );

  syr2k_problem problem;
  problem.nbe  = nbe;
  problem.npts = npts_of( generator );
  problem.lda  = problem.npts + padding( generator );
  problem.ldb  = problem.npts + padding( generator );
  problem.ldd  = nbe + padding( generator );
  problem.ldc  = alias ? problem.ldd : nbe + padding( generator );

  problem.A.assign( nbe * problem.lda, sentinel );
  problem.B.assign( nbe * problem.ldb, sentinel );
  for( int row = 0; row < nbe; ++row )
  for( int t = 0; t < problem.npts; ++t ) {
    problem.A[ row*problem.lda + t ] = entry( generator );
    problem.B[ row*problem.ldb + t ] = entry( generator );
  }

  problem.C.assign( problem.ldc * nbe, sentinel );
  problem.D.assign( problem.ldd * nbe, sentinel );
  if( beta != 0.0 ) {
    auto&   target = alias ? problem.D : problem.C;
    int64_t stride = alias ? problem.ldd : problem.ldc;
    for( int col = 0; col < nbe; ++col )
    for( int row = 0; row < nbe; ++row )
      target[ row + col*stride ] = 0.2*row - 0.3*col + 1.25;
  }
  return problem;
}

/// D = alpha (A B^T + B A^T) + beta C over the full block, with the sum of the
/// magnitudes of the same terms alongside it as the comparison's normalizer
void reference_product( const syr2k_problem& problem, double beta, bool alias,
  std::vector<double>& reference, std::vector<double>& magnitude ) {

  reference.assign( problem.D.size(), 0.0 );
  magnitude.assign( problem.D.size(), 0.0 );
  for( int col = 0; col < problem.nbe; ++col )
  for( int row = 0; row < problem.nbe; ++row ) {
    double value = 0.0, scale = 0.0;
    for( int t = 0; t < problem.npts; ++t ) {
      const double term = problem.A[ row*problem.lda + t ]*problem.B[ col*problem.ldb + t ]
                        + problem.B[ row*problem.ldb + t ]*problem.A[ col*problem.lda + t ];
      value += term;
      scale += std::abs( term );
    }
    value *= syr2k_alpha;
    scale *= std::abs( syr2k_alpha );
    if( beta != 0.0 ) {
      const double c = alias ? problem.D[ row + col*problem.ldd ]
                             : problem.C[ row + col*problem.ldc ];
      value += beta * c;
      scale += std::abs( beta * c );
    }
    reference[ row + col*problem.ldd ] = value;
    magnitude[ row + col*problem.ldd ] = scale;
  }
}

/// Run one group of problems of a single size class through that class's
/// executor and return each problem's D allocation
std::vector<std::vector<double>> run_group(
  const std::vector<syr2k_problem>& problems, syr2k_size_class size_class,
  double beta, bool alias ) {

  const int count = problems.size();
  std::vector<device_array<double>> A, B, C, D;
  A.reserve(count); B.reserve(count); C.reserve(count); D.reserve(count);
  std::vector<double*> ptr_A(count), ptr_B(count), ptr_C(count), ptr_D(count);
  std::vector<int64_t> lda(count), ldb(count), ldc(count), ldd(count);
  std::vector<cutlass::gemm::GemmCoord> sizes;
  sizes.reserve(count);

  for( int i = 0; i < count; ++i ) {
    const auto& problem = problems[i];
    A.emplace_back( problem.A );
    B.emplace_back( problem.B );
    D.emplace_back( problem.D );
    ptr_A[i] = A[i].data();
    ptr_B[i] = B[i].data();
    ptr_D[i] = D[i].data();
    if( alias ) ptr_C[i] = D[i].data();
    else {
      C.emplace_back( problem.C );
      ptr_C[i] = C.back().data();
    }
    lda[i] = problem.lda; ldb[i] = problem.ldb;
    ldc[i] = problem.ldc; ldd[i] = problem.ldd;
    sizes.emplace_back( problem.nbe, problem.nbe, problem.npts );
  }

  device_array<double*> device_A(ptr_A), device_B(ptr_B), device_C(ptr_C),
                        device_D(ptr_D);
  device_array<int64_t> device_lda(lda), device_ldb(ldb), device_ldc(ldc),
                        device_ldd(ldd);
  device_array<cutlass::gemm::GemmCoord> device_sizes(sizes);

  auto queue = device_queue::generate<util::cuda_stream>();
  if( size_class == syr2k_size_class::TINY )
    syr2k_tiny( count, device_sizes.data(),
      device_A.data(), device_B.data(), device_C.data(), device_D.data(),
      device_lda.data(), device_ldb.data(), device_ldc.data(), device_ldd.data(),
      syr2k_alpha, beta, queue );
  else {
    auto cutlass_syr2k_tile = size_class == syr2k_size_class::SMALL
      ? &cutlass_syr2k_32x32 : &cutlass_syr2k;
    cutlass_syr2k_tile( device_sizes.data(), sizes.data(), count,
      device_A.data(), device_B.data(), device_C.data(), device_D.data(),
      device_lda.data(), device_ldb.data(), device_ldc.data(), device_ldd.data(),
      syr2k_alpha, beta, queue );
  }
  REQUIRE( cudaStreamSynchronize( queue.queue_as<util::cuda_stream>() ) ==
           cudaSuccess );

  std::vector<std::vector<double>> written( count );
  for( int i = 0; i < count; ++i ) {
    written[i] = problems[i].D;
    D[i].receive( written[i] );
  }
  return written;
}

}


/// The size-class rule at the edges of its two constants
TEST_CASE( "syr2k size class rule", "[syr2k-size-classes]" ) {
  CHECK( syr2k_tiny_max_nbe  == 8  );
  CHECK( syr2k_small_max_nbe == 32 );
  CHECK( syr2k_size_class_of(1)  == syr2k_size_class::TINY    );
  CHECK( syr2k_size_class_of(8)  == syr2k_size_class::TINY    );
  CHECK( syr2k_size_class_of(9)  == syr2k_size_class::SMALL   );
  CHECK( syr2k_size_class_of(32) == syr2k_size_class::SMALL   );
  CHECK( syr2k_size_class_of(33) == syr2k_size_class::GENERAL );
}


/// Each size class's executor against a host reference product: the values on
/// the entries it writes, and the entries it must not write -- a CUTLASS tile's
/// strictly upper triangle, the padding past nbe -- still as they arrived.
TEST_CASE( "syr2k size class executors", "[syr2k-size-classes]" ) {

  const std::vector<std::pair<syr2k_size_class, std::vector<int>>> classes = {
    { syr2k_size_class::TINY,    {  1,  4,   8 } },
    { syr2k_size_class::SMALL,   {  9, 16,  32 } },
    { syr2k_size_class::GENERAL, { 33, 64, 133 } }
  };

  for( const auto& [ size_class, nbe_list ] : classes )
  // beta = 1 is the accumulating call the potential's gradient channels make
  // into the block they read, so C names the same memory as D there
  for( const auto& [ beta, alias ] :
       std::vector<std::pair<double,bool>>{ { 0.0, false }, { 1.0, true } } ) {

    std::mt19937 generator( 20260917u + 131u*unsigned(nbe_list.front())
                                      + ( beta != 0.0 ? 17u : 0u ) );
    std::vector<syr2k_problem> problems;
    for( int nbe : nbe_list )
    for( int repeat = 0; repeat < 2; ++repeat )
      problems.push_back( random_problem( nbe, beta, alias, generator ) );

    const auto written = run_group( problems, size_class, beta, alias );

    for( size_t i = 0; i < problems.size(); ++i ) {
      const auto& problem = problems[i];
      std::vector<double> reference, magnitude;
      reference_product( problem, beta, alias, reference, magnitude );

      const bool lower_only = size_class != syr2k_size_class::TINY;
      double worst_relative = 0.0;
      size_t upper_untouched = 0, padding_touched = 0;
      for( int col = 0; col < problem.nbe; ++col )
      for( int row = 0; row < problem.ldd; ++row ) {
        const size_t entry = row + col*problem.ldd;
        if( row >= problem.nbe ) {
          if( written[i][entry] != problem.D[entry] ) padding_touched++;
        } else if( lower_only and row < col ) {
          if( written[i][entry] == problem.D[entry] ) upper_untouched++;
        } else {
          worst_relative = std::max( worst_relative,
            std::abs( written[i][entry] - reference[entry] ) /
            std::max( 1.0, magnitude[entry] ) );
        }
      }

      const size_t upper_count = problem.nbe * (problem.nbe - 1) / 2;
      INFO( "nbe=" << problem.nbe << " npts=" << problem.npts
            << " lda=" << problem.lda << " ldb=" << problem.ldb
            << " ldc=" << problem.ldc << " ldd=" << problem.ldd
            << " beta=" << beta << ( alias ? " C==D" : " C!=D" )
            << " worst_relative=" << worst_relative
            << " upper_untouched=" << upper_untouched
            << " padding_touched=" << padding_touched );
      CHECK( worst_relative <= 1e-14 );
      CHECK( padding_touched == 0u );
      CHECK( upper_untouched == ( lower_only ? upper_count : 0u ) );
    }
  }
}

}
