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
#include "device/common/multiparticle_pair.hpp"
#include "device_specific/cuda_util.hpp"

namespace GauXC {

namespace {
constexpr int pair_block_size = 256;
}

/// One block per pair point range, its threads looping over the range's points
__global__ void gather_pair_densities_kernel( size_t nranges,
  const pair_point_range* ranges, const double* den_s, const double* den_z,
  double* density ) {

  if( blockIdx.x >= nranges ) return;
  const auto& range = ranges[blockIdx.x];

  for( int i = threadIdx.x; i < range.npts; i += blockDim.x ) {
    const size_t electron = range.electron_offset + i;
    const size_t particle = range.particle_offset + i;
    const size_t k        = range.first + i;
    // After the U variables the polarized shape holds (rho_alpha, rho_beta), so
    // a species' total density is den_s + den_z; the one-channel shape holds it
    // in den_s
    density[2*k + 0] = den_z ? den_s[electron] + den_z[electron] : den_s[electron];
    density[2*k + 1] = den_z ? den_s[particle] + den_z[particle] : den_s[particle];
  }
}

/// One block per pair point range. The electron's points are shared with every
/// other pair live on its grid task; the particle's belong to this pair alone.
__global__ void scatter_pair_potential_kernel( size_t nranges,
  const pair_point_range* ranges, const double* weights, const double* vrho,
  double* vrho_pos, double* vrho_neg ) {

  if( blockIdx.x >= nranges ) return;
  const auto& range = ranges[blockIdx.x];

  for( int i = threadIdx.x; i < range.npts; i += blockDim.x ) {
    const size_t electron = range.electron_offset + i;
    const size_t particle = range.particle_offset + i;
    const size_t k        = range.first + i;
    const double weight   = weights[particle];

    const double electron_potential = weight * vrho[2*k + 0];
    atomicAdd( vrho_pos + electron, electron_potential );
    if( vrho_neg ) atomicAdd( vrho_neg + electron, electron_potential );

    const double particle_potential = weight * vrho[2*k + 1];
    vrho_pos[particle] += particle_potential;
    if( vrho_neg and range.particle_channels == 2 )
      vrho_neg[particle] += particle_potential;
  }
}

/// One block per pair point range: a fixed-order block reduction of the range's
/// energy, then one addition into the pair's accumulator
__global__ void sum_pair_energy_kernel( size_t nranges,
  const pair_point_range* ranges, const double* weights, const double* eps,
  const double* density, double* pair_exc ) {

  __shared__ double partial[pair_block_size];

  if( blockIdx.x >= nranges ) return;
  const auto& range = ranges[blockIdx.x];

  double energy = 0.0;
  for( int i = threadIdx.x; i < range.npts; i += blockDim.x ) {
    const size_t k = range.first + i;
    energy += weights[range.particle_offset + i] * eps[k] * ( density[2*k + 0] + density[2*k + 1] );
  }
  partial[threadIdx.x] = energy;
  __syncthreads();

  for( int stride = blockDim.x / 2; stride > 0; stride >>= 1 ) {
    if( threadIdx.x < stride ) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if( threadIdx.x == 0 ) atomicAdd( pair_exc + range.pair, partial[0] );
}

__global__ void add_intra_w_times_f_kernel( size_t ntasks,
  const XCDeviceTask* device_tasks, const size_t* task_grid_offset,
  double* w_times_f ) {

  if( blockIdx.x >= ntasks ) return;
  const auto&  task        = device_tasks[blockIdx.x];
  const size_t grid_offset = task_grid_offset[blockIdx.x];

  for( size_t i = threadIdx.x; i < task.npts; i += blockDim.x ) {
    const double density = task.den_z ? task.den_s[i] + task.den_z[i]
                                      : task.den_s[i];
    atomicAdd( w_times_f + grid_offset + i, task.eps[i] * density );
  }
}

__global__ void add_pair_w_times_f_kernel( size_t nranges,
  const pair_point_range* ranges, const double* weights, const double* eps,
  const double* density, double* w_times_f ) {

  if( blockIdx.x >= nranges ) return;
  const auto& range = ranges[blockIdx.x];

  for( int i = threadIdx.x; i < range.npts; i += blockDim.x ) {
    const size_t k = range.first + i;
    atomicAdd( w_times_f + range.grid_offset + i,
      weights[range.particle_offset + i] * eps[k] * ( density[2*k + 0] + density[2*k + 1] ) );
  }
}


void gather_pair_densities( size_t nranges, const pair_point_range* ranges,
  const double* den_s, const double* den_z, double* density,
  device_queue queue ) {

  if( not nranges ) return;
  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  gather_pair_densities_kernel<<< nranges, pair_block_size, 0, stream >>>(
    nranges, ranges, den_s, den_z, density );
}

void scatter_pair_potential( size_t nranges, const pair_point_range* ranges,
  const double* weights, const double* vrho, double* vrho_pos,
  double* vrho_neg, device_queue queue ) {

  if( not nranges ) return;
  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  scatter_pair_potential_kernel<<< nranges, pair_block_size, 0, stream >>>(
    nranges, ranges, weights, vrho, vrho_pos, vrho_neg );
}

void sum_pair_energy( size_t nranges, const pair_point_range* ranges,
  const double* weights, const double* eps, const double* density,
  double* pair_exc, device_queue queue ) {

  if( not nranges ) return;
  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  sum_pair_energy_kernel<<< nranges, pair_block_size, 0, stream >>>(
    nranges, ranges, weights, eps, density, pair_exc );
}

void add_intra_w_times_f( size_t ntasks, const XCDeviceTask* device_tasks,
  const size_t* task_grid_offset, double* w_times_f, device_queue queue ) {

  if( not ntasks ) return;
  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  add_intra_w_times_f_kernel<<< ntasks, pair_block_size, 0, stream >>>(
    ntasks, device_tasks, task_grid_offset, w_times_f );
}

void add_pair_w_times_f( size_t nranges, const pair_point_range* ranges,
  const double* weights, const double* eps, const double* density,
  double* w_times_f, device_queue queue ) {

  if( not nranges ) return;
  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  add_pair_w_times_f_kernel<<< nranges, pair_block_size, 0, stream >>>(
    nranges, ranges, weights, eps, density, w_times_f );
}

}
