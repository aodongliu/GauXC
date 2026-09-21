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
#include "device/xc_device_data.hpp"
#include "device/xc_device_task.hpp"
#include "device/device_queue.hpp"

namespace GauXC {

/*
 * The multiparticle pair (EPC) stage. The pairs of a batch form one point list;
 * a pair_point_range says where one pair's points sit in it and where its two
 * species' points sit in the batch's per-point arrays. Every call below runs
 * one thread block per range, so one call serves every pair of the batch.
 */

/// Read both species' total densities at every pair point, interleaved as the
/// pair functional's polarized input: the electron's first, the particle's second
void gather_pair_densities( size_t nranges, const pair_point_range* ranges,
  const double* den_s, const double* den_z, double* density,
  device_queue queue );

/// Weight every pair's derivatives and add them into its two species' potential
/// channels: the electron's into both of its channels, the quantum particle's
/// into as many as pair_point_range::particle_channels names
void scatter_pair_potential( size_t nranges, const pair_point_range* ranges,
  const double* weights, const double* vrho, double* vrho_pos,
  double* vrho_neg, device_queue queue );

/// Add every pair's energy into its accumulator, over its own points:
/// E_pair = sum_k w_k eps_k ( rho_electron_k + rho_particle_k )
void sum_pair_energy( size_t nranges, const pair_point_range* ranges,
  const double* weights, const double* eps, const double* density,
  double* pair_exc, device_queue queue );


/// For the first `ntasks` species tasks (those with an intra functional):
/// w_times_f[grid_offset + i] += eps[i] * rho[i], eps already multiplied by the weight
void add_intra_w_times_f( size_t ntasks, const XCDeviceTask* device_tasks,
  const size_t* task_grid_offset, double* w_times_f, device_queue queue );

/// For every pair point range:
/// w_times_f[grid_offset + i] += w * eps * ( rho_electron + rho_particle )
void add_pair_w_times_f( size_t nranges, const pair_point_range* ranges,
  const double* weights, const double* eps, const double* density,
  double* w_times_f, device_queue queue );

}
