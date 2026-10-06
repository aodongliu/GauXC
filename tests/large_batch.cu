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
#include <gauxc/basisset.hpp>
#include <gauxc/basisset_map.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include "xc_integrator/local_work_driver/device/local_device_work_driver.hpp"
#include "xc_integrator/xc_data/device/xc_device_data.hpp"

#include <catch2/catch.hpp>
#include <cuda_runtime.h>

#include <vector>

namespace GauXC {
namespace {

/// CUDA caps gridDim.z, which the per-task kernels index by task, at 65 535
constexpr size_t grid_dim_z_max = 65535;

/// More tasks than one launch can index
constexpr size_t large_task_count = 70000;

/// One s shell on the atom at the origin
Shell<double> s_shell() {
  return Shell<double>( PrimSize(1), AngularMomentum(0), SphericalType(false),
    Shell<double>::prim_array{1.0}, Shell<double>::prim_array{1.0},
    Shell<double>::cart_array{0.0, 0.0, 0.0} );
}

/// A task of two points on atom 0 whose screening keeps shell 0 in every species
XCTask tiny_task( size_t nspecies ) {
  XCTask task;
  task.iParent      = 0;
  task.points       = { {0.1, 0.0, 0.0}, {0.0, 0.1, 0.0} };
  task.weights      = { 0.5, 0.5 };
  task.npts         = 2;
  task.dist_nearest = 1.0;
  task.bfn_screening.shell_list = { 0 };
  task.bfn_screening.nbe        = 1;
  if( nspecies > 1 ) task.bfn_screenings.assign( nspecies, task.bfn_screening );
  return task;
}

integrator_term_tracker lda_rks_exc_vxc() {
  integrator_term_tracker terms;
  terms.exc_vxc   = true;
  terms.xc_approx = integrator_xc_approx::LDA;
  terms.ks_scheme = RKS;
  return terms;
}

}

TEST_CASE( "Device batches respect the grid z limit", "[large-batch]" ) {

  auto rt  = DeviceRuntimeEnvironment( GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9 );
  auto lwd_base = LocalWorkDriverFactory::make_local_work_driver(
    ExecutionSpace::Device, "Default", LocalWorkSettings() );
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>( lwd_base.get() );
  REQUIRE( lwd );
  auto data = lwd->create_device_data( rt );

  Molecule mol;
  mol.emplace_back( AtomicNumber(1), 0.0, 0.0, 0.0 );
  const auto terms = lda_rks_exc_vxc();

  SECTION( "Single-species batches" ) {
    BasisSet<double> basis( 1, s_shell() );
    BasisSetMap basis_map( basis, mol );
    const std::vector<double> P = { 1.0 };

    data->reset_allocations();
    data->allocate_static_data_exc_vxc( 1, 1, terms, true );
    data->send_static_data_density_basis( P.data(), 1, nullptr, 0,
      nullptr, 0, nullptr, 0, basis );
    data->zero_exc_vxc_integrands( terms );

    std::vector<XCTask> tasks( large_task_count, tiny_task(1) );
    data->populate_submat_maps( 1, tasks.begin(), tasks.end(), basis_map );

    // Every batch launches a kernel with one grid z block per task
    size_t ntasks_covered = 0;
    auto task_it = tasks.begin();
    while( task_it != tasks.end() ) {
      auto task_end = data->generate_buffers( terms, basis_map, task_it, tasks.end() );
      const size_t ntasks = std::distance( task_it, task_end );
      REQUIRE( ntasks > 0 );
      CHECK( ntasks <= grid_dim_z_max );
      REQUIRE_NOTHROW( lwd->eval_uvars_lda( data.get(), RKS ) );
      REQUIRE( cudaDeviceSynchronize() == cudaSuccess );
      REQUIRE( cudaGetLastError() == cudaSuccess );
      ntasks_covered += ntasks;
      task_it = task_end;
    }
    CHECK( ntasks_covered == large_task_count );
  }

  SECTION( "Multiparticle batches" ) {
    // An electron and a quantum particle, one s shell each, in one union basis
    BasisSet<double> union_basis( 2, s_shell() );
    BasisSetMap union_basis_map( union_basis, mol );
    const std::vector<double> P = { 1.0, 0.0, 0.0, 1.0 };

    multiparticle_tracker tracker;
    tracker.species.resize( 2 );
    for( auto& species : tracker.species ) {
      species.nbf          = 1;
      species.build_vxc    = true;
      species.participates = true;
      species.xc_approx    = integrator_xc_approx::LDA;
      species.ks_scheme    = RKS;
    }
    tracker.species[0].has_intra_functional = true;
    tracker.pairs  = { multiparticle_pair{ 0, 1, true, 0 } };
    tracker.do_vxc = true;
    const std::vector<species_basis_range> species_basis_ranges = { {0, 0}, {1, 1} };

    data->reset_allocations();
    data->allocate_static_data_exc_vxc( 2, 2, terms, true );
    data->send_static_data_density_basis( P.data(), 2, nullptr, 0,
      nullptr, 0, nullptr, 0, union_basis );
    data->zero_exc_vxc_integrands( terms );
    data->allocate_static_data_multiparticle( 2, 1 );

    // Two species tasks per grid task
    std::vector<XCTask> grid_tasks( large_task_count / 2 + 1, tiny_task(2) );
    multiparticle_layout layout;
    data->build_species_tasks( terms, tracker, species_basis_ranges,
      union_basis_map, grid_tasks.begin(), grid_tasks.end(), layout );

    size_t next_task = 0;
    for( size_t ibatch = 0; ibatch < layout.batches.size(); ++ibatch ) {
      const auto& batch = layout.batches[ibatch];
      REQUIRE( batch.first_task == next_task );
      REQUIRE( batch.last_task > batch.first_task );
      CHECK( batch.last_task - batch.first_task <= grid_dim_z_max );
      data->generate_buffers_multiparticle( terms, union_basis_map, layout, ibatch );
      REQUIRE_NOTHROW( lwd->eval_uvars_lda( data.get(), RKS ) );
      REQUIRE( cudaDeviceSynchronize() == cudaSuccess );
      REQUIRE( cudaGetLastError() == cudaSuccess );
      next_task = batch.last_task;
    }
    CHECK( next_task == 2 * grid_tasks.size() );
  }
}

}
