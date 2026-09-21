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
#include "incore_replicated_xc_device_integrator.hpp"
#include "incore_replicated_xc_device_integrator_exc_vxc_multiparticle.hpp"
#include "device/local_device_work_driver.hpp"
#include "device/xc_device_aos_data.hpp"
#include <gauxc/exceptions.hpp>
#include <gauxc/basisset_map.hpp>

#include <iostream>
#include <vector>

namespace GauXC  {
namespace detail {

/*
 * Device multiparticle (NEO) EXC gradient. A species task is one ordinary
 * XCTask over the union basis, so one gradient accumulation covers the whole
 * species-task list and attributes every shell through the union basis map.
 * Per batch:
 *   A  collocation with its derivatives, the X matrix and the variables
 *   B  intra XC, over each intra species' points  [writes vrho there]
 *   C  the pair stage, over the batch's pairs     [adds into vrho]
 *   D  w_times_f per grid point                   [reads eps and the density]
 *   E  one gradient accumulation over the whole list
 *   F  the weight derivatives over the batch's grid [reads w_times_f]
 */

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_grad_( const std::vector<multiparticle_density>& densities,
                  const MultiParticleFunctionalSpec& functional_spec,
                  const MultiParticleXCTerms& terms,
                  value_type* EXC_GRAD,
                  const IntegratorSettingsXC& settings ) {

  const size_t nspecies = densities.size();
  const size_t npairs   = functional_spec.inter_functionals.size();

  // ---- Validation; on the device every leading dimension must equal NBF
  if( nspecies == 0 )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient requires at least one density");
  if( this->load_balancer_->basis_count() != nspecies )
    GAUXC_GENERIC_EXCEPTION("MultiParticle density count must match LoadBalancer basis count");
  if( functional_spec.intra_functionals.size() > nspecies )
    GAUXC_GENERIC_EXCEPTION("Too many MultiParticle intra functional entries");

  for( auto p : terms.active_intra )
    if( p >= functional_spec.intra_functionals.size() )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle active intra index");
  for( auto i : terms.active_inter )
    if( i >= npairs )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle active inter index");

  for( size_t p = 0; p < nspecies; ++p ) {
    const int64_t nbf     = this->load_balancer_->basis(p).nbf();
    const auto&   density = densities[p];
    if( density.m != density.n )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps must be square");
    if( density.m != nbf )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps dimension must match its basis");
    if( density.ldps != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC Gradient requires LDPS == NBF");
    if( density.Pz and density.ldpz != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC Gradient requires LDPZ == NBF");
  }

  for( size_t i = 0; i < npairs; ++i ) {
    const auto& pair = functional_spec.inter_functionals[i];
    if( pair.electron >= nspecies or pair.particle >= nspecies )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle inter functional index");
  }
  if( not this->reduction_driver_->takes_host_memory() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC Gradient Only Works With Host Reductions");
  if( this->reduction_driver_->takes_device_memory() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC Gradient does not support device-memory reductions");

  auto& tasks = this->load_balancer_->get_tasks();

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get());
  if( not lwd )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient requires a device local work driver");
  if( not lwd->supports_multiparticle() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC Gradient is not implemented for this local work driver");

  auto rt = detail::as_device_runtime(this->load_balancer_->runtime());
  auto device_data_ptr = this->timer_.time_op("XCIntegrator.DeviceAlloc",
    [&](){ return lwd->create_device_data(rt); });

  GAUXC_MPI_CODE( MPI_Barrier(rt.comm()); )

  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    multiparticle_exc_grad_local_work_( densities, functional_spec, terms,
      EXC_GRAD, settings, tasks.begin(), tasks.end(), *device_data_ptr );
  });

  GAUXC_MPI_CODE(
  this->timer_.time_op("XCIntegrator.ImbalanceWait_EXC_GRAD",[&](){
    MPI_Barrier(this->load_balancer_->runtime().comm());
  });
  )

  this->timer_.time_op("XCIntegrator.Allreduce", [&](){
    const int natoms = static_cast<int>( this->load_balancer_->molecule().natoms() );
    this->reduction_driver_->allreduce_inplace( EXC_GRAD, 3*natoms, ReductionOp::Sum );
  });

}


template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  multiparticle_exc_grad_local_work_(
    const std::vector<multiparticle_density>& densities,
    const MultiParticleFunctionalSpec& functional_spec,
    const MultiParticleXCTerms& terms,
    value_type* EXC_GRAD,
    const IntegratorSettingsXC& settings,
    host_task_iterator task_begin, host_task_iterator task_end,
    XCDeviceData& device_data ) {

  const size_t nspecies = densities.size();

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get());
  if( not lwd )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient requires a device local work driver");
  auto* stack_data = dynamic_cast<XCDeviceStackData*>(&device_data);
  if( not stack_data )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient requires a stack XCDeviceData");

  const auto&   mol     = this->load_balancer_->molecule();
  const auto&   molmeta = this->load_balancer_->molmeta();
  const int32_t natoms  = static_cast<int32_t>( mol.natoms() );

  IntegratorSettingsEXC_GRAD exc_grad_settings;
  if( auto* settings_exc_grad = dynamic_cast<const IntegratorSettingsEXC_GRAD*>(&settings) )
    exc_grad_settings = *settings_exc_grad;
  const bool do_weight_derivatives = exc_grad_settings.include_weight_derivatives;

  // ---- The species and the pairs of this call. The gradient contracts every
  //      participating species' potential and returns none of them
  auto call = multiparticle_device_detail::resolve_multiparticle_call(
    *this->load_balancer_, densities, functional_spec, terms, false );
  const auto& tracker = call.tracker;

  multiparticle_device_detail::sort_tasks_by_cost( task_begin, task_end );

  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored )
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");
  XCWeightAlg& weight_alg = lb_state.weight_alg;

  // ---- The union basis, the storage shape and the block-diagonal density
  std::vector<species_basis_range> species_basis_ranges;
  BasisSet<double> union_basis = multiparticle_device_detail::build_union_basis(
    tracker, *this->load_balancer_, species_basis_ranges );
  const int32_t nbf_union     = static_cast<int32_t>( union_basis.nbf() );
  const int32_t nshells_union = static_cast<int32_t>( union_basis.nshells() );

  auto shape = multiparticle_device_detail::resolve_storage_shape( tracker,
    densities, call.is_pair_electron );
  shape.terms.exc_grad = true;
  shape.terms.weights  = do_weight_derivatives;
  const bool is_gga      = shape.is_gga();
  const bool two_channel = shape.two_channel();

  std::vector<double> Ps_union, Pz_union;
  multiparticle_device_detail::build_union_density( tracker, densities,
    species_basis_ranges, shape, nbf_union, Ps_union, Pz_union );

  device_data.reset_allocations();
  device_data.allocate_static_data_exc_grad( nbf_union, nshells_union, natoms,
    shape.terms );
  device_data.send_static_data_density_basis( Ps_union.data(), nbf_union,
    two_channel ? Pz_union.data() : nullptr, two_channel ? nbf_union : 0,
    nullptr, 0, nullptr, 0, union_basis );
  if( do_weight_derivatives ) {
    device_data.allocate_static_data_weights( natoms );
    device_data.send_static_data_weights( mol, molmeta );
  }
  device_data.zero_exc_grad_integrands();

  BasisSetMap union_basis_map( union_basis, mol );

  // ---- The species-task list and its batches
  multiparticle_layout layout;
  device_data.build_species_tasks( shape.terms, tracker, species_basis_ranges,
    union_basis_map, task_begin, task_end, layout );

  size_t nlaunch = 0;   // work-driver entries issued for the last batch

  for( size_t ibatch = 0; ibatch < layout.batches.size(); ++ibatch ) {
    const auto& batch = layout.batches[ibatch];
    device_data.generate_buffers_multiparticle( shape.terms, union_basis_map,
      layout, ibatch );

    nlaunch = 0;

    //-------------- A: density and variables, over the whole list ----------
    // The gradient needs the first basis derivatives; a GGA intra functional
    // additionally needs the basis hessian
    if( is_gga ) lwd->eval_collocation_hessian ( &device_data );
    else         lwd->eval_collocation_gradient( &device_data );
    nlaunch++;

    auto do_xmat_vvar = [&]( density_id den_id ) {
      lwd->eval_xmat( shape.xmat_fac, &device_data, is_gga, den_id );
      nlaunch++;
      for( const auto& range : batch.task_work_ranges ) {
        if( is_gga ) lwd->eval_vvars_gga( &device_data, den_id, range );
        else         lwd->eval_vvars_lda( &device_data, den_id, range );
        nlaunch++;
      }
      // The polarized gradient assembly reads both channels' X matrix, which
      // the next eval_xmat would overwrite
      if( two_channel ) { lwd->save_xmat( &device_data, is_gga, den_id ); nlaunch++; }
    };
    do_xmat_vvar( DEN_S );
    if( two_channel ) do_xmat_vvar( DEN_Z );

    if( is_gga ) {
      lwd->eval_uvars_gga( &device_data, shape.terms.ks_scheme );
      nlaunch++;
    } else if( two_channel ) {
      lwd->eval_uvars_lda( &device_data, UKS );
      nlaunch++;
    }

    //-------------- B: intra XC, one call per intra species over its points --
    for( size_t p : call.intra_species ) {
      const auto& points = batch.intra_point_ranges[p];
      if( not points.count ) continue;
      if( tracker.species[p].xc_approx == GGA )
        lwd->eval_kern_exc_vxc_gga( *call.intra_functionals[p], &device_data, points );
      else
        lwd->eval_kern_exc_vxc_lda( *call.intra_functionals[p], &device_data, points );
      nlaunch++;
    }

    //-------------- C: the pair stage, over the batch's pairs --------------
    //                 ADDS into vrho, so it must follow B; the pair term
    //                 reaches the gradient through that potential
    if( not batch.pair_point_ranges.empty() ) {
      lwd->gather_pair_densities( &device_data );
      for( size_t functional_group = 0;
           functional_group < batch.pair_functional_ranges.size(); ++functional_group ) {
        const auto& points = batch.pair_functional_ranges[functional_group];
        if( not points.count ) continue;
        lwd->eval_pair_functional( *call.pair_functionals[functional_group],
          &device_data, points );
        nlaunch++;
      }
      lwd->scatter_pair_potential( &device_data );
      nlaunch += 2;
    }

    //-------------- D: w_times_f per grid point ----------------------------
    if( do_weight_derivatives ) {
      lwd->accumulate_w_times_f( &device_data );
      nlaunch++;
    }

    //-------------- E: one gradient accumulation over the whole list ------
    if( is_gga )
      lwd->inc_exc_grad_gga( &device_data, shape.terms.ks_scheme,
        do_weight_derivatives );
    else
      lwd->inc_exc_grad_lda( &device_data, shape.terms.ks_scheme,
        do_weight_derivatives );
    nlaunch++;

    //-------------- F: weight derivatives over the batch's grid -----------
    if( do_weight_derivatives ) {
      lwd->eval_weight_1st_deriv_contracted( &device_data, weight_alg,
        stack_data->w_times_f_device_, stack_data->grid_npts_ );
      nlaunch++;
    }
  }

  auto rt = detail::as_device_runtime(this->load_balancer_->runtime());
  rt.device_backend()->master_queue_synchronize();

  if( multiparticle_device_detail::trace_enabled() )
    std::cout << "[GAUXC MULTIPARTICLE GRAD] nspecies=" << nspecies
              << " nbf_union=" << nbf_union
              << " nspecies_tasks=" << layout.species_tasks.size()
              << " nbatches=" << layout.batches.size()
              << " work_driver_entries_per_batch=" << nlaunch << std::endl;

  this->timer_.time_op("XCIntegrator.DeviceToHostCopy_EXC_GRAD",[&](){
    value_type N_EL_total = 0.0;
    device_data.retrieve_exc_grad_integrands( EXC_GRAD, &N_EL_total );
    rt.device_backend()->master_queue_synchronize();
  });

}

}
}
