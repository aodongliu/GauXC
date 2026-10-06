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
#include "device/local_device_work_driver.hpp"
#include "device/xc_device_aos_data.hpp"
#include "nvtx_range.hpp"
#include <gauxc/exceptions.hpp>
#include <gauxc/util/unused.hpp>

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

namespace GauXC  {
namespace detail {

/*
 * Device multiparticle (NEO) EXC/VXC. The species' shells form a union basis,
 * the density is block diagonal over it, and a batch is one task per
 * (grid task, species): the species tasks. Per batch:
 *   A  density and variables, over the whole list
 *   B  intra XC, over each intra species' points  [writes vrho there]
 *   C  the pair stage, over the batch's pairs     [adds into vrho]
 *   D  Z matrix and VXC, over the whole list
 */

namespace multiparticle_device_detail {

/// The species and the pairs of one call: the tracker, every intra species'
/// functional and the call's distinct pair functionals
struct multiparticle_call {
  multiparticle_tracker               tracker;           ///< the species and the pairs of the call
  std::vector<const functional_type*> intra_functionals; ///< per species: its intra functional, null without one
  std::vector<const functional_type*> pair_functionals;  ///< the call's distinct pair functionals, indexed by multiparticle_pair::functional_group
  std::vector<size_t>                 intra_species;     ///< the species carrying an intra functional, in species order
};

/// Resolve one call: a species participates through an active intra functional
/// or an active pair; pairs are grouped by their distinct functionals.
/// `vxc_is_output` false: every participating species builds its potential, none is returned
template <typename DensityList>
inline multiparticle_call resolve_multiparticle_call( const LoadBalancer& load_balancer,
  const DensityList& densities,
  const MultiParticleFunctionalSpec& functional_spec,
  const MultiParticleXCTerms& terms, bool vxc_is_output ) {

  const size_t nspecies = densities.size();
  const size_t npairs   = functional_spec.inter_functionals.size();

  std::vector<bool> active_intra( nspecies, false );
  std::vector<bool> active_inter( npairs, false );
  std::vector<bool> build_vxc( nspecies, not vxc_is_output );
  for( auto p : terms.active_intra ) active_intra[p] = true;
  for( auto i : terms.active_inter ) active_inter[i] = true;
  if( vxc_is_output )
    for( auto p : terms.vxc_targets ) build_vxc[p] = true;

  multiparticle_call call;
  auto& tracker = call.tracker;
  tracker.species.resize( nspecies );
  tracker.pairs.resize( npairs );
  tracker.do_vxc = vxc_is_output ? not terms.vxc_targets.empty() : true;
  call.intra_functionals.assign( nspecies, nullptr );

  for( size_t p = 0; p < nspecies; ++p ) {
    const auto& basis   = load_balancer.basis(p);
    auto&       species = tracker.species[p];
    species.nbf       = static_cast<int32_t>( basis.nbf() );
    species.ks_scheme = densities[p].Pz ? UKS : RKS;
    species.xc_approx = LDA;
    species.build_vxc = build_vxc[p];

    const auto* functionals = p < functional_spec.intra_functionals.size() ?
      &functional_spec.intra_functionals[p] : nullptr;

    if( active_intra[p] and functionals and not functionals->empty() ) {
      // One intra functional per species
      if( functionals->size() > 1 )
        GAUXC_GENERIC_EXCEPTION("Device MultiParticle: multiple intra functionals per species NYI");

      const auto& functional = *functionals->front();
      if( functional.is_mgga() )
        GAUXC_GENERIC_EXCEPTION("MultiParticle mGGA intra-XC is not implemented");

      species.has_intra_functional = true;
      species.participates         = true;
      species.xc_approx            = functional.is_gga() ? GGA : LDA;
      call.intra_functionals[p]    = functionals->front().get();
      call.intra_species.push_back(p);
    }
  }

  for( size_t i = 0; i < npairs; ++i ) {
    const auto& pair_functional = functional_spec.inter_functionals[i];
    auto&       pair            = tracker.pairs[i];
    pair.electron = pair_functional.electron;
    pair.particle = pair_functional.particle;
    pair.active   = active_inter[i] and not pair_functional.functionals.empty();

    for( const auto& functional : pair_functional.functionals )
      if( functional->is_gga() or functional->is_mgga() )
        GAUXC_GENERIC_EXCEPTION("MultiParticle GGA/mGGA inter-XC is not implemented");
    // One functional per pair
    if( pair.active and pair_functional.functionals.size() > 1 )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle: multiple inter functionals per pair NYI");

    if( pair.active ) {
      tracker.species[pair.electron].participates = true;
      tracker.species[pair.particle].participates = true;

      const auto* functional = pair_functional.functionals.front().get();
      auto entry = std::find( call.pair_functionals.begin(),
        call.pair_functionals.end(), functional );
      pair.functional_group = std::distance( call.pair_functionals.begin(), entry );
      if( entry == call.pair_functionals.end() )
        call.pair_functionals.push_back( functional );
    }
  }

  return call;
}

/// Whether a species' density can be carried in one channel exactly: high spin
/// (Ps == Pz bitwise), LD == nbf, and no intra functional of its own
inline bool carried_in_one_channel( const double* Ps, int64_t ldps,
                                    const double* Pz, int64_t ldpz,
                                    bool has_intra_functional, int64_t nbf ) {
  if( not Pz or has_intra_functional ) return false;
  if( ldps != nbf or ldpz != nbf ) return false;
  return std::memcmp( Ps, Pz, size_t(nbf) * size_t(nbf) * sizeof(double) ) == 0;
}

/// The one storage shape the whole species-task list is carried in
struct storage_shape {
  integrator_term_tracker terms;           ///< ks_scheme (RKS: one channel, UKS: two) and xc_approx of the shape
  bool                    scalar   = true; ///< one channel serves every species
  double                  xmat_fac = 2.0;  ///< the X matrix's spin factor: 2 for one channel, 1 for two
  std::vector<bool>       one_channel;     ///< per species: its density is carried in one channel exactly (Ps=Pz, avoid duplicate computation)
  inline bool two_channel() const { return not scalar; }
  inline bool is_gga() const { return terms.xc_approx == GGA; }
};

/// The storage shape of the whole list: scalar when every species is one
/// channel (RKS, or high spin with Ps == Pz), two channels otherwise; GGA when
/// any intra functional is GGA
template <typename DensityList>
inline storage_shape resolve_storage_shape( const multiparticle_tracker& tracker,
  const DensityList& densities ) {

  const size_t nspecies = tracker.nspecies();
  storage_shape shape;
  shape.one_channel.assign( nspecies, false );

  for( size_t p = 0; p < nspecies; ++p ) {
    const auto& species = tracker.species[p];
    if( not species.participates or not densities[p].Pz ) continue;
    shape.one_channel[p] = carried_in_one_channel( densities[p].Ps,
      densities[p].ldps, densities[p].Pz, densities[p].ldpz,
      species.has_intra_functional, static_cast<int64_t>(species.nbf) );
    if( not shape.one_channel[p] ) shape.scalar = false;
  }

  shape.terms.ks_scheme = shape.scalar ? RKS : UKS;
  shape.terms.xc_approx = LDA;
  for( size_t p = 0; p < nspecies; ++p )
    if( tracker.species[p].participates and tracker.species[p].xc_approx == GGA )
      shape.terms.xc_approx = GGA;

  shape.xmat_fac = shape.scalar ? 2.0 : 1.0;
  return shape;
}

/// Concatenate every participating species' shells into the union basis and
/// record where each species starts in it
inline BasisSet<double> build_union_basis( const multiparticle_tracker& tracker,
  const LoadBalancer& load_balancer,
  std::vector<species_basis_range>& species_basis_ranges ) {

  const size_t nspecies = tracker.nspecies();
  species_basis_ranges.assign( nspecies, species_basis_range{} );
  BasisSet<double> union_basis;
  int32_t shell_offset = 0, basis_function_offset = 0;
  for( size_t p = 0; p < nspecies; ++p ) {
    if( not tracker.species[p].participates ) continue;
    species_basis_ranges[p].shell_offset          = shell_offset;
    species_basis_ranges[p].basis_function_offset = basis_function_offset;
    const auto& basis = load_balancer.basis(p);
    union_basis.insert( union_basis.end(), basis.begin(), basis.end() );
    shell_offset          += static_cast<int32_t>( basis.nshells() );
    basis_function_offset += static_cast<int32_t>( basis.nbf() );
  }
  return union_basis;
}

/// The block-diagonal density over the union basis. A scalar block holds the
/// half density, P for RKS and Ps/2 for a high-spin particle; a two-channel
/// block holds (2P, 0) for RKS and (Ps, Pz) otherwise
template <typename DensityList>
inline void build_union_density( const multiparticle_tracker& tracker,
  const DensityList& densities,
  const std::vector<species_basis_range>& species_basis_ranges,
  const storage_shape& shape, int32_t nbf_union,
  std::vector<double>& Ps_union, std::vector<double>& Pz_union ) {

  const size_t nbf_union_squared = size_t(nbf_union) * size_t(nbf_union);
  Ps_union.assign( nbf_union_squared, 0.0 );
  if( shape.two_channel() ) Pz_union.assign( nbf_union_squared, 0.0 );

  for( size_t p = 0; p < tracker.nspecies(); ++p ) {
    if( not tracker.species[p].participates ) continue;
    const size_t  nbf    = static_cast<size_t>( tracker.species[p].nbf );
    const size_t  offset = static_cast<size_t>( species_basis_ranges[p].basis_function_offset );
    const double* Ps     = densities[p].Ps;
    const double* Pz     = densities[p].Pz;
    const double  factor = shape.scalar ? ( Pz ? 0.5 : 1.0 )
                                        : ( Pz ? 1.0 : 2.0 );
    for( size_t j = 0; j < nbf; ++j )
      for( size_t i = 0; i < nbf; ++i ) {
        Ps_union[ (offset+j)*nbf_union + (offset+i) ] = factor * Ps[ j*nbf + i ];
        if( shape.two_channel() and Pz )
          Pz_union[ (offset+j)*nbf_union + (offset+i) ] = Pz[ j*nbf + i ];
      }
  }
}

/// Stable sort of the grid tasks by XCTask::cost_exc_vxc, which sums over the species
/// Each cost is evaluated once; a list already in order is left in place
inline void sort_tasks_by_cost( std::vector<XCTask>::iterator task_begin,
  std::vector<XCTask>::iterator task_end ) {
  const size_t ntasks = std::distance( task_begin, task_end );
  std::vector<size_t> cost( ntasks );
  std::transform( task_begin, task_end, cost.begin(),
    []( const XCTask& task ) { return task.cost_exc_vxc(0); } );
  if( std::is_sorted( cost.begin(), cost.end(), std::greater<size_t>() ) ) return;

  std::vector<size_t> order( ntasks );
  std::iota( order.begin(), order.end(), size_t(0) );
  std::stable_sort( order.begin(), order.end(),
    [&]( size_t i, size_t j ) { return cost[i] > cost[j]; } );

  std::vector<XCTask> sorted;
  sorted.reserve( ntasks );
  for( auto i : order ) sorted.push_back( std::move( task_begin[i] ) );
  std::move( sorted.begin(), sorted.end(), task_begin );
}

/// Keep the setup when its key matches this call, else rebuild the union
/// basis and its map. Returns whether the species-task layout must be rebuilt
inline bool update_setup( multiparticle_setup& setup,
  const multiparticle_tracker& tracker, integrator_term_tracker terms,
  const LoadBalancer& load_balancer,
  std::vector<XCTask>::iterator task_begin,
  std::vector<XCTask>::iterator task_end ) {

  std::vector<const void*> task_ids = { task_begin == task_end ? nullptr : &*task_begin };
  for( auto task = task_begin; task != task_end; ++task )
    task_ids.push_back( task->points.data() );

  std::vector<int64_t> call_key = { tracker.do_vxc, terms.weights, terms.exc_vxc,
    terms.exc_grad, terms.xc_approx, terms.ks_scheme };
  for( const auto& species : tracker.species )
    call_key.insert( call_key.end(), { species.participates, species.has_intra_functional } );
  for( const auto& pair : tracker.pairs )
    call_key.insert( call_key.end(), { pair.active, int64_t(pair.electron),
      int64_t(pair.particle), int64_t(pair.functional_group) } );

  if( setup.valid and task_ids == setup.task_ids and call_key == setup.call_key )
    return false;

  setup.valid    = false;
  setup.task_ids = std::move( task_ids );
  setup.call_key = std::move( call_key );
  setup.union_basis = build_union_basis( tracker, load_balancer,
    setup.species_basis_ranges );
  setup.union_basis_map.emplace( setup.union_basis, load_balancer.molecule() );
  setup.layout = multiparticle_layout{};
  return true;
}

} // namespace multiparticle_device_detail


template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_vxc_( const std::vector<multiparticle_density>& densities,
                 const MultiParticleFunctionalSpec& functional_spec,
                 const MultiParticleXCTerms& terms,
                 std::vector<multiparticle_vxc>& vxc,
                 value_type* intra_exc,
                 value_type* inter_pair_exc,
                 const IntegratorSettingsXC& ks_settings ) {

  const size_t nspecies = densities.size();
  const size_t npairs   = functional_spec.inter_functionals.size();

  // ---- Validation; on the device every leading dimension must equal NBF
  if( nspecies == 0 )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires at least one density");
  if( nspecies != vxc.size() )
    GAUXC_GENERIC_EXCEPTION("MultiParticle density/VXC size mismatch");
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
  for( auto p : terms.vxc_targets )
    if( p >= nspecies )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle VXC target index");

  std::vector<bool> build_vxc( nspecies, false );
  for( auto p : terms.vxc_targets ) build_vxc[p] = true;

  for( size_t p = 0; p < nspecies; ++p ) {
    const int64_t nbf     = this->load_balancer_->basis(p).nbf();
    const auto&   density = densities[p];
    const bool    has_Pz  = density.Pz != nullptr;
    if( density.m != density.n )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps must be square");
    if( density.m != nbf )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps dimension must match its basis");
    if( density.ldps < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPS");
    if( has_Pz and density.ldpz < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPZ");
    if( build_vxc[p] and (not vxc[p].VXCs or vxc[p].ldvxcs < nbf) )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDVXCS");
    if( build_vxc[p] and has_Pz and (not vxc[p].VXCz or vxc[p].ldvxcz < nbf) )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDVXCZ");

    if( density.ldps != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDPS == NBF");
    if( has_Pz and density.ldpz != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDPZ == NBF");
    if( build_vxc[p] and vxc[p].ldvxcs != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDVXCS == NBF");
    if( build_vxc[p] and has_Pz and vxc[p].ldvxcz != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDVXCZ == NBF");
  }

  for( size_t i = 0; i < npairs; ++i ) {
    const auto& pair = functional_spec.inter_functionals[i];
    if( pair.electron >= nspecies or pair.particle >= nspecies )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle inter functional index");
  }
  if( not this->reduction_driver_->takes_host_memory() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC Only Works With Host Reductions");
  if( this->reduction_driver_->takes_device_memory() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC does not support device-memory reductions");

  auto& tasks = this->load_balancer_->get_tasks();

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get());
  if( not lwd )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires a device local work driver");
  if( not lwd->supports_multiparticle() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC is not implemented for this local work driver");

  auto rt = detail::as_device_runtime(this->load_balancer_->runtime());
  auto device_data_ptr = lwd->create_device_data(rt);

  GAUXC_MPI_CODE( MPI_Barrier(rt.comm()); )

  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    multiparticle_exc_vxc_local_work_( densities, functional_spec, terms, vxc,
      intra_exc, inter_pair_exc, ks_settings, tasks.begin(), tasks.end(),
      *device_data_ptr );
  });

  GAUXC_MPI_CODE(
  this->timer_.time_op("XCIntegrator.ImbalanceWait_EXC_VXC",[&](){
    MPI_Barrier(this->load_balancer_->runtime().comm());
  });
  )

  this->timer_.time_op("XCIntegrator.Allreduce", [&](){
    for( size_t p = 0; p < nspecies; ++p ) {
      if( not build_vxc[p] ) continue;
      const int64_t nbf = this->load_balancer_->basis(p).nbf();
      this->reduction_driver_->allreduce_inplace( vxc[p].VXCs, nbf * nbf, ReductionOp::Sum );
      if( densities[p].Pz )
        this->reduction_driver_->allreduce_inplace( vxc[p].VXCz, nbf * nbf, ReductionOp::Sum );
    }

    this->reduction_driver_->allreduce_inplace( intra_exc, nspecies, ReductionOp::Sum );
    if( npairs )
      this->reduction_driver_->allreduce_inplace( inter_pair_exc, npairs, ReductionOp::Sum );
  });

}


template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  multiparticle_exc_vxc_local_work_(
    const std::vector<multiparticle_density>& densities,
    const MultiParticleFunctionalSpec& functional_spec,
    const MultiParticleXCTerms& terms,
    std::vector<multiparticle_vxc>& vxc,
    value_type* intra_exc,
    value_type* inter_pair_exc,
    const IntegratorSettingsXC& settings,
    host_task_iterator task_begin, host_task_iterator task_end,
    XCDeviceData& device_data ) {

  // The KS settings carry nothing this path consumes, as on the host.
  GauXC::util::unused(settings);

  const size_t nspecies = densities.size();
  const size_t npairs   = functional_spec.inter_functionals.size();

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get());
  if( not lwd )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires a device local work driver");

  nvtx_range call_range, phase;
  call_range.next( "multiparticle exc_vxc" );

  // ---- The species and the pairs of this call, resolved once
  auto call = multiparticle_device_detail::resolve_multiparticle_call(
    *this->load_balancer_, densities, functional_spec, terms, true );
  const auto& tracker = call.tracker;

  for( size_t p = 0; p < nspecies; ++p ) intra_exc[p] = 0.0;
  for( size_t i = 0; i < npairs; ++i )   inter_pair_exc[i] = 0.0;

  phase.next( "setup: sort tasks" );
  multiparticle_device_detail::sort_tasks_by_cost( task_begin, task_end );

  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored )
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");

  // Zero the caller's VXC blocks; a target that participates in nothing stays zero
  if( tracker.do_vxc )
  for( size_t p = 0; p < nspecies; ++p ) {
    if( not tracker.species[p].build_vxc ) continue;
    const size_t nbf = static_cast<size_t>( tracker.species[p].nbf );
    std::fill( vxc[p].VXCs, vxc[p].VXCs + nbf * nbf, 0.0 );
    if( densities[p].Pz )
      std::fill( vxc[p].VXCz, vxc[p].VXCz + nbf * nbf, 0.0 );
  }

  // ---- The union basis, the storage shape and the block-diagonal density
  phase.next( "setup: union basis" );
  auto shape = multiparticle_device_detail::resolve_storage_shape( tracker,
    densities );
  shape.terms.exc_vxc = true;
  const bool is_gga      = shape.is_gga();
  const bool two_channel = shape.two_channel();

  auto& setup = this->multiparticle_exc_vxc_setup_;
  const bool build_layout = multiparticle_device_detail::update_setup( setup,
    tracker, shape.terms, *this->load_balancer_, task_begin, task_end );
  const auto& species_basis_ranges = setup.species_basis_ranges;
  const auto& union_basis          = setup.union_basis;
  const auto& union_basis_map      = *setup.union_basis_map;
  auto&       layout               = setup.layout;
  const int32_t nbf_union         = static_cast<int32_t>( union_basis.nbf() );
  const int32_t nshells_union     = static_cast<int32_t>( union_basis.nshells() );
  const size_t  nbf_union_squared = size_t(nbf_union) * size_t(nbf_union);

  phase.next( "setup: union density" );
  std::vector<double> Ps_union, Pz_union;
  multiparticle_device_detail::build_union_density( tracker, densities,
    species_basis_ranges, shape, nbf_union, Ps_union, Pz_union );

  phase.next( "setup: static data" );
  device_data.reset_allocations();
  device_data.allocate_static_data_exc_vxc( nbf_union, nshells_union, shape.terms,
    tracker.do_vxc );
  device_data.send_static_data_density_basis( Ps_union.data(), nbf_union,
    two_channel ? Pz_union.data() : nullptr, two_channel ? nbf_union : 0,
    nullptr, 0, nullptr, 0, union_basis );
  device_data.zero_exc_vxc_integrands( shape.terms );
  device_data.allocate_static_data_multiparticle( nspecies, npairs );

  // ---- The species-task list and its batches
  phase.next( "setup: species tasks" );
  if( build_layout ) {
    device_data.build_species_tasks( shape.terms, tracker, species_basis_ranges,
      union_basis_map, task_begin, task_end, layout );
    setup.valid = true;
  }

  phase.end();

  for( size_t ibatch = 0; ibatch < layout.batches.size(); ++ibatch ) {
    const auto& batch = layout.batches[ibatch];
    nvtx_range batch_range, stage;
    batch_range.next( "batch" );
    stage.next( "generate buffers" );
    device_data.generate_buffers_multiparticle( shape.terms, union_basis_map,
      layout, ibatch );

    //-------------- A: density and variables, over the whole list ----------
    stage.next( "collocation" );
    if( is_gga ) lwd->eval_collocation_gradient( &device_data );
    else         lwd->eval_collocation( &device_data );

    // The V variables and the Z matrix launch once per work range; the X
    // matrix and the VXC accumulation once per batch
    auto do_xmat_vvar = [&]( density_id den_id ) {
      stage.next( "xmat" );
      lwd->eval_xmat( shape.xmat_fac, &device_data, false, den_id );
      stage.next( "vvars" );
      for( const auto& range : batch.task_work_ranges ) {
        if( is_gga ) lwd->eval_vvars_gga( &device_data, den_id, range );
        else         lwd->eval_vvars_lda( &device_data, den_id, range );
      }
    };
    do_xmat_vvar( DEN_S );
    if( two_channel ) do_xmat_vvar( DEN_Z );

    // U variables: (rho_s, rho_z) -> (rho_alpha, rho_beta) and gamma; a no-op for scalar LDA
    stage.next( "uvars" );
    if( is_gga )           lwd->eval_uvars_gga( &device_data, shape.terms.ks_scheme );
    else if( two_channel ) lwd->eval_uvars_lda( &device_data, UKS );

    //-------------- B: intra XC, one call per intra species over its points --
    for( size_t p : call.intra_species ) {
      const auto& points = batch.intra_point_ranges[p];
      if( not points.count ) continue;
      stage.next( "intra functional" );
      // The kernel follows the species' functional, not the storage shape
      if( tracker.species[p].xc_approx == GGA )
        lwd->eval_kern_exc_vxc_gga( *call.intra_functionals[p], &device_data, points );
      else
        lwd->eval_kern_exc_vxc_lda( *call.intra_functionals[p], &device_data, points );
      stage.next( "intra energy" );
      lwd->inc_exc( &device_data, points, device_data.intra_exc_device_data(p) );
    }

    //-------------- C: the pair stage, over the batch's pairs --------------
    //                 ADDS into vrho, so it must follow B. One evaluation per
    //                 distinct pair functional, however many pairs share it.
    if( not batch.pair_point_ranges.empty() ) {
      stage.next( "pair gather" );
      lwd->gather_pair_densities( &device_data );
      stage.next( "pair functional" );
      for( size_t functional_group = 0;
           functional_group < batch.pair_functional_ranges.size(); ++functional_group ) {
        const auto& points = batch.pair_functional_ranges[functional_group];
        if( not points.count ) continue;
        lwd->eval_pair_functional( *call.pair_functionals[functional_group], &device_data, points );
      }
      stage.next( "pair scatter" );
      lwd->scatter_pair_potential( &device_data );
      stage.next( "pair energy" );
      lwd->sum_pair_energy( &device_data );
    }

    //-------------- D: Z matrix and VXC over the whole list ---------------
    if( tracker.do_vxc ) {
      auto do_zmat_vxc = [&]( density_id den_id ) {
        const auto scheme = shape.terms.ks_scheme;
        stage.next( "zmat" );
        for( const auto& range : batch.task_work_ranges ) {
          if( is_gga ) lwd->eval_zmat_gga_vxc( &device_data, scheme, den_id, range );
          else         lwd->eval_zmat_lda_vxc( &device_data, scheme, den_id, range );
        }
        stage.next( "vxc syr2k" );
        lwd->inc_vxc( &device_data, den_id, false, batch.task_work_ranges );
      };
      do_zmat_vxc( DEN_S );
      if( two_channel ) do_zmat_vxc( DEN_Z );
    }
  }

  phase.next( "copy back" );
  if( tracker.do_vxc ) {
    lwd->symmetrize_vxc( &device_data, DEN_S );
    if( two_channel ) lwd->symmetrize_vxc( &device_data, DEN_Z );
  }

  auto rt = detail::as_device_runtime(this->load_balancer_->runtime());
  rt.device_backend()->master_queue_synchronize();

  double EXC_unused = 0.0, N_EL_total = 0.0;
  std::vector<double> VXCs_union, VXCz_union;
  if( tracker.do_vxc ) {
    VXCs_union.resize( nbf_union_squared );
    if( two_channel ) VXCz_union.resize( nbf_union_squared );
  }

  this->timer_.time_op("XCIntegrator.DeviceToHostCopy_EXC_VXC",[&](){
    device_data.retrieve_exc_vxc_integrands( &EXC_unused, &N_EL_total,
      tracker.do_vxc ? VXCs_union.data() : nullptr, nbf_union,
      (tracker.do_vxc and two_channel) ? VXCz_union.data() : nullptr,
      two_channel ? nbf_union : 0, nullptr, 0, nullptr, 0 );
    rt.device_backend()->master_queue_synchronize();
    device_data.retrieve_intra_exc( intra_exc, nspecies );
    device_data.retrieve_pair_exc( inter_pair_exc, npairs );
  });

  // Cut the union VXC into per-species blocks. A one-channel species returns
  // the same matrix in both channels, half its one accumulator
  if( tracker.do_vxc )
  for( size_t p = 0; p < nspecies; ++p ) {
    const auto& species = tracker.species[p];
    if( not species.participates or not species.build_vxc ) continue;
    const size_t nbf    = static_cast<size_t>( species.nbf );
    const size_t offset = static_cast<size_t>( species_basis_ranges[p].basis_function_offset );
    if( shape.one_channel[p] ) {
      for( size_t j = 0; j < nbf; ++j )
        for( size_t i = 0; i < nbf; ++i ) {
          const double v = 0.5 * VXCs_union[ (offset+j)*nbf_union + (offset+i) ];
          vxc[p].VXCs[ j*nbf + i ] = v;
          vxc[p].VXCz[ j*nbf + i ] = v;
        }
      continue;
    }
    for( size_t j = 0; j < nbf; ++j )
      for( size_t i = 0; i < nbf; ++i )
        vxc[p].VXCs[ j*nbf + i ] = VXCs_union[ (offset+j)*nbf_union + (offset+i) ];
    if( not densities[p].Pz ) continue;
    for( size_t j = 0; j < nbf; ++j )
      for( size_t i = 0; i < nbf; ++i )
        vxc[p].VXCz[ j*nbf + i ] = VXCz_union[ (offset+j)*nbf_union + (offset+i) ];
  }

}

}
}
