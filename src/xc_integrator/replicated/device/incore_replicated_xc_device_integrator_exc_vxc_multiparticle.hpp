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
#include <gauxc/exceptions.hpp>
#include <gauxc/util/unused.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace GauXC  {
namespace detail {

/*
 * Device multiparticle (NEO) EXC/VXC. The species' shells form a union basis,
 * the density is block diagonal over it, and a batch is one task per
 * (grid task, species): the species tasks. Per batch:
 *   A  density and variables, over the whole list
 *   B  intra XC, over each intra species' points  [writes vrho there]
 *   C  the pair stage, over the batch's pairs     [adds into vrho; not evaluated yet]
 *   D  Z matrix and VXC, over the whole list
 */

namespace multiparticle_device_detail {

/// Whether a quantum particle's density can be carried in one channel exactly:
/// high spin (Ps == Pz bitwise), LD == nbf, no intra functional of its own, and
/// not the electron of an active pair (the pair potential reaches both of the
/// electron's spin channels)
inline bool carried_in_one_channel( const double* Ps, int64_t ldps,
                                    const double* Pz, int64_t ldpz,
                                    bool has_intra_functional, bool is_pair_electron,
                                    int64_t nbf ) {
  if( not Pz or has_intra_functional or is_pair_electron ) return false;
  if( ldps != nbf or ldpz != nbf ) return false;
  return std::memcmp( Ps, Pz, size_t(nbf) * size_t(nbf) * sizeof(double) ) == 0;
}

/// The one storage shape the whole species-task list is carried in
struct storage_shape {
  integrator_term_tracker terms;           ///< ks_scheme (RKS: one channel, UKS: two) and xc_approx of the shape
  std::vector<char>       one_channel;     ///< per species: its density is carried in one channel
  bool                    scalar   = true; ///< one channel serves every species
  double                  xmat_fac = 2.0;  ///< the X matrix's spin factor: 2 for one channel, 1 for two
  inline bool two_channel() const { return not scalar; }
  inline bool is_gga() const { return terms.xc_approx == GGA; }
};

/// The storage shape of the whole list: scalar when every species is one
/// channel (RKS, or high spin with Ps == Pz), two channels otherwise; GGA when
/// any intra functional is GGA
template <typename DensityList>
inline storage_shape resolve_storage_shape( const multiparticle_tracker& tracker,
  const DensityList& densities, const std::vector<char>& is_pair_electron ) {

  const size_t nspecies = tracker.nspecies();
  storage_shape shape;
  shape.one_channel.assign( nspecies, 0 );

  for( size_t p = 0; p < nspecies; ++p ) {
    const auto& species = tracker.species[p];
    if( not species.participates or not densities[p].Pz ) continue;
    shape.one_channel[p] = carried_in_one_channel( densities[p].Ps, densities[p].ldps,
      densities[p].Pz, densities[p].ldpz, species.has_intra_functional,
      is_pair_electron[p] != 0, static_cast<int64_t>(species.nbf) );
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
inline void sort_tasks_by_cost( std::vector<XCTask>::iterator task_begin,
  std::vector<XCTask>::iterator task_end ) {
  std::stable_sort( task_begin, task_end,
    []( const XCTask& a, const XCTask& b ) {
      return a.cost_exc_vxc(0) > b.cost_exc_vxc(0);
    });
}

/// GAUXC_MULTIPARTICLE_TRACE=1: print the shape of the species-task list, per call
inline bool trace_enabled() {
  static const bool enabled = [](){
    const char* v = std::getenv("GAUXC_MULTIPARTICLE_TRACE");
    return v and std::string(v) != "0";
  }();
  return enabled;
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

  std::vector<bool> active_intra( nspecies, false );
  std::vector<bool> active_inter( npairs, false );
  std::vector<bool> build_vxc( nspecies, false );
  for( auto p : terms.active_intra ) active_intra[p] = true;
  for( auto i : terms.active_inter ) active_inter[i] = true;
  for( auto p : terms.vxc_targets )  build_vxc[p]    = true;

  // ---- The species and the pairs of this call, resolved once. A species
  //      participates through an active intra functional or an active pair.
  multiparticle_tracker tracker;
  tracker.species.resize( nspecies );
  tracker.pairs.resize( npairs );
  tracker.do_vxc = not terms.vxc_targets.empty();

  std::vector<const functional_type*> intra_functional( nspecies, nullptr );

  for( size_t p = 0; p < nspecies; ++p ) {
    const auto& basis   = this->load_balancer_->basis(p);
    auto&       species = tracker.species[p];
    species.index     = p;
    species.nbf       = static_cast<int32_t>( basis.nbf() );
    species.nshells   = static_cast<int32_t>( basis.nshells() );
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
      intra_functional[p]          = functionals->front().get();
    }
  }

  std::vector<char> is_pair_electron( nspecies, 0 );
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
      is_pair_electron[pair.electron] = 1;
    }
  }

  for( size_t p = 0; p < nspecies; ++p ) intra_exc[p] = 0.0;
  for( size_t i = 0; i < npairs; ++i )   inter_pair_exc[i] = 0.0;

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
  std::vector<size_t> intra_species;
  for( size_t p = 0; p < nspecies; ++p )
    if( tracker.species[p].has_intra_functional ) intra_species.push_back(p);

  std::vector<species_basis_range> species_basis_ranges;
  BasisSet<double> union_basis = multiparticle_device_detail::build_union_basis(
    tracker, *this->load_balancer_, species_basis_ranges );
  const int32_t nbf_union         = static_cast<int32_t>( union_basis.nbf() );
  const int32_t nshells_union     = static_cast<int32_t>( union_basis.nshells() );
  const size_t  nbf_union_squared = size_t(nbf_union) * size_t(nbf_union);

  auto shape = multiparticle_device_detail::resolve_storage_shape( tracker, densities, is_pair_electron );
  shape.terms.exc_vxc = true;
  const bool is_gga      = shape.is_gga();
  const bool two_channel = shape.two_channel();

  std::vector<double> Ps_union, Pz_union;
  multiparticle_device_detail::build_union_density( tracker, densities,
    species_basis_ranges, shape, nbf_union, Ps_union, Pz_union );

  device_data.reset_allocations();
  device_data.allocate_static_data_exc_vxc( nbf_union, nshells_union, shape.terms,
    tracker.do_vxc );
  device_data.send_static_data_density_basis( Ps_union.data(), nbf_union,
    two_channel ? Pz_union.data() : nullptr, two_channel ? nbf_union : 0,
    nullptr, 0, nullptr, 0, union_basis );
  device_data.zero_exc_vxc_integrands( shape.terms );
  device_data.allocate_static_data_multiparticle( nspecies );

  BasisSetMap union_basis_map( union_basis, this->load_balancer_->molecule() );

  // ---- The species-task list and its batches
  multiparticle_layout layout;
  device_data.build_species_tasks( shape.terms, tracker, species_basis_ranges,
    union_basis_map, task_begin, task_end, layout );

  // The trace: the shape of the list, per batch
  if( multiparticle_device_detail::trace_enabled() )
  for( const auto& batch : layout.batches ) {
    std::cout << "[GAUXC MULTIPARTICLE] batch species_tasks="
              << (batch.last_task - batch.first_task)
              << " npts_grid=" << batch.npts_grid
              << " npts_total=" << batch.npts_total
              << " npts_intra=" << batch.npts_intra
              << " work_ranges=" << batch.task_work_ranges.size();
    for( const auto& range : batch.task_work_ranges )
      std::cout << " [" << range.count << "@nbe<=" << range.nbe_max << "]";
    std::cout << std::endl;
  }

  size_t nlaunch = 0;   // work-driver entries issued for the last batch

  for( size_t ibatch = 0; ibatch < layout.batches.size(); ++ibatch ) {
    const auto& batch = layout.batches[ibatch];
    device_data.generate_buffers_multiparticle( shape.terms, union_basis_map,
      layout, ibatch );

    nlaunch = 0;

    //-------------- A: density and variables, over the whole list ----------
    if( is_gga ) lwd->eval_collocation_gradient( &device_data );
    else         lwd->eval_collocation( &device_data );
    nlaunch++;

    // The V variables and the Z matrix launch once per work range; the X
    // matrix and the VXC accumulation once per batch
    auto do_xmat_vvar = [&]( density_id den_id ) {
      lwd->eval_xmat( shape.xmat_fac, &device_data, false, den_id );
      nlaunch++;
      for( const auto& range : batch.task_work_ranges ) {
        if( is_gga ) lwd->eval_vvars_gga( &device_data, den_id, range );
        else         lwd->eval_vvars_lda( &device_data, den_id, range );
        nlaunch++;
      }
    };
    do_xmat_vvar( DEN_S );
    if( two_channel ) do_xmat_vvar( DEN_Z );

    // U variables: (rho_s, rho_z) -> (rho_alpha, rho_beta) and gamma; a no-op for scalar LDA
    if( is_gga ) {
      lwd->eval_uvars_gga( &device_data, shape.terms.ks_scheme );
      nlaunch++;
    } else if( two_channel ) {
      lwd->eval_uvars_lda( &device_data, UKS );
      nlaunch++;
    }

    //-------------- B: intra XC, one call per intra species over its points --
    for( size_t p : intra_species ) {
      const auto& points = batch.intra_point_ranges[p];
      if( not points.count ) continue;
      // The kernel follows the species' functional, not the storage shape
      if( tracker.species[p].xc_approx == GGA )
        lwd->eval_kern_exc_vxc_gga( *intra_functional[p], &device_data, points );
      else
        lwd->eval_kern_exc_vxc_lda( *intra_functional[p], &device_data, points );
      lwd->inc_exc( &device_data, points, device_data.intra_exc_device_data(p) );
      lwd->inc_nel( &device_data, points );
      nlaunch += 3;
    }

    //-------------- C: the pair stage; adds into vrho. Not evaluated yet ---

    //-------------- D: Z matrix and VXC over the whole list ---------------
    if( tracker.do_vxc ) {
      auto do_zmat_vxc = [&]( density_id den_id ) {
        const auto scheme = shape.terms.ks_scheme;
        for( const auto& range : batch.task_work_ranges ) {
          if( is_gga ) lwd->eval_zmat_gga_vxc( &device_data, scheme, den_id, range );
          else         lwd->eval_zmat_lda_vxc( &device_data, scheme, den_id, range );
          nlaunch++;
        }
        lwd->inc_vxc( &device_data, den_id, false );
        nlaunch++;
      };
      do_zmat_vxc( DEN_S );
      if( two_channel ) do_zmat_vxc( DEN_Z );
    }
  }

  if( tracker.do_vxc ) {
    lwd->symmetrize_vxc( &device_data, DEN_S );
    if( two_channel ) lwd->symmetrize_vxc( &device_data, DEN_Z );
  }

  auto rt = detail::as_device_runtime(this->load_balancer_->runtime());
  rt.device_backend()->master_queue_synchronize();

  if( multiparticle_device_detail::trace_enabled() )
    std::cout << "[GAUXC MULTIPARTICLE] nspecies=" << nspecies
              << " nbf_union=" << nbf_union
              << " nspecies_tasks=" << layout.species_tasks.size()
              << " nbatches=" << layout.batches.size()
              << " work_driver_entries_per_batch=" << nlaunch << std::endl;

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
  });

  // Cut the union VXC into per-species blocks; a particle carried in one
  // channel returns half of its one accumulator in each channel
  if( tracker.do_vxc )
  for( size_t p = 0; p < nspecies; ++p ) {
    const auto& species = tracker.species[p];
    if( not species.participates or not species.build_vxc ) continue;
    const size_t nbf    = static_cast<size_t>( species.nbf );
    const size_t offset = static_cast<size_t>( species_basis_ranges[p].basis_function_offset );
    const double scale  = ( shape.scalar and shape.one_channel[p] ) ? 0.5 : 1.0;
    for( size_t j = 0; j < nbf; ++j )
      for( size_t i = 0; i < nbf; ++i )
        vxc[p].VXCs[ j*nbf + i ] = scale * VXCs_union[ (offset+j)*nbf_union + (offset+i) ];
    if( not densities[p].Pz ) continue;
    for( size_t j = 0; j < nbf; ++j )
      for( size_t i = 0; i < nbf; ++i )
        vxc[p].VXCz[ j*nbf + i ] = shape.one_channel[p]
          ? vxc[p].VXCs[ j*nbf + i ]
          : VXCz_union[ (offset+j)*nbf_union + (offset+i) ];
  }

}

}
}
