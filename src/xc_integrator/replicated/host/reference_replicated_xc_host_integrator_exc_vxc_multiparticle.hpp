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

#include "reference_replicated_xc_host_integrator.hpp"
#include "integrator_util/integrator_common.hpp"
#include "host/local_host_work_driver.hpp"

#include <algorithm>
#include <cstring>

namespace GauXC::detail {

/// MultiParticle EXC/VXC: per-rank local work followed by the global reduction,
/// matching the single-particle exc_vxc convention.
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( const std::vector<multiparticle_density>& densities,
                 const MultiParticleFunctionalSpec& functional_spec,
                 const MultiParticleXCTerms& terms,
                 std::vector<multiparticle_vxc>& vxc,
                 value_type* intra_exc,
                 value_type* inter_pair_exc,
                 const IntegratorSettingsXC& ks_settings ) {

  const size_t np     = densities.size();
  const size_t ninter = functional_spec.inter_functionals.size();

  // Check that densities / VXC are sane
  if( np == 0 )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires at least one density");
  if( np != vxc.size() )
    GAUXC_GENERIC_EXCEPTION("MultiParticle density/VXC size mismatch");
  if( this->load_balancer_->basis_count() != np )
    GAUXC_GENERIC_EXCEPTION("MultiParticle density count must match LoadBalancer basis count");
  if( functional_spec.intra_functionals.size() > np )
    GAUXC_GENERIC_EXCEPTION("Too many MultiParticle intra functional entries");

  for( auto p : terms.active_intra )
    if( p >= functional_spec.intra_functionals.size() )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle active intra index");
  for( auto i : terms.active_inter )
    if( i >= ninter )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle active inter index");
  for( auto p : terms.vxc_targets )
    if( p >= np )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle VXC target index");

  std::vector<bool> build_vxc(np, false);
  for( auto p : terms.vxc_targets ) build_vxc[p] = true;

  for( size_t p = 0; p < np; ++p ) {
    const int64_t nbf = this->load_balancer_->basis(p).nbf();
    const auto& den   = densities[p];
    const bool  has_z = den.Pz != nullptr;
    if( den.m != den.n )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps must be square");
    if( den.m != nbf )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps dimension must match its basis");
    if( den.ldps < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPS");
    if( den.Pz and den.ldpz < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPZ");
    if( build_vxc[p] and (not vxc[p].VXCs or vxc[p].ldvxcs < nbf) )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDVXCS");
    if( build_vxc[p] and has_z and (not vxc[p].VXCz or vxc[p].ldvxcz < nbf) )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDVXCZ");
  }

  for( size_t i = 0; i < ninter; ++i ) {
    const auto& pair = functional_spec.inter_functionals[i];
    if( pair.electron >= np or pair.particle >= np )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle inter functional index");
  }

  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    multiparticle_exc_vxc_local_work_( densities, functional_spec, terms,
                                       vxc, intra_exc, inter_pair_exc, ks_settings,
                                       tasks.begin(), tasks.end() );
  });

  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){
    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    for( size_t p = 0; p < np; ++p ) {
      if( not build_vxc[p] ) continue;
      const int64_t nbf = this->load_balancer_->basis(p).nbf();
      this->reduction_driver_->allreduce_inplace( vxc[p].VXCs, nbf * nbf, ReductionOp::Sum );
      if( densities[p].Pz )
        this->reduction_driver_->allreduce_inplace( vxc[p].VXCz, nbf * nbf, ReductionOp::Sum );
    }

    this->reduction_driver_->allreduce_inplace( intra_exc, np, ReductionOp::Sum );
    if( ninter )
      this->reduction_driver_->allreduce_inplace( inter_pair_exc, ninter, ReductionOp::Sum );
  });

}


/// Generic host EXC/VXC driver for multiple quantum-particle basis sets.
/// Each particle density is evaluated once on the shared grid, then all intra-
/// and inter-particle XC contributions are accumulated before forming VXC.
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  multiparticle_exc_vxc_local_work_( const std::vector<multiparticle_density>& densities,
                                     const MultiParticleFunctionalSpec& functional_spec,
                                     const MultiParticleXCTerms& terms,
                                     std::vector<multiparticle_vxc>& vxc,
                                     value_type* intra_exc,
                                     value_type* inter_pair_exc,
                                     const IntegratorSettingsXC& settings,
                                     task_iterator task_begin, task_iterator task_end ) {

  // Misc KS settings
  IntegratorSettingsKS ks_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsKS*>(&settings) ) {
    ks_settings = *tmp;
  }

  const size_t np     = densities.size();
  const size_t ninter = functional_spec.inter_functionals.size();

  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());
  if( not lwd )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires a host local work driver");

  std::vector<bool> active_intra(np, false);
  std::vector<bool> active_inter(ninter, false);
  std::vector<bool> build_vxc(np, false);
  for( auto p : terms.active_intra ) active_intra[p] = true;
  for( auto i : terms.active_inter ) active_inter[i] = true;
  for( auto p : terms.vxc_targets )  build_vxc[p]    = true;

  // Derive per-particle bases, spin, GGA needs, and zero the integrands.
  std::vector<int64_t> nbf(np);
  std::vector<bool> has_z(np, false);
  std::vector<bool> one_channel(np, false);
  std::vector<bool> need_grad(np, false);
  std::vector<bool> participates(np, false);

  for( size_t p = 0; p < np; ++p ) {
    const auto& basis = this->load_balancer_->basis(p);
    nbf[p] = basis.nbf();
    const auto& den = densities[p];
    has_z[p] = den.Pz != nullptr;

    const auto* funcs = p < functional_spec.intra_functionals.size() ?
      &functional_spec.intra_functionals[p] : nullptr;
    if( active_intra[p] and funcs )
    for( const auto& func : *funcs ) {
      participates[p] = true;
      if( func->is_mgga() )
        GAUXC_GENERIC_EXCEPTION("MultiParticle mGGA intra-XC is not implemented");
      if( func->is_gga() )
        need_grad[p] = true;
    }

    // A high-spin species with no intra functional of its own is carried in one channel (Ps=Pz, avoid duplicate computation)
    const bool has_intra = active_intra[p] and funcs and not funcs->empty();
    one_channel[p] = has_z[p] and not has_intra and
      den.ldps == nbf[p] and den.ldpz == nbf[p] and
      std::memcmp( den.Ps, den.Pz, size_t(nbf[p]) * size_t(nbf[p]) * sizeof(value_type) ) == 0;

    if( build_vxc[p] )
      std::fill(vxc[p].VXCs, vxc[p].VXCs + nbf[p] * vxc[p].ldvxcs, 0.0);
    if( build_vxc[p] and has_z[p] )
      std::fill(vxc[p].VXCz, vxc[p].VXCz + nbf[p] * vxc[p].ldvxcz, 0.0);
    intra_exc[p] = 0.0;
  }

  // The RKS density of a one-channel species: half its total density
  std::vector<std::vector<value_type>> half_density(np);
  for( size_t p = 0; p < np; ++p ) {
    if( not one_channel[p] ) continue;
    const size_t nbf_squared = size_t(nbf[p]) * size_t(nbf[p]);
    half_density[p].resize( nbf_squared );
    for( size_t k = 0; k < nbf_squared; ++k )
      half_density[p][k] = 0.5 * densities[p].Ps[k];
  }

  for( size_t i = 0; i < ninter; ++i ) {
    const auto& pair = functional_spec.inter_functionals[i];
    if( active_inter[i] and not pair.functionals.empty() ) {
      participates[pair.electron] = true;
      participates[pair.particle] = true;
    }
    for( const auto& func : pair.functionals ) {
      if( func->is_gga() or func->is_mgga() )
        GAUXC_GENERIC_EXCEPTION("MultiParticle GGA/mGGA inter-XC is not implemented");
    }
    inter_pair_exc[i] = 0.0;
  }

  // Sort high-cost batches first, matching the existing reference host driver.
  auto& tasks = this->load_balancer_->get_tasks();
  auto task_comparator = [&]( const XCTask& a, const XCTask& b ) {
    auto cost = [&]( const XCTask& task ) {
      size_t nbe = 0;
      for( size_t p = 0; p < np; ++p )
        if( participates[p] ) nbe += task.basis_screening(p).nbe;
      return task.points.size() * nbe;
    };
    return cost(a) > cost(b);
  };
  std::sort( tasks.begin(), tasks.end(), task_comparator );

  // Check that partition weights have been calculated.
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored )
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");

  std::vector<value_type> intra_work(np, 0.0);
  std::vector<value_type> inter_work(ninter, 0.0);

  const size_t ntasks = std::distance( task_begin, task_end );

  #pragma omp parallel
  {

  // Thread-local scratch. Vectors are resized, not reconstructed, per task so
  // capacity is reused across batches.
  struct particle_scratch {
    bool active = false;
    bool is_uks = false;
    bool gga = false;
    int32_t nbe = 0;
    int32_t nshells = 0;
    const int32_t* shell_list = nullptr;
    std::vector<std::array<int32_t, 3>> submat_map;
    std::vector<value_type> basis_eval;
    std::vector<value_type> den_scr;
    std::vector<value_type> gamma;
    std::vector<value_type> eps;
    std::vector<value_type> vrho;
    std::vector<value_type> vgamma;
    std::vector<value_type> zmat;
    std::vector<value_type> nbe_scr;
  };

  std::vector<particle_scratch> scratch(np);
  std::vector<value_type> intra_local(np, 0.0);
  std::vector<value_type> inter_local(ninter, 0.0);
  std::vector<value_type> eps_tmp;
  std::vector<value_type> vrho_tmp;
  std::vector<value_type> vgamma_tmp;
  std::vector<value_type> rho_pair;
  std::vector<value_type> eps_pair;
  std::vector<value_type> vrho_pair;
  std::vector<value_type> eps_acc;

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    // Alias current task
    const auto& task = *( task_begin + iT );
    const int32_t npts = static_cast<int32_t>(task.points.size());
    const auto* points = task.points.data()->data();
    const auto* weights = task.weights.data();

    //----------------------Start Per-Particle Density Evaluation------------------------
    for( size_t p = 0; p < np; ++p ) {
      auto& s = scratch[p];
      s.is_uks = has_z[p] and not one_channel[p];
      s.gga = need_grad[p];
      s.active = false;
      s.nbe = 0;
      s.nshells = 0;
      s.shell_list = nullptr;
      s.submat_map.clear();

      if( not participates[p] )
        continue;

      const auto& screening = task.basis_screening(p);
      s.nbe = screening.nbe;
      s.nshells = static_cast<int32_t>(screening.shell_list.size());
      s.shell_list = screening.shell_list.data();
      s.active = s.nshells != 0;

      const size_t spin_dim = s.is_uks ? 2 : 1;
      const size_t gga_dim = s.is_uks ? 3 : 1;

      // Clear only the point-dependent quantities that kernels accumulate into.
      s.eps.assign(npts, 0.0);
      s.vrho.assign(spin_dim * npts, 0.0);
      if( s.gga ) {
        s.gamma.assign(gga_dim * npts, 0.0);
        s.vgamma.assign(gga_dim * npts, 0.0);
      }

      if( not s.active ) {
        s.den_scr.assign(spin_dim * npts, 0.0);
        continue;
      }

      const auto& basis = this->load_balancer_->basis(p);
      const auto& basis_map = this->load_balancer_->basis_map(p);

      // Get the submatrix map for batch
      std::tie(s.submat_map, std::ignore) =
        gen_compressed_submat_map(basis_map, screening.shell_list, nbf[p], nbf[p]);

      s.nbe_scr.resize(s.nbe * s.nbe);
      s.basis_eval.resize((s.gga ? 4 : 1) * npts * s.nbe);
      s.den_scr.resize(spin_dim * (s.gga ? 4 : 1) * npts);
      s.zmat.resize(spin_dim * npts * s.nbe);

      auto* basis_eval = s.basis_eval.data();
      auto* den_eval = s.den_scr.data();
      auto* zmat = s.zmat.data();
      auto* zmat_z = s.is_uks ? zmat + s.nbe * npts : nullptr;

      value_type* dbasis_x_eval = nullptr;
      value_type* dbasis_y_eval = nullptr;
      value_type* dbasis_z_eval = nullptr;
      value_type* dden_x_eval = nullptr;
      value_type* dden_y_eval = nullptr;
      value_type* dden_z_eval = nullptr;

      if( s.gga ) {
        dbasis_x_eval = basis_eval + npts * s.nbe;
        dbasis_y_eval = dbasis_x_eval + npts * s.nbe;
        dbasis_z_eval = dbasis_y_eval + npts * s.nbe;
        dden_x_eval = den_eval + spin_dim * npts;
        dden_y_eval = dden_x_eval + spin_dim * npts;
        dden_z_eval = dden_y_eval + spin_dim * npts;
      }

      // Evaluate collocation, X = P * B, then U variables for this particle.
      if( s.gga )
        lwd->eval_collocation_gradient( npts, s.nshells, s.nbe, points, basis,
          s.shell_list, basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
      else
        lwd->eval_collocation( npts, s.nshells, s.nbe, points, basis,
          s.shell_list, basis_eval );

      // Evaluate X matrix (fac * P * B) -> store in Z
      // A one-channel species uses the half density Ps/2
      const auto xmat_fac = s.is_uks ? 1.0 : 2.0;
      const auto* Ps = one_channel[p] ? half_density[p].data() : densities[p].Ps;
      lwd->eval_xmat( npts, nbf[p], s.nbe, s.submat_map, xmat_fac,
        Ps, densities[p].ldps, basis_eval, s.nbe, zmat, s.nbe,
        s.nbe_scr.data() );

      if( s.is_uks )
      // X matrix for Pz
        lwd->eval_xmat( npts, nbf[p], s.nbe, s.submat_map, 1.0,
          densities[p].Pz, densities[p].ldpz, basis_eval, s.nbe, zmat_z,
          s.nbe, s.nbe_scr.data() );

      // Evaluate U and V variables
      if( s.gga ) {
        if( s.is_uks )
          lwd->eval_uvvar_gga_uks( npts, s.nbe, basis_eval, dbasis_x_eval,
            dbasis_y_eval, dbasis_z_eval, zmat, s.nbe, zmat_z, s.nbe,
            den_eval, dden_x_eval, dden_y_eval, dden_z_eval, s.gamma.data() );
        else
          lwd->eval_uvvar_gga_rks( npts, s.nbe, basis_eval, dbasis_x_eval,
            dbasis_y_eval, dbasis_z_eval, zmat, s.nbe, den_eval,
            dden_x_eval, dden_y_eval, dden_z_eval, s.gamma.data() );
      } else {
        if( s.is_uks )
          lwd->eval_uvvar_lda_uks( npts, s.nbe, basis_eval, zmat, s.nbe,
            zmat_z, s.nbe, den_eval );
        else
          lwd->eval_uvvar_lda_rks( npts, s.nbe, basis_eval, zmat, s.nbe,
            den_eval );
      }
    }
    //----------------------End Per-Particle Density Evaluation------------------------

    auto rho_total = [&]( size_t p, int32_t i ) -> value_type {
      const auto& s = scratch[p];
      return s.is_uks ? s.den_scr[2*i] + s.den_scr[2*i+1] : s.den_scr[i];
    };

    //----------------------Start Intra-Particle XC Evaluation------------------------
    for( size_t p = 0; p < np; ++p ) {
      if( not active_intra[p] ) continue;
      const auto* funcs = p < functional_spec.intra_functionals.size() ?
        &functional_spec.intra_functionals[p] : nullptr;
      if( !funcs or funcs->empty() ) continue;

      auto& s = scratch[p];
      const size_t spin_dim = s.is_uks ? 2 : 1;
      const size_t gga_dim = s.is_uks ? 3 : 1;

      // Evaluate XC functional
      if( funcs->size() == 1 ) {
        const auto& func = funcs->front();
        if( func->is_gga() ) {
          func->eval_exc_vxc( npts, s.den_scr.data(), s.gamma.data(),
            s.eps.data(), s.vrho.data(), s.vgamma.data() );
        } else {
          func->eval_exc_vxc( npts, s.den_scr.data(), s.eps.data(),
            s.vrho.data() );
        }
      } else {
        eps_tmp.resize(npts);
        vrho_tmp.resize(spin_dim * npts);
        vgamma_tmp.resize(gga_dim * npts);

        for( const auto& func : *funcs ) {
          if( func->is_gga() ) {
            std::fill(vgamma_tmp.begin(), vgamma_tmp.end(), 0.0);
            func->eval_exc_vxc( npts, s.den_scr.data(), s.gamma.data(),
              eps_tmp.data(), vrho_tmp.data(), vgamma_tmp.data() );
          } else {
            func->eval_exc_vxc( npts, s.den_scr.data(), eps_tmp.data(),
              vrho_tmp.data() );
          }

          for( int32_t i = 0; i < npts; ++i )
            s.eps[i] += eps_tmp[i];
          for( size_t i = 0; i < spin_dim * npts; ++i )
            s.vrho[i] += vrho_tmp[i];
          if( func->is_gga() )
            for( size_t i = 0; i < gga_dim * npts; ++i )
              s.vgamma[i] += vgamma_tmp[i];
        }
      }

      // Scalar integrations
      value_type e_local = 0.0;
      for( int32_t i = 0; i < npts; ++i )
        e_local += weights[i] * s.eps[i] * rho_total(p, i);
      intra_local[p] += e_local;
    }
    //----------------------End Intra-Particle XC Evaluation------------------------

    //----------------------Start Inter-Particle XC Evaluation------------------------
    for( size_t iPair = 0; iPair < ninter; ++iPair ) {
      if( not active_inter[iPair] ) continue;
      const auto& pair = functional_spec.inter_functionals[iPair];
      if( pair.functionals.empty() ) continue;
      auto& se = scratch[pair.electron];
      auto& sp = scratch[pair.particle];
      rho_pair.resize(2 * npts);
      eps_pair.resize(npts);
      vrho_pair.resize(2 * npts);
      if( pair.functionals.size() > 1 )
        eps_acc.assign(npts, 0.0);

      for( int32_t i = 0; i < npts; ++i ) {
        // ExchCXX EPC kernels receive total electron density as spin-up and
        // total quantum-particle density as spin-down, matching old NEO GauXC.
        rho_pair[2*i] = rho_total(pair.electron, i);
        rho_pair[2*i+1] = rho_total(pair.particle, i);
      }

      for( const auto& func : pair.functionals ) {
        func->eval_exc_vxc( npts, rho_pair.data(), eps_pair.data(),
          vrho_pair.data() );

        for( int32_t i = 0; i < npts; ++i ) {
          if( pair.functionals.size() > 1 )
            eps_acc[i] += eps_pair[i];

          if( build_vxc[pair.electron] ) {
            if( se.is_uks ) {
              se.vrho[2*i]   += vrho_pair[2*i];
              se.vrho[2*i+1] += vrho_pair[2*i];
            } else {
              se.vrho[i] += vrho_pair[2*i];
            }
          }

          if( build_vxc[pair.particle] ) {
            if( sp.is_uks ) {
              sp.vrho[2*i]   += vrho_pair[2*i+1];
              sp.vrho[2*i+1] += vrho_pair[2*i+1];
            } else {
              sp.vrho[i] += vrho_pair[2*i+1];
            }
          }
        }
      }

      const auto* eps_energy = pair.functionals.size() > 1 ?
        eps_acc.data() : eps_pair.data();
      value_type e_local = 0.0;
      for( int32_t i = 0; i < npts; ++i )
        // Keep the old GauXC NEO EPC convention: integrate ExchCXX eps over
        // both packed densities.
        e_local += weights[i] * eps_energy[i] *
          (rho_pair[2*i] + rho_pair[2*i+1]);
      inter_local[iPair] += e_local;
    }
    //----------------------End Inter-Particle XC Evaluation------------------------

    //----------------------Begin VXC Z-Matrix Assembly----------------------------
    for( size_t p = 0; p < np; ++p ) {
      auto& s = scratch[p];
      if( not build_vxc[p] ) continue;
      if( not s.active ) continue;

      const size_t spin_dim = s.is_uks ? 2 : 1;
      const size_t gga_dim = s.is_uks ? 3 : 1;

      for( int32_t i = 0; i < npts; ++i ) {
      // Factor weights into XC results
        for( size_t is = 0; is < spin_dim; ++is )
          s.vrho[spin_dim*i + is] *= weights[i];
      }
      if( s.gga ) {
        for( int32_t i = 0; i < npts; ++i )
          for( size_t ig = 0; ig < gga_dim; ++ig )
            s.vgamma[gga_dim*i + ig] *= weights[i];
      }

      auto* basis_eval = s.basis_eval.data();
      auto* zmat = s.zmat.data();
      auto* zmat_z = s.is_uks ? zmat + s.nbe * npts : nullptr;
      auto* den_eval = s.den_scr.data();
      auto* gamma = s.gamma.data();
      auto* vrho = s.vrho.data();
      auto* vgamma = s.vgamma.data();

      value_type* dbasis_x_eval = nullptr;
      value_type* dbasis_y_eval = nullptr;
      value_type* dbasis_z_eval = nullptr;
      value_type* dden_x_eval = nullptr;
      value_type* dden_y_eval = nullptr;
      value_type* dden_z_eval = nullptr;

      if( s.gga ) {
        static_cast<void>(gamma);
        dbasis_x_eval = basis_eval + npts * s.nbe;
        dbasis_y_eval = dbasis_x_eval + npts * s.nbe;
        dbasis_z_eval = dbasis_y_eval + npts * s.nbe;
        dden_x_eval = den_eval + spin_dim * npts;
        dden_y_eval = dden_x_eval + spin_dim * npts;
        dden_z_eval = dden_y_eval + spin_dim * npts;

        // Evaluate Z matrix for VXC
        if( s.is_uks )
          lwd->eval_zmat_gga_vxc_uks( npts, s.nbe, vrho, vgamma,
            basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
            dden_x_eval, dden_y_eval, dden_z_eval, zmat, s.nbe, zmat_z,
            s.nbe );
        else
          lwd->eval_zmat_gga_vxc_rks( npts, s.nbe, vrho, vgamma,
            basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
            dden_x_eval, dden_y_eval, dden_z_eval, zmat, s.nbe );
      } else {
        if( s.is_uks )
          lwd->eval_zmat_lda_vxc_uks( npts, s.nbe, vrho, basis_eval, zmat,
            s.nbe, zmat_z, s.nbe );
        else
          lwd->eval_zmat_lda_vxc_rks( npts, s.nbe, vrho, basis_eval, zmat,
            s.nbe );
      }

      // Increment VXC
      lwd->inc_vxc( npts, nbf[p], s.nbe, basis_eval, s.submat_map, zmat,
        s.nbe, vxc[p].VXCs, vxc[p].ldvxcs, s.nbe_scr.data() );
      if( s.is_uks )
        lwd->inc_vxc( npts, nbf[p], s.nbe, basis_eval, s.submat_map, zmat_z,
          s.nbe, vxc[p].VXCz, vxc[p].ldvxcz, s.nbe_scr.data() );
    }
    //----------------------End VXC Z-Matrix Assembly------------------------------
  }

  #pragma omp critical
  {
    for( size_t p = 0; p < np; ++p )
      intra_work[p] += intra_local[p];
    for( size_t i = 0; i < ninter; ++i )
      inter_work[i] += inter_local[i];
  }

  }

  // Symmetrize local VXC blocks before the global reduction.
  for( size_t p = 0; p < np; ++p ) {
    intra_exc[p] = intra_work[p];
    if( not build_vxc[p] ) continue;

    for( int32_t j = 0; j < nbf[p]; ++j )
    for( int32_t i = j+1; i < nbf[p]; ++i )
      vxc[p].VXCs[j + i*vxc[p].ldvxcs] = vxc[p].VXCs[i + j*vxc[p].ldvxcs];

    // A one-channel species returns the same matrix in both channels, half its
    // one accumulator
    if( one_channel[p] ) {
      for( int32_t j = 0; j < nbf[p]; ++j )
      for( int32_t i = 0; i < nbf[p]; ++i ) {
        const auto v = 0.5 * vxc[p].VXCs[i + j*vxc[p].ldvxcs];
        vxc[p].VXCs[i + j*vxc[p].ldvxcs] = v;
        vxc[p].VXCz[i + j*vxc[p].ldvxcz] = v;
      }
      continue;
    }

    if( has_z[p] ) {
      for( int32_t j = 0; j < nbf[p]; ++j )
      for( int32_t i = j+1; i < nbf[p]; ++i )
        vxc[p].VXCz[j + i*vxc[p].ldvxcz] = vxc[p].VXCz[i + j*vxc[p].ldvxcz];
    }
  }
  for( size_t i = 0; i < ninter; ++i ) inter_pair_exc[i] = inter_work[i];

}

} // namespace GauXC::detail
