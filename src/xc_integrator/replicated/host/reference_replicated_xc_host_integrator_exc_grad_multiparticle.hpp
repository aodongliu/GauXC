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
#include "host/blas.hpp"
#include <cstring>
#include <stdexcept>

namespace GauXC::detail {

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_grad_( const std::vector<multiparticle_density>& densities,
                  const MultiParticleFunctionalSpec& functional_spec,
                  const MultiParticleXCTerms& terms,
                  value_type* EXC_GRAD,
                  const IntegratorSettingsXC& settings ) {

  const size_t np     = densities.size();
  const size_t ninter = functional_spec.inter_functionals.size();

  // Check that P is sane
  if( np == 0 )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient requires at least one density");
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

  for( size_t p = 0; p < np; ++p ) {
    const int64_t nbf = this->load_balancer_->basis(p).nbf();
    const auto& den = densities[p];
    if( den.m != den.n )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps must be square");
    if( den.m != nbf )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps dimension must match its basis");
    if( den.ldps < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPS");
    if( den.Pz and den.ldpz < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPZ");
  }

  for( size_t i = 0; i < ninter; ++i ) {
    const auto& pair = functional_spec.inter_functionals[i];
    if( pair.electron >= np or pair.particle >= np )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle inter functional index");
  }

  // Compute Local contributions to the EXC gradient
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    multiparticle_exc_grad_local_work_( densities, functional_spec, terms, EXC_GRAD, settings );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    const int natoms = static_cast<int>(this->load_balancer_->molecule().natoms());
    this->reduction_driver_->allreduce_inplace( EXC_GRAD, 3*natoms, ReductionOp::Sum );
  });

}


template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  multiparticle_exc_grad_local_work_( const std::vector<multiparticle_density>& densities,
                                      const MultiParticleFunctionalSpec& functional_spec,
                                      const MultiParticleXCTerms& terms,
                                      value_type* EXC_GRAD,
                                      const IntegratorSettingsXC& settings ) {

  const size_t np     = densities.size();
  const size_t ninter = functional_spec.inter_functionals.size();

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& mol     = this->load_balancer_->molecule();
  const auto& molmeta = this->load_balancer_->molmeta();
  const int32_t natoms = static_cast<int32_t>(mol.natoms());

  // Weight-derivative settings, defualt on
  IntegratorSettingsEXC_GRAD exc_grad_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsEXC_GRAD*>(&settings) ) {
    exc_grad_settings = *tmp;
  }
  const bool do_weight_derivatives = exc_grad_settings.include_weight_derivatives;

  std::vector<bool> active_intra(np, false);
  std::vector<bool> active_inter(ninter, false);
  for( auto p : terms.active_intra ) active_intra[p] = true;
  for( auto i : terms.active_inter ) active_inter[i] = true;

  // Derive per-particle bases, spin, and required evaluation orders per basis.
  std::vector<int64_t> nbf(np);
  std::vector<bool> has_z(np, false);
  std::vector<bool> one_channel(np, false);
  std::vector<bool> need_grad(np, false);   // GGA intra: needs gamma + basis hessian
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
        GAUXC_GENERIC_EXCEPTION("MultiParticle mGGA intra-XC gradient is not implemented");
      if( func->is_gga() )
        need_grad[p] = true;
    }

    // A high-spin species with no intra functional of its own is carried in one channel (Ps=Pz, avoid duplicate computation)
    const bool has_intra = active_intra[p] and funcs and not funcs->empty();
    one_channel[p] = has_z[p] and not has_intra and
      den.ldps == nbf[p] and den.ldpz == nbf[p] and
      std::memcmp( den.Ps, den.Pz, size_t(nbf[p]) * size_t(nbf[p]) * sizeof(value_type) ) == 0;
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
        GAUXC_GENERIC_EXCEPTION("MultiParticle GGA/mGGA inter-XC gradient is not implemented");
    }
  }

  // Get tasks and sort high-cost batches first.
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


  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored )
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");
  XCWeightAlg& weight_alg = lb_state.weight_alg;

  // Zero the gradient
  for( int32_t i = 0; i < 3*natoms; ++i ) EXC_GRAD[i] = 0.;

  // Loop over tasks
  const size_t ntasks = tasks.size();
  #pragma omp parallel
  {

  // Thread-local scratch, resized (not reconstructed) per task.
  struct particle_scratch {
    bool active = false;
    bool is_uks = false;
    bool gga = false;
    int32_t nbe = 0;
    int32_t nshells = 0;
    const int32_t* shell_list = nullptr;
    std::vector<std::array<int32_t, 3>> submat_map;
    std::vector<value_type> basis_eval;
    std::vector<value_type> den;
    std::vector<value_type> gamma;
    std::vector<value_type> eps;
    std::vector<value_type> vrho;
    std::vector<value_type> vgamma;
    std::vector<value_type> zmat;
    std::vector<value_type> nbe_scr;
  };

  std::vector<particle_scratch> scratch(np);
  std::vector<value_type> eps_tmp;
  std::vector<value_type> vrho_tmp;
  std::vector<value_type> vgamma_tmp;
  std::vector<value_type> rho_pair;
  std::vector<value_type> eps_pair;
  std::vector<value_type> vrho_pair;
  std::vector<value_type> w_times_f;

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    const auto& task = tasks[iT];
    const int32_t npts = static_cast<int32_t>(task.points.size());
    const auto* points = task.points.data()->data();
    const auto* weights = task.weights.data();

    //----------------------Start Per-Particle Density Evaluation------------------------
    // Mirrors eval_exc_vxc_ multiparticle, except that the gradient additionally
    // needs the basis hessian (GGA intra) and the X matrix over the basis
    // derivative blocks.
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

      s.eps.assign(npts, 0.0);
      s.vrho.assign(spin_dim * npts, 0.0);
      if( s.gga ) {
        s.gamma.assign(gga_dim * npts, 0.0);
        s.vgamma.assign(gga_dim * npts, 0.0);
      }

      if( not s.active ) {
        s.den.assign(spin_dim * npts, 0.0);
        continue;
      }

      const auto& basis = this->load_balancer_->basis(p);
      const auto& basis_map = this->load_balancer_->basis_map(p);

      // Get the submatrix map for batch
      std::tie(s.submat_map, std::ignore) =
        gen_compressed_submat_map(basis_map, screening.shell_list, nbf[p], nbf[p]);

      // Basis storage: [ B, B_x, B_y, B_z (, B_xx, B_xy, B_xz, B_yy, B_yz, B_zz) ]
      // Contiguity of B and its first derivatives is required by the stacked
      // eval_xmat call below.
      const size_t basis_slots = s.gga ? 10 : 4;
      // Z storage: [ xN (, xN_x, xN_y, xN_z) (, xZ (, xZ_x, xZ_y, xZ_z)) ]
      const size_t xmat_len = s.gga ? 4 : 1;

      s.nbe_scr.resize(s.nbe * s.nbe);
      s.basis_eval.resize(basis_slots * npts * s.nbe);
      s.den.resize(spin_dim * (s.gga ? 4 : 1) * npts);
      s.zmat.resize(spin_dim * xmat_len * npts * s.nbe);

      auto* basis_eval = s.basis_eval.data();
      auto* den_eval = s.den.data();
      auto* zmat = s.zmat.data();
      auto* zmat_z = s.is_uks ? zmat + xmat_len * npts * s.nbe : nullptr;

      auto* dbasis_x_eval = basis_eval    + npts * s.nbe;
      auto* dbasis_y_eval = dbasis_x_eval + npts * s.nbe;
      auto* dbasis_z_eval = dbasis_y_eval + npts * s.nbe;

      value_type* d2basis_xx_eval = nullptr;
      value_type* d2basis_xy_eval = nullptr;
      value_type* d2basis_xz_eval = nullptr;
      value_type* d2basis_yy_eval = nullptr;
      value_type* d2basis_yz_eval = nullptr;
      value_type* d2basis_zz_eval = nullptr;
      value_type* dden_x_eval = nullptr;
      value_type* dden_y_eval = nullptr;
      value_type* dden_z_eval = nullptr;

      if( s.gga ) {
        d2basis_xx_eval = dbasis_z_eval   + npts * s.nbe;
        d2basis_xy_eval = d2basis_xx_eval + npts * s.nbe;
        d2basis_xz_eval = d2basis_xy_eval + npts * s.nbe;
        d2basis_yy_eval = d2basis_xz_eval + npts * s.nbe;
        d2basis_yz_eval = d2basis_yy_eval + npts * s.nbe;
        d2basis_zz_eval = d2basis_yz_eval + npts * s.nbe;
        dden_x_eval = den_eval + spin_dim * npts;
        dden_y_eval = dden_x_eval + spin_dim * npts;
        dden_z_eval = dden_y_eval + spin_dim * npts;
      }

      // The gradient always needs first basis derivatives; GGA intra terms
      // additionally need the basis hessian.
      if( s.gga )
        lwd->eval_collocation_hessian( npts, s.nshells, s.nbe, points, basis,
          s.shell_list, basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
          d2basis_xx_eval, d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval,
          d2basis_yz_eval, d2basis_zz_eval );
      else
        lwd->eval_collocation_gradient( npts, s.nshells, s.nbe, points, basis,
          s.shell_list, basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );

      // X = fac * P * [B (, B_x, B_y, B_z)] (stacked over derivative blocks).
      // A one-channel species uses the half density Ps/2
      const auto xmat_fac = s.is_uks ? 1.0 : 2.0;
      const auto* Ps = one_channel[p] ? half_density[p].data() : densities[p].Ps;
      lwd->eval_xmat( xmat_len * npts, nbf[p], s.nbe, s.submat_map, xmat_fac,
        Ps, densities[p].ldps, basis_eval, s.nbe, zmat, s.nbe,
        s.nbe_scr.data() );

      if( s.is_uks )
        lwd->eval_xmat( xmat_len * npts, nbf[p], s.nbe, s.submat_map, 1.0,
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
      return s.is_uks ? s.den[2*i] + s.den[2*i+1] : s.den[i];
    };

    // Per-point total energy density accumulator for the grid-weight term.
    if( do_weight_derivatives )
      w_times_f.assign(npts, 0.0);

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
          func->eval_exc_vxc( npts, s.den.data(), s.gamma.data(),
            s.eps.data(), s.vrho.data(), s.vgamma.data() );
        } else {
          func->eval_exc_vxc( npts, s.den.data(), s.eps.data(),
            s.vrho.data() );
        }
      } else {
        eps_tmp.resize(npts);
        vrho_tmp.resize(spin_dim * npts);
        vgamma_tmp.resize(gga_dim * npts);

        for( const auto& func : *funcs ) {
          if( func->is_gga() ) {
            std::fill(vgamma_tmp.begin(), vgamma_tmp.end(), 0.0);
            func->eval_exc_vxc( npts, s.den.data(), s.gamma.data(),
              eps_tmp.data(), vrho_tmp.data(), vgamma_tmp.data() );
          } else {
            func->eval_exc_vxc( npts, s.den.data(), eps_tmp.data(),
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

      if( do_weight_derivatives )
        for( int32_t i = 0; i < npts; ++i )
          w_times_f[i] += s.eps[i] * rho_total(p, i);
    }
    //----------------------End Intra-Particle XC Evaluation------------------------

    //----------------------Start Inter-Particle XC Evaluation------------------------
    // EPC vrho contributions are scattered into each particle's vrho with the
    // same spin-channel conventions as the EXC/VXC driver; they then enter the
    // gradient through the standard LDA contraction with that particle's basis
    // derivatives below.
    for( size_t iPair = 0; iPair < ninter; ++iPair ) {
      if( not active_inter[iPair] ) continue;
      const auto& pair = functional_spec.inter_functionals[iPair];
      if( pair.functionals.empty() ) continue;
      auto& se = scratch[pair.electron];
      auto& sp = scratch[pair.particle];
      rho_pair.resize(2 * npts);
      eps_pair.resize(npts);
      vrho_pair.resize(2 * npts);

      for( int32_t i = 0; i < npts; ++i ) {
        // ExchCXX EPC kernels receive total electron density as spin-up and
        // total quantum-particle density as spin-down.
        rho_pair[2*i] = rho_total(pair.electron, i);
        rho_pair[2*i+1] = rho_total(pair.particle, i);
      }

      for( const auto& func : pair.functionals ) {
        func->eval_exc_vxc( npts, rho_pair.data(), eps_pair.data(),
          vrho_pair.data() );

        for( int32_t i = 0; i < npts; ++i ) {
          if( se.is_uks ) {
            se.vrho[2*i]   += vrho_pair[2*i];
            se.vrho[2*i+1] += vrho_pair[2*i];
          } else {
            se.vrho[i] += vrho_pair[2*i];
          }

          if( sp.is_uks ) {
            sp.vrho[2*i]   += vrho_pair[2*i+1];
            sp.vrho[2*i+1] += vrho_pair[2*i+1];
          } else {
            sp.vrho[i] += vrho_pair[2*i+1];
          }
        }

        if( do_weight_derivatives )
          for( int32_t i = 0; i < npts; ++i )
            // Match the EXC convention: integrate ExchCXX eps over both
            // packed densities.
            w_times_f[i] += eps_pair[i] * (rho_pair[2*i] + rho_pair[2*i+1]);
      }
    }
    //----------------------End Inter-Particle XC Evaluation------------------------

    //----------------------Grid-Weight Derivative Contribution--------------------
    bool any_active = false;
    for( size_t p = 0; p < np; ++p ) any_active |= scratch[p].active;

    // grid weight contribution to exc grad
    if( do_weight_derivatives and any_active ) {
      for( int32_t i = 0; i < npts; ++i ) w_times_f[i] *= weights[i];
      lwd->eval_weight_1st_deriv_contracted( weight_alg, mol, molmeta,
        task, w_times_f.data(), EXC_GRAD );
    }
    //------------------------------------------------------------------------------

    //----------------------Start Per-Particle Gradient Accumulation---------------
    // Identical math to the single-particle exc_grad accumulation (LDA + GGA
    // terms), applied per basis with that particle's vrho/vgamma. Shells on
    // the task's parent atom are accumulated by translational invariance when
    // weight derivatives are enabled.
    for( size_t p = 0; p < np; ++p ) {
      auto& s = scratch[p];
      if( not s.active ) continue;

      const auto& basis = this->load_balancer_->basis(p);
      const auto& basis_map = this->load_balancer_->basis_map(p);

      const size_t spin_dim_scal = s.is_uks ? 2 : 1;
      const size_t gga_dim_scal = s.is_uks ? 3 : 1;
      const bool is_rks = not s.is_uks;
      const size_t xmat_len = s.gga ? 4 : 1;

      const auto* basis_eval = s.basis_eval.data();
      const auto* den_eval = s.den.data();
      const auto* zmat = s.zmat.data();
      const auto* vrho = s.vrho.data();
      const auto* vgamma = s.vgamma.data();

      const auto* dbasis_x_eval = basis_eval    + npts * s.nbe;
      const auto* dbasis_y_eval = dbasis_x_eval + npts * s.nbe;
      const auto* dbasis_z_eval = dbasis_y_eval + npts * s.nbe;

      const value_type* d2basis_xx_eval = nullptr;
      const value_type* d2basis_xy_eval = nullptr;
      const value_type* d2basis_xz_eval = nullptr;
      const value_type* d2basis_yy_eval = nullptr;
      const value_type* d2basis_yz_eval = nullptr;
      const value_type* d2basis_zz_eval = nullptr;
      const value_type* dden_x_eval = nullptr;
      const value_type* dden_y_eval = nullptr;
      const value_type* dden_z_eval = nullptr;

      const value_type* xNmat   = zmat;
      const value_type* xNmat_x = nullptr;
      const value_type* xNmat_y = nullptr;
      const value_type* xNmat_z = nullptr;
      const value_type* xZmat   = s.is_uks ? zmat + xmat_len * npts * s.nbe : nullptr;
      const value_type* xZmat_x = nullptr;
      const value_type* xZmat_y = nullptr;
      const value_type* xZmat_z = nullptr;

      if( s.gga ) {
        d2basis_xx_eval = dbasis_z_eval   + npts * s.nbe;
        d2basis_xy_eval = d2basis_xx_eval + npts * s.nbe;
        d2basis_xz_eval = d2basis_xy_eval + npts * s.nbe;
        d2basis_yy_eval = d2basis_xz_eval + npts * s.nbe;
        d2basis_yz_eval = d2basis_yy_eval + npts * s.nbe;
        d2basis_zz_eval = d2basis_yz_eval + npts * s.nbe;
        dden_x_eval = den_eval + spin_dim_scal * npts;
        dden_y_eval = dden_x_eval + spin_dim_scal * npts;
        dden_z_eval = dden_y_eval + spin_dim_scal * npts;
        xNmat_x = xNmat   + npts * s.nbe;
        xNmat_y = xNmat_x + npts * s.nbe;
        xNmat_z = xNmat_y + npts * s.nbe;
        if( s.is_uks ) {
          xZmat_x = xZmat   + npts * s.nbe;
          xZmat_y = xZmat_x + npts * s.nbe;
          xZmat_z = xZmat_y + npts * s.nbe;
        }
      }

      size_t bf_off = 0;
      // Increment EXC Gradient
      for( int32_t ish = 0; ish < s.nshells; ++ish ) {
        const int sh_idx = s.shell_list[ish];
        const int sh_sz  = basis[sh_idx].size();
        const int iAt    = basis_map.shell_to_center( sh_idx );
        if( iAt == task.iParent and do_weight_derivatives ) {
          bf_off += sh_sz; // Increment basis offset
          continue;
        }

        double g_acc_x(0), g_acc_y(0), g_acc_z(0);
        for( int ibf = 0, mu = static_cast<int>(bf_off); ibf < sh_sz; ++ibf, ++mu )
        for( int ipt = 0; ipt < npts; ++ipt ) {

          const int32_t mu_i = mu + ipt*s.nbe;

          // LDA contributions (intra LDA pieces + EPC)
          // vrhop is actually vrhon for RKS
          const double vrhop_ipt = weights[ipt] * vrho[spin_dim_scal * ipt];
          const double vrhom_ipt = s.is_uks ? weights[ipt] * vrho[spin_dim_scal * ipt + 1] : 0.0;

          const double xN = xNmat[mu_i]; // X = N * B
          const double xZ = s.is_uks ? xZmat[mu_i] : 0.0;

          const double dbx = dbasis_x_eval[mu_i]; // B_x
          const double dby = dbasis_y_eval[mu_i]; // B_y
          const double dbz = dbasis_z_eval[mu_i]; // B_z

          if(is_rks) {
            g_acc_x += vrhop_ipt * xN * dbx;
            g_acc_y += vrhop_ipt * xN * dby;
            g_acc_z += vrhop_ipt * xN * dbz;
          } else {
            const auto vrhon_ipt = vrhop_ipt + vrhom_ipt;
            const auto vrhoz_ipt = vrhop_ipt - vrhom_ipt;
            g_acc_x += 0.5 * vrhon_ipt * xN * dbx;
            g_acc_y += 0.5 * vrhon_ipt * xN * dby;
            g_acc_z += 0.5 * vrhon_ipt * xN * dbz;

            g_acc_x += 0.5 * vrhoz_ipt * xZ * dbx;
            g_acc_y += 0.5 * vrhoz_ipt * xZ * dby;
            g_acc_z += 0.5 * vrhoz_ipt * xZ * dbz;
          }

          if( s.gga ) {
            // GGA contributions (intra only; inter-XC is LDA-type)
            const double vgammapp_ipt = weights[ipt] * vgamma[gga_dim_scal * ipt + 0];
            const double vgammapm_ipt = s.is_uks ? weights[ipt] * vgamma[gga_dim_scal * ipt + 1] : 0.0;
            const double vgammamm_ipt = s.is_uks ? weights[ipt] * vgamma[gga_dim_scal * ipt + 2] : 0.0;

            const double ddenn_x = dden_x_eval[spin_dim_scal * ipt];
            const double ddenn_y = dden_y_eval[spin_dim_scal * ipt];
            const double ddenn_z = dden_z_eval[spin_dim_scal * ipt];
            const double ddenz_x = s.is_uks ? dden_x_eval[spin_dim_scal * ipt + 1] : 0.0;
            const double ddenz_y = s.is_uks ? dden_y_eval[spin_dim_scal * ipt + 1] : 0.0;
            const double ddenz_z = s.is_uks ? dden_z_eval[spin_dim_scal * ipt + 1] : 0.0;

            const double xNx = xNmat_x[mu_i]; // XN_x = N * B_x
            const double xNy = xNmat_y[mu_i]; // XN_y = N * B_y
            const double xNz = xNmat_z[mu_i]; // XN_z = N * B_z

            const double xZx = s.is_uks ? xZmat_x[mu_i] : 0.0;
            const double xZy = s.is_uks ? xZmat_y[mu_i] : 0.0;
            const double xZz = s.is_uks ? xZmat_z[mu_i] : 0.0;

            const double d2bxx = d2basis_xx_eval[mu_i]; // B^2_xx
            const double d2bxy = d2basis_xy_eval[mu_i]; // B^2_xy
            const double d2bxz = d2basis_xz_eval[mu_i]; // B^2_xz
            const double d2byy = d2basis_yy_eval[mu_i]; // B^2_yy
            const double d2byz = d2basis_yz_eval[mu_i]; // B^2_yz
            const double d2bzz = d2basis_zz_eval[mu_i]; // B^2_zz

            if(is_rks) {
              // sum_j B^2_{ij} * d_j n
              const auto d2_term_x = d2bxx * ddenn_x + d2bxy * ddenn_y + d2bxz * ddenn_z;
              const auto d2_term_y = d2bxy * ddenn_x + d2byy * ddenn_y + d2byz * ddenn_z;
              const auto d2_term_z = d2bxz * ddenn_x + d2byz * ddenn_y + d2bzz * ddenn_z;

              // sum_j (d_j n) * xN^j
              const double d11_xmat_term = ddenn_x * xNx + ddenn_y * xNy + ddenn_z * xNz;

              g_acc_x += 2 * vgammapp_ipt * ( xN * d2_term_x + dbx * d11_xmat_term );
              g_acc_y += 2 * vgammapp_ipt * ( xN * d2_term_y + dby * d11_xmat_term );
              g_acc_z += 2 * vgammapp_ipt * ( xN * d2_term_z + dbz * d11_xmat_term );
            } else {
              // sum_j B^2_{ij} * d_j n
              const auto d2n_term_x = d2bxx * ddenn_x + d2bxy * ddenn_y + d2bxz * ddenn_z;
              const auto d2n_term_y = d2bxy * ddenn_x + d2byy * ddenn_y + d2byz * ddenn_z;
              const auto d2n_term_z = d2bxz * ddenn_x + d2byz * ddenn_y + d2bzz * ddenn_z;

              // sum_j B^2_{ij} * d_j m_z
              const auto d2z_term_x = d2bxx * ddenz_x + d2bxy * ddenz_y + d2bxz * ddenz_z;
              const auto d2z_term_y = d2bxy * ddenz_x + d2byy * ddenz_y + d2byz * ddenz_z;
              const auto d2z_term_z = d2bxz * ddenz_x + d2byz * ddenz_y + d2bzz * ddenz_z;

              // sum_j (d_j n) * xN^j
              const double d11nn_xmat_term = ddenn_x * xNx + ddenn_y * xNy + ddenn_z * xNz;
              // sum_j (d_j n) * xZ^j
              const double d11nz_xmat_term = ddenn_x * xZx + ddenn_y * xZy + ddenn_z * xZz;
              // sum_j (d_j m_z) * xN^j
              const double d11zn_xmat_term = ddenz_x * xNx + ddenz_y * xNy + ddenz_z * xNz;
              // sum_j (d_j m_z) * xZ^j
              const double d11zz_xmat_term = ddenz_x * xZx + ddenz_y * xZy + ddenz_z * xZz;

              g_acc_x += 0.5 * (vgammapp_ipt + vgammapm_ipt + vgammamm_ipt) * (d2n_term_x * xN + d11nn_xmat_term * dbx);
              g_acc_x += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2z_term_x * xN + d11zn_xmat_term * dbx);
              g_acc_x += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2n_term_x * xZ + d11nz_xmat_term * dbx);
              g_acc_x += 0.5 * (vgammapp_ipt - vgammapm_ipt + vgammamm_ipt) * (d2z_term_x * xZ + d11zz_xmat_term * dbx);

              g_acc_y += 0.5 * (vgammapp_ipt + vgammapm_ipt + vgammamm_ipt) * (d2n_term_y * xN + d11nn_xmat_term * dby);
              g_acc_y += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2z_term_y * xN + d11zn_xmat_term * dby);
              g_acc_y += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2n_term_y * xZ + d11nz_xmat_term * dby);
              g_acc_y += 0.5 * (vgammapp_ipt - vgammapm_ipt + vgammamm_ipt) * (d2z_term_y * xZ + d11zz_xmat_term * dby);

              g_acc_z += 0.5 * (vgammapp_ipt + vgammapm_ipt + vgammamm_ipt) * (d2n_term_z * xN + d11nn_xmat_term * dbz);
              g_acc_z += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2z_term_z * xN + d11zn_xmat_term * dbz);
              g_acc_z += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2n_term_z * xZ + d11nz_xmat_term * dbz);
              g_acc_z += 0.5 * (vgammapp_ipt - vgammapm_ipt + vgammamm_ipt) * (d2z_term_z * xZ + d11zz_xmat_term * dbz);
            }
          }

        } // loop over bfns + grid points

        #pragma omp atomic
        EXC_GRAD[3*iAt + 0] += -2 * g_acc_x;
        #pragma omp atomic
        EXC_GRAD[3*iAt + 1] += -2 * g_acc_y;
        #pragma omp atomic
        EXC_GRAD[3*iAt + 2] += -2 * g_acc_z;

        if( do_weight_derivatives ) {
          #pragma omp atomic
          EXC_GRAD[3*task.iParent + 0] -= -2 * g_acc_x;
          #pragma omp atomic
          EXC_GRAD[3*task.iParent + 1] -= -2 * g_acc_y;
          #pragma omp atomic
          EXC_GRAD[3*task.iParent + 2] -= -2 * g_acc_z;
        }

        bf_off += sh_sz; // Increment basis offset
      } // End loop over shells
    }
    //----------------------End Per-Particle Gradient Accumulation-----------------

  } // End loop over tasks

  } // OpenMP region


}

} // namespace GauXC::detail
