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

#include <gauxc/xc_integrator/replicated/replicated_xc_integrator_impl.hpp>
#include <gauxc/exceptions.hpp>

// Implementations of ReplicatedXCIntegrator public API

namespace GauXC  {
namespace detail {


template <typename MatrixType>
ReplicatedXCIntegrator<MatrixType>::
  ReplicatedXCIntegrator( std::unique_ptr<pimpl_type>&& pimpl ) : 
    pimpl_(std::move(pimpl)){ }

template <typename MatrixType>
ReplicatedXCIntegrator<MatrixType>::ReplicatedXCIntegrator(): 
  ReplicatedXCIntegrator(nullptr){ }

template <typename MatrixType>
ReplicatedXCIntegrator<MatrixType>::~ReplicatedXCIntegrator() noexcept = default; 
template <typename MatrixType>
ReplicatedXCIntegrator<MatrixType>::
  ReplicatedXCIntegrator(ReplicatedXCIntegrator&&) noexcept = default; 

template <typename MatrixType>
const util::Timer& ReplicatedXCIntegrator<MatrixType>::get_timings_() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->get_timings();
}

template <typename MatrixType>
const LoadBalancer& ReplicatedXCIntegrator<MatrixType>::get_load_balancer_() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->get_load_balancer();
}
template <typename MatrixType>
LoadBalancer& ReplicatedXCIntegrator<MatrixType>::get_load_balancer_() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->get_load_balancer();
}


template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::value_type 
  ReplicatedXCIntegrator<MatrixType>::integrate_den_( const MatrixType& P ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  value_type N_EL;
  
  pimpl_->integrate_den( P.rows(), P.cols(), P.data(), P.rows(), &N_EL );

  return N_EL;
}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::value_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  value_type EXC;
  
  pimpl_->eval_exc( P.rows(), P.cols(), P.data(), P.rows(), &EXC, ks_settings );

  return EXC;
}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::value_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  value_type EXC;
  
  const size_t n = Ps.rows();
  pimpl_->eval_exc( n, n, Ps.data(), n, Pz.data(), n, &EXC, ks_settings );

  return EXC;
}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::value_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  value_type EXC;
  
  const size_t n = Ps.rows();
  pimpl_->eval_exc( n, n, Ps.data(), n, Pz.data(), n, Py.data(), n, Px.data(), n, &EXC, ks_settings );

  return EXC;
}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::value_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px, 
     const MatrixType& Ps_SS, const MatrixType& Pz_SS, const MatrixType& Py_SS, const MatrixType& Px_SS, 
     const MatrixType& Ps_SS_imag, const MatrixType& Pz_SS_imag, const MatrixType& Py_SS_imag, const MatrixType& Px_SS_imag, const IntegratorSettingsXC& ks_settings ) {
      //dks
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  value_type EXC;
  
  const size_t n = Ps.rows();
  pimpl_->eval_exc( n, n, Ps.data(), n, Pz.data(), n, Py.data(), n, Px.data(), n,
                    Ps_SS.data(), n, Pz_SS.data(), n, Py_SS.data(), n, Px_SS.data(), n,
                    Ps_SS_imag.data(), Pz_SS_imag.data(), Py_SS_imag.data(), Px_SS_imag.data(), &EXC, ks_settings );

  return EXC;
}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exc_vxc_type_rks 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_vxc_( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  matrix_type VXC( P.rows(), P.cols() );
  value_type  EXC;

  pimpl_->eval_exc_vxc( P.rows(), P.cols(), P.data(), P.rows(),
                        VXC.data(), VXC.rows(), &EXC, ks_settings );

  return std::make_tuple( EXC, VXC );

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exc_vxc_type_uks
  ReplicatedXCIntegrator<MatrixType>::eval_exc_vxc_( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  matrix_type VXCs( Ps.rows(), Ps.cols() );
  matrix_type VXCz( Pz.rows(), Pz.cols() );
  value_type  EXC;

  pimpl_->eval_exc_vxc( Ps.rows(), Ps.cols(), Ps.data(), Ps.rows(),
                        Pz.data(), Pz.rows(),
                        VXCs.data(), VXCs.rows(),
                        VXCz.data(), VXCz.rows(), &EXC, ks_settings );

  return std::make_tuple( EXC, VXCs, VXCz );

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exc_vxc_type_gks
  ReplicatedXCIntegrator<MatrixType>::eval_exc_vxc_( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px,
                                                     const IntegratorSettingsXC& ks_settings) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  matrix_type VXCs( Ps.rows(), Ps.cols() );
  matrix_type VXCz( Pz.rows(), Pz.cols() );
  matrix_type VXCy( Py.rows(), Py.cols() );
  matrix_type VXCx( Px.rows(), Px.cols() );
  value_type  EXC;

  pimpl_->eval_exc_vxc( Ps.rows(), Ps.cols(), Ps.data(), Ps.rows(),
                        Pz.data(), Pz.rows(),
                        Py.data(), Py.rows(),
                        Px.data(), Px.rows(),
                        VXCs.data(), VXCs.rows(),
                        VXCz.data(), VXCz.rows(),
                        VXCy.data(), VXCy.rows(),
                        VXCx.data(), VXCx.rows(), &EXC, ks_settings );

  return std::make_tuple( EXC, VXCs, VXCz, VXCy, VXCx);

}

// dks
template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exc_vxc_type_dks
  ReplicatedXCIntegrator<MatrixType>::eval_exc_vxc_( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px,
                                                     const MatrixType& Ps_SS, const MatrixType& Pz_SS, const MatrixType& Py_SS, const MatrixType& Px_SS,
                                                     const MatrixType& Ps_SS_imag, const MatrixType& Pz_SS_imag, const MatrixType& Py_SS_imag, const MatrixType& Px_SS_imag, 
                                                     const IntegratorSettingsXC& ks_settings) {
  
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  matrix_type VXCs( Ps.rows(), Ps.cols() );
  matrix_type VXCz( Pz.rows(), Pz.cols() );
  matrix_type VXCy( Py.rows(), Py.cols() );
  matrix_type VXCx( Px.rows(), Px.cols() );
  matrix_type VXCs_SS( Ps_SS.rows(), Ps_SS.cols() );
  matrix_type VXCz_SS( Pz_SS.rows(), Pz_SS.cols() );
  matrix_type VXCy_SS( Py_SS.rows(), Py_SS.cols() );
  matrix_type VXCx_SS( Px_SS.rows(), Px_SS.cols() );
  matrix_type VXCs_SS_im( Ps_SS_imag.rows(), Ps_SS_imag.cols() );
  matrix_type VXCz_SS_im( Pz_SS_imag.rows(), Pz_SS_imag.cols() );
  matrix_type VXCy_SS_im( Py_SS_imag.rows(), Py_SS_imag.cols() );
  matrix_type VXCx_SS_im( Px_SS_imag.rows(), Px_SS_imag.cols() );
  value_type  EXC;
  pimpl_->eval_exc_vxc( Ps.rows(), Ps.cols(), Ps.data(), Ps.rows(),
                        Pz.data(), Pz.rows(),
                        Py.data(), Py.rows(),
                        Px.data(), Px.rows(),
                        Ps_SS.data(), Ps_SS.rows(),
                        Pz_SS.data(), Pz_SS.rows(),
                        Py_SS.data(), Py_SS.rows(),
                        Px_SS.data(), Px_SS.rows(),
                        Ps_SS_imag.data(), 
                        Pz_SS_imag.data(), 
                        Py_SS_imag.data(), 
                        Px_SS_imag.data(), 
                        VXCs.data(), VXCs.rows(),
                        VXCz.data(), VXCz.rows(),
                        VXCy.data(), VXCy.rows(),
                        VXCx.data(), VXCx.rows(),
                        VXCs_SS.data(), VXCs_SS.rows(),
                        VXCz_SS.data(), VXCz_SS.rows(),
                        VXCy_SS.data(), VXCy_SS.rows(),
                        VXCx_SS.data(), VXCx_SS.rows(),
                        VXCs_SS_im.data(), VXCs_SS_im.rows(),
                        VXCz_SS_im.data(), VXCz_SS_im.rows(),
                        VXCy_SS_im.data(), VXCy_SS_im.rows(),
                        VXCx_SS_im.data(), VXCx_SS_im.rows(), &EXC, ks_settings );
                        
  return std::make_tuple( EXC, VXCs, VXCz, VXCy, VXCx, VXCs_SS, VXCz_SS, VXCy_SS, VXCx_SS, VXCs_SS_im, VXCz_SS_im, VXCy_SS_im, VXCx_SS_im);

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::multiparticle_exc_vxc_type
  ReplicatedXCIntegrator<MatrixType>::eval_exc_vxc_(
    const std::vector<multiparticle_density>& densities,
    const MultiParticleFunctionalSpec& functional_spec,
    const MultiParticleXCTerms& terms,
    const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();

  using raw_density = typename pimpl_type::multiparticle_density;
  using raw_vxc_type = typename pimpl_type::multiparticle_vxc;

  const size_t np = densities.size();
  multiparticle_exc_vxc_type ret;
  ret.intra_exc.assign(np, value_type{0});
  ret.inter_exc = value_type{0};
  ret.inter_pair_exc.assign(functional_spec.inter_functionals.size(), value_type{0});
  ret.VXCs.reserve(np);
  ret.VXCz.reserve(np);

  // Determine indices of particles to build VXC for
  std::vector<bool> build_vxc(np, false);
  for( auto p : terms.vxc_targets ) {
    if( p >= np )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle VXC target index");
    build_vxc[p] = true;
  }

  std::vector<raw_density> raw_densities;
  std::vector<raw_vxc_type> raw_vxcs;
  raw_densities.reserve(np);
  raw_vxcs.reserve(np);

  for( size_t i = 0; i < np; ++i ) {
    const auto& density = densities[i];
    if( density.Ps == nullptr )
      GAUXC_GENERIC_EXCEPTION("MultiParticle density is missing Ps");
    if( density.Ps->rows() != density.Ps->cols() )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps must be square");
    if( density.Pz and
        ( density.Pz->rows() != density.Ps->rows() or
          density.Pz->cols() != density.Ps->cols() ) )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Pz must match Ps dimensions");

    if( build_vxc[i] )
      ret.VXCs.emplace_back( density.Ps->rows(), density.Ps->cols() );
    else
      ret.VXCs.emplace_back();
    if( build_vxc[i] and density.Pz )
      ret.VXCz.emplace_back( density.Pz->rows(), density.Pz->cols() );
    else
      ret.VXCz.emplace_back();

    raw_densities.push_back( raw_density{
      density.Ps->rows(), density.Ps->cols(), density.Ps->data(), density.Ps->rows(),
      density.Pz ? density.Pz->data() : nullptr,
      density.Pz ? density.Pz->rows() : int64_t{0}
    } );

    raw_vxcs.push_back( raw_vxc_type{
      build_vxc[i] ? ret.VXCs.back().data() : nullptr,
      build_vxc[i] ? ret.VXCs.back().rows() : int64_t{0},
      build_vxc[i] and density.Pz ? ret.VXCz.back().data() : nullptr,
      build_vxc[i] and density.Pz ? ret.VXCz.back().rows() : int64_t{0}
    } );
  }

  pimpl_->eval_exc_vxc( raw_densities, functional_spec, terms, raw_vxcs,
                        ret.intra_exc.data(), ret.inter_pair_exc.data(),
                        ks_settings );

  for( const auto& energy : ret.inter_pair_exc ) ret.inter_exc += energy;

  return ret;

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exc_grad_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_grad_(
    const std::vector<multiparticle_density>& densities,
    const MultiParticleFunctionalSpec& functional_spec,
    const MultiParticleXCTerms& terms,
    const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();

  using raw_density = typename pimpl_type::multiparticle_density;

  std::vector<raw_density> raw_densities;
  raw_densities.reserve(densities.size());

  for( const auto& density : densities ) {
    if( density.Ps == nullptr )
      GAUXC_GENERIC_EXCEPTION("MultiParticle density is missing Ps");
    if( density.Ps->rows() != density.Ps->cols() )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps must be square");
    if( density.Pz and
        ( density.Pz->rows() != density.Ps->rows() or
          density.Pz->cols() != density.Ps->cols() ) )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Pz must match Ps dimensions");

    raw_densities.push_back( raw_density{
      density.Ps->rows(), density.Ps->cols(), density.Ps->data(), density.Ps->rows(),
      density.Pz ? density.Pz->data() : nullptr,
      density.Pz ? density.Pz->rows() : int64_t{0}
    } );
  }

  std::vector<value_type> EXC_GRAD( 3*pimpl_->load_balancer().molecule().natoms() );
  pimpl_->eval_exc_grad( raw_densities, functional_spec, terms,
                         EXC_GRAD.data(), ks_settings );

  return EXC_GRAD;

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exc_grad_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_grad_( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();

  std::vector<value_type> EXC_GRAD( 3*pimpl_->load_balancer().molecule().natoms() );
  pimpl_->eval_exc_grad( P.rows(), P.cols(), P.data(), P.rows(),
                         EXC_GRAD.data(), ks_settings );

  return EXC_GRAD;

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exc_grad_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exc_grad_( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();

  std::vector<value_type> EXC_GRAD( 3*pimpl_->load_balancer().molecule().natoms() );
  pimpl_->eval_exc_grad( Ps.rows(), Ps.cols(), Ps.data(), Ps.rows(), Pz.data(), Pz.rows(),
                         EXC_GRAD.data(), ks_settings );

  return EXC_GRAD;

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::exx_type 
  ReplicatedXCIntegrator<MatrixType>::eval_exx_( const MatrixType& P, const IntegratorSettingsEXX& settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  
  matrix_type K( P.rows(), P.cols() );

  pimpl_->eval_exx( P.rows(), P.cols(), P.data(), P.rows(),
                    K.data(), K.rows(), settings );

  return K;

}
template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::fxc_contraction_type_rks
  ReplicatedXCIntegrator<MatrixType>::eval_fxc_contraction_( const MatrixType& P, 
    const MatrixType& tP, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  matrix_type FXC( P.rows(), P.cols() );

  pimpl_->eval_fxc_contraction( P.rows(), P.cols(), P.data(), P.rows(),
                        tP.data(), tP.rows(),
                        FXC.data(), FXC.rows(), ks_settings );

  return FXC;
}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::fxc_contraction_type_uks
  ReplicatedXCIntegrator<MatrixType>::eval_fxc_contraction_( const MatrixType& Ps, const MatrixType& Pz, 
    const MatrixType& tPs, const MatrixType& tPz, const IntegratorSettingsXC& ks_settings ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  matrix_type FXCs( Ps.rows(), Ps.cols() );
  matrix_type FXCz( Pz.rows(), Pz.cols() );

  pimpl_->eval_fxc_contraction( Ps.rows(), Ps.cols(), Ps.data(), Ps.rows(),
                        Pz.data(), Pz.rows(),
                        tPs.data(), tPs.rows(),
                        tPz.data(), tPz.rows(),
                        FXCs.data(), FXCs.rows(),
                        FXCz.data(), FXCz.rows(), ks_settings );

  return std::make_tuple( FXCs, FXCz );

}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::dd_psi_type
  ReplicatedXCIntegrator<MatrixType>::eval_dd_psi_( const MatrixType& P, unsigned max_Ylm ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();

  const size_t natoms = pimpl_->load_balancer().molecule().natoms();
  const size_t Ylm_sz = (max_Ylm + 1) * ( max_Ylm + 1);
  std::vector<value_type> ddPsi(natoms * Ylm_sz, 0.0);
  pimpl_->eval_dd_psi(P.rows(), P.cols(), P.data(), P.rows(), max_Ylm, ddPsi.data(), Ylm_sz);
  return ddPsi;
}

template <typename MatrixType>
typename ReplicatedXCIntegrator<MatrixType>::dd_psi_potential_type
  ReplicatedXCIntegrator<MatrixType>::eval_dd_psi_potential_( const MatrixType& X, unsigned max_Ylm ) {

  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();

  const size_t nbf = pimpl_->load_balancer().basis().nbf();
  matrix_type Vddx(nbf, nbf);
  Vddx.setZero(); 
  pimpl_->eval_dd_psi_potential(X.rows(), X.cols(), X.data(), max_Ylm, Vddx.data());
  return Vddx;                      

}

}
}
