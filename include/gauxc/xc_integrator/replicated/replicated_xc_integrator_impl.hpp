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

#include <gauxc/xc_integrator/replicated_xc_integrator.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include <gauxc/reduction_driver.hpp>
#include <gauxc/types.hpp>
#include <gauxc/basisset.hpp>

namespace GauXC  {
namespace detail {


/// Base class for ReplicatedXCIntegrator implementations
template <typename ValueType>
class ReplicatedXCIntegratorImpl {

public:

  using value_type = ValueType;
  using basis_type = BasisSet< value_type >;
  struct multiparticle_density {
    int64_t m = 0;
    int64_t n = 0;
    const value_type* Ps = nullptr;
    int64_t ldps = 0;
    const value_type* Pz = nullptr;
    int64_t ldpz = 0;
  };
  struct multiparticle_vxc {
    value_type* VXCs = nullptr;
    int64_t ldvxcs = 0;
    value_type* VXCz = nullptr;
    int64_t ldvxcz = 0;
  };

protected:

  std::shared_ptr< functional_type > func_;               ///< XC functional
  std::shared_ptr< LoadBalancer >    load_balancer_;      ///< Load Balancer
  std::unique_ptr< LocalWorkDriver > local_work_driver_;  ///< Local Work Driver
  std::shared_ptr< ReductionDriver > reduction_driver_;   ///< Reduction Driver

  util::Timer timer_;


  virtual void integrate_den_( int64_t m, int64_t n, const value_type* P,
                               int64_t ldp, value_type* N_EL ) = 0;

  virtual void eval_exc_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                          value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                          const value_type* Pz, int64_t ldpz, 
                          value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                          const value_type* Pz, int64_t ldpz, 
                          const value_type* Py, int64_t ldpy, 
                          const value_type* Px, int64_t ldpx, 
                          value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;

  virtual void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                          const value_type* Pz, int64_t ldpz, 
                          const value_type* Py, int64_t ldpy, 
                          const value_type* Px, int64_t ldpx, 
                          const value_type* Ps_SS, int64_t ldps_ss,
                          const value_type* Pz_SS, int64_t ldpz_ss, 
                          const value_type* Py_SS, int64_t ldpy_ss, 
                          const value_type* Px_SS, int64_t ldpx_ss, 
                          const value_type* Ps_SS_imag,
                          const value_type* Pz_SS_imag,
                          const value_type* Py_SS_imag,
                          const value_type* Px_SS_imag,
                          value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;

  virtual void eval_exc_vxc_( int64_t m, int64_t n, const value_type* P,
                              int64_t ldp, value_type* VXC, int64_t ldvxc,
                              value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps,
                              int64_t ldps,
                              const value_type* Pz,
                              int64_t ldpz,
                              value_type* VXCs, int64_t ldvxcs,
                              value_type* VXCz, int64_t ldvxcz,
                              value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps,
                              int64_t ldps,
                              const value_type* Pz,
                              int64_t ldpz,
                              const value_type* Py,
                              int64_t ldpy,
                              const value_type* Px,
                              int64_t ldpx,
                              value_type* VXCs, int64_t ldvxcs,
                              value_type* VXCz, int64_t ldvxcz,
                              value_type* VXCy, int64_t ldvxcy,
                              value_type* VXCx, int64_t ldvxcx,
                              value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;
// dks
  virtual void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps,
                              int64_t ldps,
                              const value_type* Pz,
                              int64_t ldpz,
                              const value_type* Py,
                              int64_t ldpy,
                              const value_type* Px,
                              int64_t ldpx,
                              const value_type* Ps_SS,
                              int64_t ldps_ss,
                              const value_type* Pz_SS,
                              int64_t ldpz_ss,
                              const value_type* Py_SS,
                              int64_t ldpy_ss,
                              const value_type* Px_SS,
                              int64_t ldpx_ss,
                              const value_type* Ps_SS_imag,
                              const value_type* Pz_SS_imag,
                              const value_type* Py_SS_imag,
                              const value_type* Px_SS_imag,
                              value_type* VXCs, int64_t ldvxcs,
                              value_type* VXCz, int64_t ldvxcz,
                              value_type* VXCy, int64_t ldvxcy,
                              value_type* VXCx, int64_t ldvxcx,
                              value_type* VXCs_SS, int64_t ldvxcs_ss,
                              value_type* VXCz_SS, int64_t ldvxcz_ss,
                              value_type* VXCy_SS, int64_t ldvxcy_ss,
                              value_type* VXCx_SS, int64_t ldvxcx_ss,
                              value_type* VXCs_SS_im, int64_t ldvxcs_ss_im,
                              value_type* VXCz_SS_im, int64_t ldvxcz_ss_im,
                              value_type* VXCy_SS_im, int64_t ldvxcy_ss_im,
                              value_type* VXCx_SS_im, int64_t ldvxcx_ss_im,
                              value_type* EXC, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual void eval_exc_vxc_( const std::vector<multiparticle_density>&,
                              const MultiParticleFunctionalSpec&,
                              const MultiParticleXCTerms&,
                              std::vector<multiparticle_vxc>&,
                              value_type*, value_type*,
                              const IntegratorSettingsXC& ) {
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC is not implemented for this integrator");
  }

  virtual void eval_exc_grad_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                               value_type* EXC_GRAD, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual void eval_exc_grad_( int64_t m, int64_t n, const value_type* P, int64_t ldps, 
                               const value_type* Pz, int64_t lpdz, value_type* EXC_GRAD, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual void eval_exc_grad_( const std::vector<multiparticle_density>&,
                               const MultiParticleFunctionalSpec&,
                               const MultiParticleXCTerms&,
                               value_type*,
                               const IntegratorSettingsXC& ) {
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient is not implemented for this integrator");
  }
  virtual void eval_exx_( int64_t m, int64_t n, const value_type* P,
                          int64_t ldp, value_type* K, int64_t ldk,
                          const IntegratorSettingsEXX& settings ) = 0;
  virtual void eval_fxc_contraction_( int64_t m, int64_t n, 
                            const value_type* P, int64_t ldp,
                            const value_type* tP, int64_t ldtp,
                            value_type* FXC, int64_t ldfxc,
                            const IntegratorSettingsXC& ks_settings )=0;
  virtual void eval_fxc_contraction_( int64_t m, int64_t n, 
                            const value_type* Ps, int64_t ldps,   
                            const value_type* Pz, int64_t ldpz,
                            const value_type* tPs, int64_t ldtps,
                            const value_type* tPz, int64_t ldtpz,
                            value_type* FXCs, int64_t ldfxcs,
                            value_type* FXCz, int64_t ldfxcz,
                            const IntegratorSettingsXC& ks_settings )=0;
  virtual void eval_dd_psi_( int64_t m, int64_t n, const value_type* P, int64_t ldp, unsigned max_Ylm, 
                             value_type* ddPsi, int64_t ldPsi ) = 0;
  virtual void eval_dd_psi_potential_( int64_t m, int64_t n, const value_type* X, unsigned max_Ylm,
                             value_type* Vddx) = 0;

public:

  ReplicatedXCIntegratorImpl( std::shared_ptr< functional_type >   func,
                              std::shared_ptr< LoadBalancer >      lb, 
                              std::unique_ptr< LocalWorkDriver >&& lwd,
                              std::shared_ptr< ReductionDriver>    rd
                              );

  virtual ~ReplicatedXCIntegratorImpl() noexcept;

  void integrate_den( int64_t m, int64_t n, const value_type* P,
                      int64_t ldp, value_type* N_EL );

  void eval_exc( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                 value_type* EXC, const IntegratorSettingsXC& ks_settings ); 
  void eval_exc( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz, 
                 value_type* EXC, const IntegratorSettingsXC& ks_settings );
  void eval_exc( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz, 
                 const value_type* Py, int64_t ldpy, 
                 const value_type* Px, int64_t ldpx, 
                 value_type* EXC, const IntegratorSettingsXC& ks_settings );

  void eval_exc( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz, 
                 const value_type* Py, int64_t ldpy, 
                 const value_type* Px, int64_t ldpx, 
                 const value_type* Ps_SS, int64_t ldps_ss,
                 const value_type* Pz_SS, int64_t ldpz_ss, 
                 const value_type* Py_SS, int64_t ldpy_ss, 
                 const value_type* Px_SS, int64_t ldpx_ss, 
                 const value_type* Ps_SS_imag,
                 const value_type* Pz_SS_imag,
                 const value_type* Py_SS_imag,
                 const value_type* Px_SS_imag,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings );

  void eval_exc_vxc( int64_t m, int64_t n, const value_type* P,
                     int64_t ldp, value_type* VXC, int64_t ldvxc,
                     value_type* EXC, const IntegratorSettingsXC& ks_settings ); 

  void eval_exc_vxc( int64_t m, int64_t n, const value_type* Ps,
                     int64_t ldps,
                     const value_type* Pz,
                     int64_t ldpz,
                     value_type* VXCs, int64_t ldvxcs,
                     value_type* VXCz, int64_t ldvxcz,
                     value_type* EXC, const IntegratorSettingsXC& ks_settings );
  void eval_exc_vxc( int64_t m, int64_t n, const value_type* Ps,
                     int64_t ldps,
                     const value_type* Pz,
                     int64_t ldpz,
                     const value_type* Py,
                     int64_t ldpy,
                     const value_type* Px,
                     int64_t ldpx,
                     value_type* VXCs, int64_t ldvxcs,
                     value_type* VXCz, int64_t ldvxcz,
                     value_type* VXCy, int64_t ldvxcy,
                     value_type* VXCx, int64_t ldvxcx,
                     value_type* EXC, const IntegratorSettingsXC& ks_settings );
// dks
  void eval_exc_vxc( int64_t m, int64_t n, const value_type* Ps,
                     int64_t ldps,
                     const value_type* Pz,
                     int64_t ldpz,
                     const value_type* Py,
                     int64_t ldpy,
                     const value_type* Px,
                     int64_t ldpx,
                     const value_type* Ps_SS,
                     int64_t ldps_ss,
                     const value_type* Pz_SS,
                     int64_t ldpz_ss,
                     const value_type* Py_SS,
                     int64_t ldpy_ss,
                     const value_type* Px_SS,
                     int64_t ldpx_ss,
                     const value_type* Ps_SS_imag,
                     const value_type* Pz_SS_imag,
                     const value_type* Py_SS_imag,
                     const value_type* Px_SS_imag,
                     value_type* VXCs, int64_t ldvxcs,
                     value_type* VXCz, int64_t ldvxcz,
                     value_type* VXCy, int64_t ldvxcy,
                     value_type* VXCx, int64_t ldvxcx,
                     value_type* VXCs_SS, int64_t ldvxcs_ss,
                     value_type* VXCz_SS, int64_t ldvxcz_ss,
                     value_type* VXCy_SS, int64_t ldvxcy_ss,
                     value_type* VXCx_SS, int64_t ldvxcx_ss,
                     value_type* VXCs_SS_im, int64_t ldvxcs_ss_im,
                     value_type* VXCz_SS_im, int64_t ldvxcz_ss_im,
                     value_type* VXCy_SS_im, int64_t ldvxcy_ss_im,
                     value_type* VXCx_SS_im, int64_t ldvxcx_ss_im,
                     value_type* EXC, const IntegratorSettingsXC& ks_settings );
  
  void eval_exc_vxc( const std::vector<multiparticle_density>& densities,
                     const MultiParticleFunctionalSpec& functional_spec,
                     const MultiParticleXCTerms& terms,
                     std::vector<multiparticle_vxc>& vxc,
                     value_type* intra_exc,
                     value_type* inter_pair_exc,
                     const IntegratorSettingsXC& ks_settings );


  void eval_exc_grad( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                      value_type* EXC_GRAD, const IntegratorSettingsXC& ks_settings );
  void eval_exc_grad( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                      const value_type* Pz, int64_t ldpz, value_type* EXC_GRAD, const IntegratorSettingsXC& ks_settings );
  void eval_exc_grad( const std::vector<multiparticle_density>& densities,
                      const MultiParticleFunctionalSpec& functional_spec,
                      const MultiParticleXCTerms& terms,
                      value_type* EXC_GRAD,
                      const IntegratorSettingsXC& ks_settings );

  void eval_exx( int64_t m, int64_t n, const value_type* P,
                 int64_t ldp, value_type* K, int64_t ldk,
                 const IntegratorSettingsEXX& settings );

  void eval_fxc_contraction( int64_t m, int64_t n, const value_type* P,
                      int64_t ldp,
                      const value_type* tP, int64_t ldtp,
                      value_type* FXC, int64_t ldfxc,
                      const IntegratorSettingsXC& ks_settings );

  void eval_fxc_contraction( int64_t m, int64_t n, const value_type* Ps,
                      int64_t ldps,
                      const value_type* Pz, int64_t ldpz,
                      const value_type* tPs, int64_t ldtps,
                      const value_type* tPz, int64_t ldtpz,
                      value_type* FXCs, int64_t ldfxcs,
                      value_type* FXCz, int64_t ldfxcz,
                      const IntegratorSettingsXC& ks_settings );

  void eval_dd_psi( int64_t m, int64_t n, const value_type* P,
                     int64_t ldp, unsigned max_Ylm, 
                     value_type* ddPsi, int64_t ldPsi );
  void eval_dd_psi_potential( int64_t m, int64_t n, const value_type* X, unsigned max_Ylm, 
                      value_type* Vddx );

  inline const util::Timer& get_timings() const { return timer_; }

  inline std::unique_ptr< LocalWorkDriver > release_local_work_driver() {
    return std::move( local_work_driver_ );
  }

  inline const auto& load_balancer() const { return *load_balancer_; }
  inline auto& load_balancer() { return *load_balancer_; }
  inline const auto& get_load_balancer() const { return load_balancer(); }
  inline auto& get_load_balancer() { return load_balancer(); }
};


extern template class ReplicatedXCIntegratorImpl<double>;

}
}
