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
#include <gauxc/xc_integrator/replicated/replicated_xc_host_integrator.hpp>
#include "xc_host_data.hpp"

namespace GauXC::detail {

template <typename ValueType>
class ReferenceReplicatedXCHostIntegrator : 
  public ReplicatedXCHostIntegrator<ValueType> {

  using base_type  = ReplicatedXCHostIntegrator<ValueType>;

public:

  static constexpr bool is_device = false;
  using value_type = typename base_type::value_type;
  using basis_type = typename base_type::basis_type;
  using multiparticle_density = typename base_type::multiparticle_density;
  using multiparticle_vxc = typename base_type::multiparticle_vxc;
  using task_container = std::vector<XCTask>;
  using task_iterator  = typename task_container::iterator;


protected:

  // Density Integration 
  void integrate_den_( int64_t m, int64_t n, const value_type* P, int64_t ldp, value_type* N_EL ) override;

  /// RKS EXC
  void eval_exc_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                  value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

  /// UKS EXC
  void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                  const value_type* Pz, int64_t ldpz,
                  value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

  /// GKS EXC 
  void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                      const value_type* Pz, int64_t ldpz,
                      const value_type* Py, int64_t ldpy,
                      const value_type* Px, int64_t ldpx,
                      value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

    /// DKS EXC - also serves as the generic implementation
  void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
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
                      value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

  /// RKS EXC/VXC
  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                      value_type* VXC, int64_t ldvxc, value_type* EXC, 
                      const IntegratorSettingsXC& ks_settings ) override;

  /// UKS EXC/VXC
  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                      const value_type* Pz, int64_t ldpz,
                      value_type* VXCs, int64_t ldvxcs,
                      value_type* VXCz, int64_t ldvxcz,
                      value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

  
  /// GKS EXC/VXC
  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
                      const value_type* Pz, int64_t ldpz,
                      const value_type* Py, int64_t ldpy,
                      const value_type* Px, int64_t ldpx,
                      value_type* VXCs, int64_t ldvxcs,
                      value_type* VXCz, int64_t ldvxcz,
                      value_type* VXCy, int64_t ldvxcy,
                      value_type* VXCx, int64_t ldvxcx,
                      value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

  /// MultiParticle EXC/VXC
  void eval_exc_vxc_( const std::vector<multiparticle_density>& densities,
                      const MultiParticleFunctionalSpec& functional_spec,
                      const MultiParticleXCTerms& terms,
                      std::vector<multiparticle_vxc>& vxc,
                      value_type* intra_exc,
                      value_type* inter_pair_exc,
                      const IntegratorSettingsXC& ks_settings ) override;


  /// DKS EXC/VXC - also serves as the generic implementation
  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps,
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
                      value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

  /// RKS EXC Gradient
  void eval_exc_grad_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                       value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) override;
  /// UKS EXC Gradient
  void eval_exc_grad_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                       const value_type* Pz, int64_t lpdz, value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) override;

  /// MultiParticle EXC Gradient
  void eval_exc_grad_( const std::vector<multiparticle_density>& densities,
                       const MultiParticleFunctionalSpec& functional_spec,
                       const MultiParticleXCTerms& terms,
                       value_type* EXC_GRAD,
                       const IntegratorSettingsXC& settings ) override;

  /// sn-LinK
  void eval_exx_( int64_t m, int64_t n, const value_type* P,
                  int64_t ldp, value_type* K, int64_t ldk,
                  const IntegratorSettingsEXX& settings ) override;

  /// RKS FXC contraction
  void eval_fxc_contraction_( int64_t m, int64_t n, 
                    const value_type* P, int64_t ldp, 
                    const value_type* tP, int64_t ldtp,
                    value_type* FXC, int64_t ldfxc,
                    const IntegratorSettingsXC& ks_settings ) override;

  // UKS FXC contraction
  void eval_fxc_contraction_( int64_t m, int64_t n, 
                    const value_type* Ps, int64_t ldps,   
                    const value_type* Pz, int64_t ldpz,
                    const value_type* tPs, int64_t ldtps,
                    const value_type* tPz, int64_t ldtpz,
                    value_type* FXCs, int64_t ldfxcs,
                    value_type* FXCz, int64_t ldfxcz,
                    const IntegratorSettingsXC& ks_settings ) override;

  /// ddX PSi 
  void eval_dd_psi_( int64_t m, int64_t n, const value_type* P,
                     int64_t ldp, unsigned max_Ylm, value_type* ddPsi, int64_t ldPsi ) override;

  /// ddX PhiX
  void eval_dd_psi_potential_( int64_t m, int64_t n, const value_type* X, unsigned max_Ylm, value_type* Vddx ) override;

  // Implementation details of integrate_den
  void integrate_den_local_work_( const value_type* P, int64_t ldp, 
                                   value_type *N_EL );

  // Implementation details of exc_vxc (for RKS/UKS/GKS/DKS deduced from input character)
  void exc_vxc_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
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
                            value_type* EXC, value_type *N_EL, value_type *spin_N_EL, 
                            const IntegratorSettingsXC& ks_settings,
                            task_iterator task_begin, task_iterator task_end );
                            
  // Implementation details of the MultiParticle exc_vxc (per-rank local work)
  void multiparticle_exc_vxc_local_work_( const std::vector<multiparticle_density>& densities,
                            const MultiParticleFunctionalSpec& functional_spec,
                            const MultiParticleXCTerms& terms,
                            std::vector<multiparticle_vxc>& vxc,
                            value_type* intra_exc,
                            value_type* inter_pair_exc,
                            const IntegratorSettingsXC& ks_settings,
                            task_iterator task_begin, task_iterator task_end );

  // Implemetation details of exc_grad
  void exc_grad_local_work_( const value_type* Ps, int64_t ldps, const value_type* Pz, int64_t ldpz,
                             value_type* EXC_GRAD, const IntegratorSettingsXC& ks_settings );

  // Implementation details of the MultiParticle exc_grad (per-rank local work)
  void multiparticle_exc_grad_local_work_( const std::vector<multiparticle_density>& densities,
                             const MultiParticleFunctionalSpec& functional_spec,
                             const MultiParticleXCTerms& terms,
                             value_type* EXC_GRAD,
                             const IntegratorSettingsXC& settings );

  // Implementation details of sn-LinK
  void exx_local_work_( const value_type* P, int64_t ldp, value_type* K, int64_t ldk,
    const IntegratorSettingsEXX& settings );

  // Implementation details of UKS FXC contraction
  void fxc_contraction_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* tPs, int64_t ldtps,
                            const value_type* tPz, int64_t ldtpz,
                            value_type* FXCs, int64_t ldfxcs,
                            value_type* FXCz, int64_t ldfxcz,
                            value_type *N_EL, const IntegratorSettingsXC& ks_settings,
                            task_iterator task_begin, task_iterator task_end );

  // Implementation details of ddX Psi
  void dd_psi_local_work_( const value_type* P, int64_t ldp, unsigned max_Ylm, value_type* ddPsi, int64_t ldPsi );    

  void dd_psi_potential_local_work_( const value_type* X, value_type* Vddx, unsigned max_Ylm );
  
public:

  template <typename... Args>
  ReferenceReplicatedXCHostIntegrator( Args&&... args ) :
    base_type( std::forward<Args>(args)... ) { }

  virtual ~ReferenceReplicatedXCHostIntegrator() noexcept;


  template <typename... Args>
  void exc_vxc_local_work(Args&&... args) {
    exc_vxc_local_work_( std::forward<Args>(args)... );
  }


};

extern template class ReferenceReplicatedXCHostIntegrator<double>;

} // namespace GauXC::detail
