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
#include "local_device_work_driver.hpp"


namespace GauXC {
namespace detail {

struct LocalDeviceWorkDriverPIMPL;


/// Base class for local work drivers in Device execution spaces 
struct LocalDeviceWorkDriverPIMPL {

  using host_task_iterator = LocalDeviceWorkDriver::host_task_iterator;

  LocalDeviceWorkDriverPIMPL();
  virtual ~LocalDeviceWorkDriverPIMPL() noexcept;

  LocalDeviceWorkDriverPIMPL( const LocalDeviceWorkDriverPIMPL& )     = delete;
  LocalDeviceWorkDriverPIMPL( LocalDeviceWorkDriverPIMPL&& ) noexcept = delete;


  // Public APIs

  virtual void partition_weights( XCDeviceData* ) = 0;
  virtual void eval_weight_1st_deriv_contracted( XCDeviceData*, XCWeightAlg ) = 0;
  virtual void eval_collocation( XCDeviceData* ) = 0;
  virtual void eval_collocation_gradient( XCDeviceData* ) = 0;
  virtual void eval_collocation_hessian( XCDeviceData* ) = 0;
  virtual void eval_collocation_laplacian( XCDeviceData* ) = 0;
  virtual void eval_collocation_lapgrad( XCDeviceData* ) = 0;
  virtual void eval_xmat( double fac, XCDeviceData*, bool do_grad, density_id den ) = 0;
  virtual void save_xmat( XCDeviceData*, bool do_grad, density_id den ) = 0;
  virtual void eval_exx_fmat( XCDeviceData* ) = 0;
  //virtual void eval_exx_gmat( XCDeviceData* ) = 0;
  virtual void eval_exx_gmat( XCDeviceData*, const BasisSetMap& ) = 0;
  virtual void eval_uvars_lda( XCDeviceData*, integrator_ks_scheme ) = 0;
  virtual void eval_uvars_gga( XCDeviceData*, integrator_ks_scheme ) = 0;
  virtual void eval_uvars_mgga( XCDeviceData*, integrator_ks_scheme, bool ) = 0;
  virtual void eval_vvars_lda ( XCDeviceData*, density_id ) = 0;
  virtual void eval_vvars_gga ( XCDeviceData*, density_id ) = 0;
  virtual void eval_vvars_mgga( XCDeviceData*, density_id, bool ) = 0;
  virtual void eval_kern_exc_vxc_lda( const functional_type&, XCDeviceData* ) = 0;
  virtual void eval_kern_exc_vxc_gga( const functional_type&, XCDeviceData* ) = 0;
  virtual void eval_kern_exc_vxc_mgga( const functional_type&, XCDeviceData* ) = 0;
  virtual void eval_kern_vxc_fxc_lda( const functional_type&, XCDeviceData* ) = 0;
  virtual void eval_kern_vxc_fxc_gga( const functional_type&, XCDeviceData* ) = 0;
  virtual void eval_kern_vxc_fxc_mgga( const functional_type&, XCDeviceData* ) = 0;
  virtual void eval_zmat_lda_vxc( XCDeviceData*, integrator_ks_scheme, density_id ) = 0;
  virtual void eval_zmat_gga_vxc( XCDeviceData*, integrator_ks_scheme, density_id ) = 0;
  virtual void eval_zmat_mgga_vxc( XCDeviceData*, integrator_ks_scheme, bool, density_id ) = 0;
  virtual void eval_mmat_mgga_vxc( XCDeviceData*, integrator_ks_scheme, bool, density_id ) = 0;
  virtual void eval_zmat_lda_fxc( XCDeviceData*, density_id ) = 0;
  virtual void eval_zmat_gga_fxc( XCDeviceData*, density_id ) = 0;
  virtual void eval_zmat_mgga_fxc( XCDeviceData*, bool, density_id ) = 0;
  virtual void eval_mmat_mgga_fxc( XCDeviceData*, bool, density_id ) = 0;
  virtual void inc_exc( XCDeviceData* ) = 0;
  virtual void inc_nel( XCDeviceData* ) = 0;
  virtual void inc_vxc( XCDeviceData* , density_id, bool) = 0;
  virtual void inc_fxc( XCDeviceData* , density_id, bool) = 0;  
  virtual void inc_exc_grad_lda( XCDeviceData*, integrator_ks_scheme, bool  ) = 0;
  virtual void inc_exc_grad_gga( XCDeviceData*, integrator_ks_scheme, bool  ) = 0;
  virtual void inc_exc_grad_mgga( XCDeviceData*, integrator_ks_scheme , bool, bool ) = 0;
  virtual void inc_exx_k( XCDeviceData* ) = 0;
  virtual void symmetrize_vxc( XCDeviceData*, density_id ) = 0;
  virtual void symmetrize_fxc( XCDeviceData*, density_id ) = 0;
  virtual void symmetrize_exx_k( XCDeviceData* ) = 0;

  //second derivative
  virtual void eval_xmat_trial( double fac, XCDeviceData*, bool do_grad, density_id den ) = 0;
  virtual void eval_tmat_lda( XCDeviceData*, integrator_ks_scheme ) = 0;
  virtual void eval_tmat_gga( XCDeviceData*, integrator_ks_scheme ) = 0;
  virtual void eval_tmat_mgga( XCDeviceData*, integrator_ks_scheme, bool ) = 0;
  virtual void eval_vvars_lda_trial ( XCDeviceData*, density_id ) = 0;
  virtual void eval_vvars_gga_trial ( XCDeviceData*, density_id ) = 0;
  virtual void eval_vvars_mgga_trial( XCDeviceData*, density_id, bool ) = 0;

  virtual void eval_exx_ek_screening_bfn_stats( XCDeviceData* ) = 0;
  virtual void exx_ek_shellpair_collision( double eps_E, double eps_K, 
    XCDeviceData*, host_task_iterator, host_task_iterator, 
    const ShellPairCollection<double>&) = 0;

  virtual std::unique_ptr<XCDeviceData> create_device_data(const DeviceRuntimeEnvironment&) = 0;

  /****************************************************************************
   *      Multiparticle (NEO) inter-species API -- design Phase-2 §1.6        *
   *                                                                          *
   *  These are the ONLY two entries the multiparticle driver adds; every      *
   *  other stage of the species-serial loop reuses the single-species entries *
   *  above verbatim through the per-species context slots.                    *
   *                                                                          *
   *  They are deliberately not pure virtual.  A backend whose data layer is   *
   *  not multiparticle-aware (MAGMA / CUTLASS -- see WP2A2 §7.1) opts out by  *
   *  returning false from `supports_multiparticle()`, which the driver checks *
   *  once before any allocation, so these bodies are unreachable there.  The  *
   *  throwing defaults exist so a future backend fails loudly rather than     *
   *  silently doing nothing.                                                  *
   ****************************************************************************/

  /// Can this work driver (and its XCDeviceData) run the multiparticle path?
  virtual bool supports_multiparticle() const { return true; }

  /// EPC pack / evaluate / de-interleave / weight / scatter for one pair
  virtual void eval_kern_exc_vxc_inter_lda( const functional_type&, XCDeviceData*,
    const multiparticle_tracker&, size_t ) {
    GAUXC_GENERIC_EXCEPTION("MultiParticle inter-species XC is NYI for this LWD");
  }

  /// Accumulate one pair's inter-species EXC into its device accumulator
  virtual void inc_inter_exc( XCDeviceData*, const multiparticle_tracker&,
    size_t ) {
    GAUXC_GENERIC_EXCEPTION("MultiParticle inter-species EXC is NYI for this LWD");
  }

};

}
}

