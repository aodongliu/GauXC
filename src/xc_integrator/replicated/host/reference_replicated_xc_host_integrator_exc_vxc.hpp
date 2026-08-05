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
#include <stdexcept>

#include <gauxc/physcon.hpp>

namespace GauXC::detail {

/**
 *  Generic implementation of EXC/VXC for RKS/UKS/GKS/DKS
 *  
 *  If passed pointers are null-y and the leading dimensions
 *  are zero, RKS/UKS are deduced. RKS/UKS drivers delegate
 *  to this function/
 */
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( int64_t m, int64_t n,
                 const value_type* Ps, int64_t ldps,
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
                 value_type* EXC, const IntegratorSettingsXC& ks_settings ) {


  const auto& basis = this->load_balancer_->basis();

  
  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();

  if( m != n )
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Be Square");
  if( m != nbf )
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Have Same Dimension as Basis");

  if( ldps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPS");
  if( ldpz and ldpz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPZ");
  if( ldpy and ldpy < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPX");
  if( ldpx and ldpx < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPY");

  if( ldvxcs < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCS");
  if( ldvxcz and ldvxcz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCZ");
  if( ldvxcy and ldvxcy < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCX");
  if( ldvxcx and ldvxcx < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCY");

  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Temporary electron count to judge integrator accuracy
  value_type N_EL;
  value_type spin_N_EL;


  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    exc_vxc_local_work_( basis, Ps, ldps, Pz, ldpz, Py, ldpy, Px, ldpx, 
                         Ps_SS, ldps_ss, Pz_SS, ldpz_ss, Py_SS, ldpy_ss, Px_SS, ldpx_ss, 
                         Ps_SS_imag, Pz_SS_imag, Py_SS_imag, Px_SS_imag, 
                         VXCs, ldvxcs, VXCz, ldvxcz,
                         VXCy, ldvxcy, VXCx, ldvxcx,
                         VXCs_SS, ldvxcs_ss, VXCz_SS, ldvxcz_ss,
                         VXCy_SS, ldvxcy_ss, VXCx_SS, ldvxcx_ss,
                         VXCs_SS_im, ldvxcs_ss_im, VXCz_SS_im, ldvxcz_ss_im,
                         VXCy_SS_im, ldvxcy_ss_im, VXCx_SS_im, ldvxcx_ss_im,
                         EXC, &N_EL, &spin_N_EL, ks_settings,
                         tasks.begin(), tasks.end() );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");
    
    this->reduction_driver_->allreduce_inplace( VXCs, nbf*nbf, ReductionOp::Sum );
    if(VXCz) this->reduction_driver_->allreduce_inplace( VXCz, nbf*nbf, ReductionOp::Sum );
    if(VXCy) this->reduction_driver_->allreduce_inplace( VXCy, nbf*nbf, ReductionOp::Sum ); 
    if(VXCx) this->reduction_driver_->allreduce_inplace( VXCx, nbf*nbf, ReductionOp::Sum );
    if(VXCs_SS) this->reduction_driver_->allreduce_inplace( VXCs_SS, nbf*nbf, ReductionOp::Sum );
    if(VXCz_SS) this->reduction_driver_->allreduce_inplace( VXCz_SS, nbf*nbf, ReductionOp::Sum );
    if(VXCy_SS) this->reduction_driver_->allreduce_inplace( VXCy_SS, nbf*nbf, ReductionOp::Sum ); 
    if(VXCx_SS) this->reduction_driver_->allreduce_inplace( VXCx_SS, nbf*nbf, ReductionOp::Sum );
    if(VXCs_SS_im) this->reduction_driver_->allreduce_inplace( VXCs_SS_im, nbf*nbf, ReductionOp::Sum );
    if(VXCz_SS_im) this->reduction_driver_->allreduce_inplace( VXCz_SS_im, nbf*nbf, ReductionOp::Sum );
    if(VXCy_SS_im) this->reduction_driver_->allreduce_inplace( VXCy_SS_im, nbf*nbf, ReductionOp::Sum ); 
    if(VXCx_SS_im) this->reduction_driver_->allreduce_inplace( VXCx_SS_im, nbf*nbf, ReductionOp::Sum );

    this->reduction_driver_->allreduce_inplace( EXC,   1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &spin_N_EL, 1    , ReductionOp::Sum );

  });


}


/// Generic implementation details of EXC/VXC local work - deduces RKS/UKS/GKS/DKS
/// based on null-y / zero parameters
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  exc_vxc_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
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
                       const IntegratorSettingsXC& settings,
                       task_iterator task_begin, task_iterator task_end ) {
    
  const bool is_dks = (Pz != nullptr) and (Py != nullptr) and (Px != nullptr) and (Ps_SS != nullptr);
  const bool is_gks = (Pz != nullptr) and (Py != nullptr) and (Px != nullptr) and (Ps_SS == nullptr);
  const bool is_uks = (Pz != nullptr) and (Py == nullptr) and (Px == nullptr) and (Ps_SS == nullptr);
  const bool is_rks = not is_uks and not is_gks and not is_dks;

  if (not is_rks and not is_uks and not is_gks and not is_dks) {
    GAUXC_GENERIC_EXCEPTION("Must Be Either RKS, UKS, GKS, or DKS!");
  }

  const bool is_exc_only = (!VXCs) and (!VXCz) and (!VXCy) and (!VXCx);
  //if(is_exc_only) std::cout << "EXC ONLY" << std::endl;
  // Misc KS settings
  IntegratorSettingsKS ks_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsKS*>(&settings) ) {
    ks_settings = *tmp;
  }

  const double gks_dtol = ks_settings.gks_dtol;

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& func  = *this->func_;
  const auto& mol   = this->load_balancer_->molecule();

  const bool needs_laplacian = func.needs_laplacian(); 
  
  if (func.is_mgga() and is_gks) {
    GAUXC_GENERIC_EXCEPTION("GKS Not Yet Implemented With MGGA Functionals!");
  }

  // Get basis map
  BasisSetMap basis_map(basis,mol);

  const int32_t nbf = basis.nbf();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };

  auto& tasks = this->load_balancer_->get_tasks();
  std::sort( task_begin, task_end, task_comparator );

  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");
  }

  // Zero out integrands
  
  if(VXCs)
  for( auto j = 0; j < nbf; ++j ) {
    for( auto i = 0; i < nbf; ++i ) {
      VXCs[i + j*ldvxcs] = 0.;
    }
  }

  if(VXCz) {
    for( auto j = 0; j < nbf; ++j ) {
      for( auto i = 0; i < nbf; ++i ) {
        VXCz[i + j*ldvxcz] = 0.;
      }
    }
  }

  if(VXCx and VXCy) {
    for( auto j = 0; j < nbf; ++j ) {
      for( auto i = 0; i < nbf; ++i ) {
        VXCy[i + j*ldvxcy] = 0.;
        VXCx[i + j*ldvxcx] = 0.;
      }
    }
  }

  if(VXCs_SS)
  for( auto j = 0; j < nbf; ++j ) {
    for( auto i = 0; i < nbf; ++i ) {
      VXCs_SS[i + j*ldvxcs] = 0.;
      VXCz_SS[i + j*ldvxcz] = 0.;
      VXCy_SS[i + j*ldvxcy] = 0.;
      VXCx_SS[i + j*ldvxcx] = 0.;
      VXCs_SS_im[i + j*ldvxcs] = 0.;
      VXCz_SS_im[i + j*ldvxcz] = 0.;
      VXCy_SS_im[i + j*ldvxcy] = 0.;
      VXCx_SS_im[i + j*ldvxcx] = 0.;
    }
  }
 
  double EXC_WORK = 0.0;
  double NEL_WORK = 0.0;
  double spin_NEL_WORK = 0.0;

  // Loop over tasks
  const size_t ntasks = std::distance(task_begin, task_end);

  #pragma omp parallel
  {

  XCHostData<value_type> host_data; // Thread local host data

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {
     
    //std::cout << iT << "/" << ntasks << std::endl;
    //if(is_exc_only) printf("%lu / %lu\n", iT, ntasks);
    // Alias current task
    const auto& task = *(task_begin + iT);

    // Get tasks constants
    const int32_t  npts    = static_cast<int32_t>(task.points.size());
    const int32_t  nbe     = task.bfn_screening.nbe;
    const int32_t  nshells = static_cast<int32_t>(task.bfn_screening.shell_list.size());

    const auto* points      = task.points.data()->data();
    const auto* weights     = task.weights.data();
    const int32_t* shell_list = task.bfn_screening.shell_list.data();
    
    // Allocate enough memory for batch
   
    const size_t spin_dim_scal = is_rks ? 1 : is_uks ? 2 : 4; // last case is_gks
    const size_t sds          = is_rks ? 1 : 2;
    const size_t gks_mod_KH = is_gks ? 6*npts : is_dks ? 6*npts : 0; // used to store K and H
    const size_t mgga_dim_scal = func.is_mgga() ? 4 : 1; // basis + d1basis
    const size_t dks_scal = is_dks ? 4 : 1;
    const size_t dks_im_mats = is_dks ? 9*npts*nbe : 0;

    // Things that every calc needs
    host_data.nbe_scr .resize(nbe  * nbe);
    host_data.zmat    .resize(npts * nbe * spin_dim_scal * mgga_dim_scal * dks_scal + gks_mod_KH + dks_im_mats); 
    host_data.eps     .resize(npts);
    host_data.vrho    .resize(npts * sds);

    // LDA data requirements
    if( func.is_lda() ){
      if( is_dks ){
        host_data.basis_eval .resize( 4 * npts * nbe ); // basis + grad
        host_data.den_scr    .resize( spin_dim_scal * npts);
      } else {
        host_data.basis_eval .resize( npts * nbe );
        host_data.den_scr    .resize( npts * spin_dim_scal);
      }
    }
     
    // GGA data requirements
    const size_t gga_dim_scal = is_rks ? 1 : 3;
    if( func.is_gga() ){
      if (is_dks ){
        host_data.basis_eval .resize( 10 * npts * nbe ); // basis + grad (3) + hess (6)
        host_data.den_scr    .resize( spin_dim_scal * 4 * npts );
        host_data.gamma      .resize( gga_dim_scal * npts );
        host_data.vgamma     .resize( gga_dim_scal * npts );
      } else {
        host_data.basis_eval .resize( 4 * npts * nbe );
        host_data.den_scr    .resize( spin_dim_scal * 4 * npts );
        host_data.gamma      .resize( gga_dim_scal * npts );
        host_data.vgamma     .resize( gga_dim_scal * npts );
      }
    }

    if( func.is_mgga() ){
      if ( needs_laplacian ) {
        host_data.basis_eval .resize( 11 * npts * nbe ); // basis + grad (3) + hess (6) + lapl 
        host_data.lapl       .resize( spin_dim_scal * npts );
        host_data.vlapl      .resize( spin_dim_scal * npts );
      } else {
        host_data.basis_eval .resize( 4 * npts * nbe ); // basis + grad (3)
      }

      host_data.den_scr    .resize( spin_dim_scal * 4 * npts );
      host_data.gamma      .resize( gga_dim_scal * npts );
      host_data.vgamma     .resize( gga_dim_scal * npts );
      host_data.tau        .resize( npts * spin_dim_scal );
      host_data.vtau       .resize( npts * spin_dim_scal );
      
    }

    // Alias/Partition out scratch memory
    auto* basis_eval = host_data.basis_eval.data();
    auto* den_eval   = host_data.den_scr.data();
    auto* nbe_scr    = host_data.nbe_scr.data();
    auto* zmat       = host_data.zmat.data();

    decltype(zmat) zmat_z = nullptr;
    decltype(zmat) zmat_x = nullptr;
    decltype(zmat) zmat_y = nullptr;

    decltype(zmat) zmat_s_ss = nullptr;
    decltype(zmat) zmat_z_ss = nullptr;
    decltype(zmat) zmat_x_ss = nullptr;
    decltype(zmat) zmat_y_ss = nullptr;


    decltype(zmat) xmat_y_s_ss = nullptr;
    decltype(zmat) xmat_y_z_ss = nullptr;
    decltype(zmat) xmat_y_x_ss = nullptr;
    decltype(zmat) xmat_y_y_ss = nullptr;

    decltype(zmat) xmat_z_s_ss = nullptr;
    decltype(zmat) xmat_z_z_ss = nullptr;
    decltype(zmat) xmat_z_x_ss = nullptr;
    decltype(zmat) xmat_z_y_ss = nullptr;

    decltype(zmat) immat_x_z = nullptr;
    decltype(zmat) immat_y_x = nullptr;
    decltype(zmat) immat_z_y = nullptr;

    decltype(zmat) immat_y_z = nullptr;
    decltype(zmat) immat_z_x = nullptr;
    decltype(zmat) immat_x_y = nullptr;

    decltype(zmat) immat_y_s = nullptr;
    decltype(zmat) immat_z_s = nullptr;
    decltype(zmat) immat_x_s = nullptr;


    if(!is_rks) {
      zmat_z = zmat + mgga_dim_scal * nbe * npts;
    }
    if(!is_uks) {
      zmat_x = zmat_z + nbe * npts;
      zmat_y = zmat_x + nbe * npts;
    }
    if(is_dks) {
      zmat_s_ss = zmat_y + nbe * npts;
      zmat_z_ss = zmat_s_ss + nbe * npts;
      zmat_x_ss = zmat_z_ss + nbe * npts;
      zmat_y_ss = zmat_x_ss + nbe * npts;

      xmat_y_s_ss = zmat_y_ss + nbe * npts;
      xmat_y_z_ss = xmat_y_s_ss + nbe * npts;
      xmat_y_x_ss = xmat_y_z_ss + nbe * npts;
      xmat_y_y_ss = xmat_y_x_ss + nbe * npts;

      xmat_z_s_ss = xmat_y_y_ss + nbe * npts;
      xmat_z_z_ss = xmat_z_s_ss + nbe * npts;
      xmat_z_x_ss = xmat_z_z_ss + nbe * npts;
      xmat_z_y_ss = xmat_z_x_ss + nbe * npts;

      immat_x_z = xmat_z_y_ss + nbe * npts;
      immat_y_x = immat_x_z + nbe * npts;
      immat_z_y = immat_y_x + nbe * npts;

      immat_y_z = immat_z_y  + nbe * npts;
      immat_z_x = immat_y_z + nbe * npts;
      immat_x_y = immat_z_x + nbe * npts;

      immat_y_s = immat_x_y  + nbe * npts;
      immat_z_s = immat_y_s + nbe * npts;
      immat_x_s = immat_z_s + nbe * npts;

    }
     
    auto* eps        = host_data.eps.data();
    auto* gamma      = host_data.gamma.data();
    auto* tau        = host_data.tau.data();
    auto* lapl       = host_data.lapl.data();
    auto* vrho       = host_data.vrho.data();
    auto* vgamma     = host_data.vgamma.data();
    auto* vtau       = host_data.vtau.data();
    auto* vlapl      = host_data.vlapl.data();

    value_type* dbasis_x_eval = nullptr;
    value_type* dbasis_y_eval = nullptr;
    value_type* dbasis_z_eval = nullptr;
    value_type* d2basis_xx_eval = nullptr;
    value_type* d2basis_xy_eval = nullptr;
    value_type* d2basis_xz_eval = nullptr;
    value_type* d2basis_yy_eval = nullptr;
    value_type* d2basis_yz_eval = nullptr;
    value_type* d2basis_zz_eval = nullptr;
    value_type* lbasis_eval = nullptr;
    value_type* dden_x_eval = nullptr;
    value_type* dden_y_eval = nullptr;
    value_type* dden_z_eval = nullptr;
    value_type* K = nullptr;
    value_type* H = nullptr;
    if (is_gks or is_dks) { K = zmat + npts * nbe * 4 * dks_scal + dks_im_mats; }
    value_type* mmat_x      = nullptr;
    value_type* mmat_y      = nullptr;
    value_type* mmat_z      = nullptr;
    value_type* mmat_x_z    = nullptr;
    value_type* mmat_y_z    = nullptr;
    value_type* mmat_z_z    = nullptr;

    if( func.is_lda() and is_dks ){
        dbasis_x_eval = basis_eval    + npts * nbe;
        dbasis_y_eval = dbasis_x_eval + npts * nbe;
        dbasis_z_eval = dbasis_y_eval + npts * nbe;
    }
    if( func.is_gga() ) {
      if( is_dks ){
        dbasis_x_eval = basis_eval    + npts * nbe;
        dbasis_y_eval = dbasis_x_eval + npts * nbe;
        dbasis_z_eval = dbasis_y_eval + npts * nbe;
        
        dden_x_eval   = den_eval    + spin_dim_scal * npts;
        dden_y_eval   = dden_x_eval + spin_dim_scal * npts;
        dden_z_eval   = dden_y_eval + spin_dim_scal * npts;
        d2basis_xx_eval = dbasis_z_eval + npts * nbe;
        d2basis_xy_eval = d2basis_xx_eval + npts * nbe;
        d2basis_xz_eval = d2basis_xy_eval + npts * nbe;
        d2basis_yy_eval = d2basis_xz_eval + npts * nbe;
        d2basis_yz_eval = d2basis_yy_eval + npts * nbe;
        d2basis_zz_eval = d2basis_yz_eval + npts * nbe;
        H = K + 3*npts;
      } else {
        dbasis_x_eval = basis_eval    + npts * nbe;
        dbasis_y_eval = dbasis_x_eval + npts * nbe;
        dbasis_z_eval = dbasis_y_eval + npts * nbe;
        dden_x_eval   = den_eval    + spin_dim_scal * npts;
        dden_y_eval   = dden_x_eval + spin_dim_scal * npts;
        dden_z_eval   = dden_y_eval + spin_dim_scal * npts;
      if (is_gks) { H = K + 3*npts;}
      }
    }

    if ( func.is_mgga() ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
      dden_x_eval   = den_eval    + spin_dim_scal * npts;
      dden_y_eval   = dden_x_eval + spin_dim_scal * npts;
      dden_z_eval   = dden_y_eval + spin_dim_scal * npts;
      mmat_x        = zmat + npts * nbe;
      mmat_y        = mmat_x + npts * nbe;
      mmat_z        = mmat_y + npts * nbe;
      if ( needs_laplacian ) {
        d2basis_xx_eval = dbasis_z_eval + npts * nbe;
        d2basis_xy_eval = d2basis_xx_eval + npts * nbe;
        d2basis_xz_eval = d2basis_xy_eval + npts * nbe;
        d2basis_yy_eval = d2basis_xz_eval + npts * nbe;
        d2basis_yz_eval = d2basis_yy_eval + npts * nbe;
        d2basis_zz_eval = d2basis_yz_eval + npts * nbe;
        lbasis_eval     = d2basis_zz_eval + npts * nbe;
      }
      if(is_uks) {
        mmat_x_z = zmat_z + npts * nbe;
        mmat_y_z = mmat_x_z + npts * nbe;
        mmat_z_z = mmat_y_z + npts * nbe;
      }
    }

    // Get the submatrix map for batch
    std::vector< std::array<int32_t, 3> > submat_map;
    std::tie(submat_map, std::ignore) =
          gen_compressed_submat_map(basis_map, task.bfn_screening.shell_list, nbf, nbf);

    // Evaluate Collocation (+ Grad and Hessian)
    if( func.is_mgga() ) {
      if ( needs_laplacian ) {
        // TODO: Modify gau2grid to compute Laplacian instead of full hessian
        lwd->eval_collocation_hessian( npts, nshells, nbe, points, basis, shell_list,
          basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval,
          d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval,
          d2basis_zz_eval);
        blas::lacpy( 'A', nbe, npts, d2basis_xx_eval, nbe, lbasis_eval, nbe );
        blas::axpy( nbe * npts, 1., d2basis_yy_eval, 1, lbasis_eval, 1);
        blas::axpy( nbe * npts, 1., d2basis_zz_eval, 1, lbasis_eval, 1);
      } else {
        lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
          basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
      }
    }
    // Evaluate Collocation (+ Grad)
    else if( func.is_gga() ) {
      if( is_dks ){
        lwd->eval_collocation_hessian( npts, nshells, nbe, points, basis, shell_list,
          basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval,
          d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval,
          d2basis_zz_eval);
      } else {
      lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
      }
    }
    else {
      if( is_dks ) {
        lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
      } else {
      lwd->eval_collocation( npts, nshells, nbe, points, basis, shell_list,
        basis_eval );
      }
    }

    // Evaluate X matrix (fac * P * B) -> store in Z
    const auto xmat_fac = is_rks ? 2.0 : 1.0; // TODO Fix for spinor RKS input
    lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, xmat_fac, Ps, ldps, basis_eval, nbe,
      zmat, nbe, nbe_scr );
    // X matrix for Pz
    if(not is_rks) {
      lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, 1.0, Pz, ldpz, basis_eval, nbe,
        zmat_z, nbe, nbe_scr);
    }
    if(not is_uks and not is_rks) {
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Py, ldpy, basis_eval, nbe,
        zmat_y, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Px, ldpx, basis_eval, nbe,
        zmat_x, nbe, nbe_scr);
    }
    if(is_dks) {
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Ps_SS, ldps_ss, dbasis_x_eval, nbe,
        zmat_s_ss, nbe, nbe_scr );
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Pz_SS, ldpz_ss, dbasis_x_eval, nbe,
        zmat_z_ss, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Py_SS, ldpy_ss, dbasis_x_eval, nbe,
        zmat_y_ss, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Px_SS, ldpx_ss, dbasis_x_eval, nbe,
        zmat_x_ss, nbe, nbe_scr);

      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Ps_SS, ldps_ss, dbasis_y_eval, nbe,
        xmat_y_s_ss, nbe, nbe_scr );
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Pz_SS, ldpz_ss, dbasis_y_eval, nbe,
        xmat_y_z_ss, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Py_SS, ldpy_ss, dbasis_y_eval, nbe,
        xmat_y_y_ss, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Px_SS, ldpx_ss, dbasis_y_eval, nbe,
        xmat_y_x_ss, nbe, nbe_scr);

      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Ps_SS, ldps_ss, dbasis_z_eval, nbe,
        xmat_z_s_ss, nbe, nbe_scr );
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Pz_SS, ldpz_ss, dbasis_z_eval, nbe,
        xmat_z_z_ss, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Py_SS, ldpy_ss, dbasis_z_eval, nbe,
        xmat_z_y_ss, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Px_SS, ldpx_ss, dbasis_z_eval, nbe,
        xmat_z_x_ss, nbe, nbe_scr);

       // Antisymmetric contributions
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Pz_SS_imag, ldpz_ss, dbasis_x_eval, nbe,
        immat_x_z, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Pz_SS_imag, ldpz_ss, dbasis_y_eval, nbe,
        immat_y_z, nbe, nbe_scr);

      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Py_SS_imag, ldpy_ss, dbasis_z_eval, nbe,
        immat_z_y, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Py_SS_imag, ldpy_ss, dbasis_x_eval, nbe,
        immat_x_y, nbe, nbe_scr);

      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Px_SS_imag, ldpx_ss, dbasis_y_eval, nbe,
        immat_y_x, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Px_SS_imag, ldpx_ss, dbasis_z_eval, nbe,
        immat_z_x, nbe, nbe_scr);

      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Ps_SS_imag, ldps_ss, dbasis_x_eval, nbe,
        immat_x_s, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Ps_SS_imag, ldps_ss, dbasis_y_eval, nbe,
        immat_y_s, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Ps_SS_imag, ldps_ss, dbasis_z_eval, nbe,
        immat_z_s, nbe, nbe_scr);

    }
    

    // Evaluate U and V variables
    if( func.is_mgga() ) {
      if (is_rks) {
        lwd->eval_uvvar_mgga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, zmat, nbe, mmat_x, mmat_y, mmat_z, 
          nbe, den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl);
      } else if (is_uks) {
        lwd->eval_uvvar_mgga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, zmat, nbe, zmat_z, nbe, 
          mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe, 
          den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl);
      }
    } else if ( func.is_gga() ) {
      if(is_rks) {
        lwd->eval_uvvar_gga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
          gamma );
      } else if(is_uks) {
        lwd->eval_uvvar_gga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, zmat_z, nbe, den_eval, dden_x_eval, 
          dden_y_eval, dden_z_eval, gamma );
      } else if(is_gks) {
        lwd->eval_uvvar_gga_gks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe, den_eval, dden_x_eval,
          dden_y_eval, dden_z_eval, gamma, K, H, gks_dtol );
      } else if(is_dks) {
        lwd->eval_uvvar_gga_dks( npts, nbe, basis_eval, 
          dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, 
          d2basis_xx_eval, d2basis_xy_eval, d2basis_xz_eval, 
          d2basis_yy_eval, d2basis_yz_eval, d2basis_zz_eval,
          zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe,
          zmat_s_ss, nbe, zmat_z_ss, nbe, zmat_x_ss, nbe, zmat_y_ss, nbe, 
          xmat_y_s_ss, xmat_y_z_ss, xmat_y_x_ss, xmat_y_y_ss,
          xmat_z_s_ss, xmat_z_z_ss, xmat_z_x_ss, xmat_z_y_ss,
          immat_x_z, immat_y_x, immat_z_y,
          immat_y_z, immat_z_x, immat_x_y,
          immat_x_s, immat_y_s, immat_z_s,
          den_eval, dden_x_eval, dden_y_eval, dden_z_eval, 
          gamma, K, H, gks_dtol );
      }

     } else {
      if(is_rks) {
      } else if(is_uks) {
        lwd->eval_uvvar_lda_uks( npts, nbe, basis_eval, zmat, nbe, zmat_z, nbe,
          den_eval );
      } else if(is_gks) {
        lwd->eval_uvvar_lda_gks( npts, nbe, basis_eval, zmat, nbe, zmat_z, nbe,
          zmat_x, nbe, zmat_y, nbe, den_eval, K, gks_dtol );
      } else if(is_dks){
        lwd->eval_uvvar_lda_dks ( npts, nbe, basis_eval, 
          dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, 
          zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe,
          zmat_s_ss, nbe, zmat_z_ss, nbe, zmat_x_ss, nbe, zmat_y_ss, nbe, 
          xmat_y_s_ss, xmat_y_z_ss, xmat_y_x_ss, xmat_y_y_ss,
          xmat_z_s_ss, xmat_z_z_ss, xmat_z_x_ss, xmat_z_y_ss,
          immat_x_z, immat_y_x, immat_z_y,
          immat_y_z, immat_z_x, immat_x_y,
          immat_x_s, immat_y_s, immat_z_s,
          den_eval, K, gks_dtol );
      }
     }
    
    // Evaluate XC functional
    if( func.is_mgga() )
      func.eval_exc_vxc( npts, den_eval, gamma, lapl, tau, eps, vrho, vgamma, vlapl, vtau);
    else if( func.is_gga() )
      func.eval_exc_vxc( npts, den_eval, gamma, eps, vrho, vgamma );
    else
      func.eval_exc_vxc( npts, den_eval, eps, vrho );
    // Factor weights into XC results
    for( int32_t i = 0; i < npts; ++i ) {
      eps[i]  *= weights[i];
      vrho[sds*i] *= weights[i];
      if(not is_rks) vrho[sds*i+1] *= weights[i];
    }
    if( func.is_gga() ){
      for( int32_t i = 0; i < npts; ++i ) {
         vgamma[gga_dim_scal*i] *= weights[i];
         if(not is_rks) {
           vgamma[gga_dim_scal*i+1] *= weights[i];
           vgamma[gga_dim_scal*i+2] *= weights[i];
         }
      }
    }
    if( func.is_mgga() ){
      for( int32_t i = 0; i < npts; ++i) {
        vtau[spin_dim_scal*i]  *= weights[i];
        vgamma[gga_dim_scal*i] *= weights[i];
        if(not is_rks) {
          vgamma[gga_dim_scal*i+1] *= weights[i];
          vgamma[gga_dim_scal*i+2] *= weights[i];
          vtau[spin_dim_scal*i+1]  *= weights[i];
        }

        // TODO: Add checks for Lapacian-dependent functionals
        if( needs_laplacian ) {
          vlapl[spin_dim_scal*i] *= weights[i];
          if(not is_rks) {
            vlapl[spin_dim_scal*i+1] *= weights[i];
          }
        }
      }
    }

    // Scalar integrations
    double NEL_local = 0.0;
    double spin_NEL_local = 0.0;
    double EXC_local  = 0.0;
    std::cout<< std::fixed << std::setprecision(10);

    for( int32_t i = 0; i < npts; ++i ) {
      const auto den = is_rks ? den_eval[i] : (den_eval[2*i] + den_eval[2*i+1]);
      const auto spin_den = is_rks ? den_eval[i] : (den_eval[2*i] - den_eval[2*i+1]);
      NEL_local += weights[i] * den;
      spin_NEL_local += weights[i] * spin_den;
      EXC_local += eps[i]     * den;
    }


    // Atomic updates
    #pragma omp atomic
    EXC_WORK += EXC_local;
    #pragma omp atomic
    NEL_WORK += NEL_local;
    #pragma omp atomic
    spin_NEL_WORK += spin_NEL_local;


    if(is_exc_only) continue;

    // Evaluate Z matrix for VXC
    if( func.is_mgga() ) {
      if(is_rks) {
        lwd->eval_zmat_mgga_vxc_rks( npts, nbe, vrho, vgamma, vlapl, basis_eval, dbasis_x_eval,
                                     dbasis_y_eval, dbasis_z_eval, lbasis_eval,
                                     dden_x_eval, dden_y_eval, dden_z_eval, zmat, nbe);
        lwd->eval_mmat_mgga_vxc_rks( npts, nbe, vtau, vlapl, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
                                     mmat_x, mmat_y, mmat_z, nbe);
      } else if (is_uks) {
        lwd->eval_zmat_mgga_vxc_uks( npts, nbe, vrho, vgamma, vlapl, basis_eval, dbasis_x_eval,
                                     dbasis_y_eval, dbasis_z_eval, lbasis_eval,
                                     dden_x_eval, dden_y_eval, dden_z_eval, zmat, nbe, zmat_z, nbe);
        lwd->eval_mmat_mgga_vxc_uks( npts, nbe, vtau, vlapl, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
                                     mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe);
      }
    }
    else if( func.is_gga() ) {
      if(is_rks) {
        lwd->eval_zmat_gga_vxc_rks( npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval,
                                dden_z_eval, zmat, nbe);
      } else if(is_uks) {
        lwd->eval_zmat_gga_vxc_uks( npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval,
                                dden_z_eval, zmat, nbe, zmat_z, nbe);
      } else if(is_gks) {
        lwd->eval_zmat_gga_vxc_gks( npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval,
                                dden_z_eval, zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe,
                                K, H);
      } else if(is_dks) {
        lwd->eval_zmat_gga_vxc_dks( npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, 
                                d2basis_xx_eval, d2basis_xy_eval, d2basis_xz_eval, 
                                d2basis_yy_eval, d2basis_yz_eval, d2basis_zz_eval,
                                dden_x_eval, dden_y_eval, dden_z_eval, 
                                zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe,
                                zmat_s_ss, nbe, zmat_z_ss, nbe, zmat_x_ss, nbe, zmat_y_ss, nbe,
                                xmat_y_s_ss, xmat_y_z_ss, xmat_y_x_ss, xmat_y_y_ss,
                                xmat_z_s_ss, xmat_z_z_ss, xmat_z_x_ss, xmat_z_y_ss,
                                K, H);
      }
       
    } else {
      if(is_rks) {
        lwd->eval_zmat_lda_vxc_rks( npts, nbe, vrho, basis_eval, zmat, nbe );
      } else if(is_uks) {
        lwd->eval_zmat_lda_vxc_uks( npts, nbe, vrho, basis_eval, zmat, nbe, zmat_z, nbe );
      } else if(is_gks) {
        lwd->eval_zmat_lda_vxc_gks( npts, nbe, vrho, basis_eval, zmat, nbe, zmat_z, nbe, 
                                    zmat_x, nbe, zmat_y, nbe, K);
      } else if(is_dks) {
        lwd->eval_zmat_lda_vxc_dks( npts, nbe, vrho, basis_eval, 
          dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, 
          zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe, 
          zmat_s_ss, nbe, zmat_z_ss, nbe, zmat_x_ss, nbe, zmat_y_ss, nbe,
          xmat_y_s_ss, xmat_y_z_ss, xmat_y_x_ss, xmat_y_y_ss,
          xmat_z_s_ss, xmat_z_z_ss, xmat_z_x_ss, xmat_z_y_ss,
          K);
      }
    }
    

    // Increment LT of VXC
    {

      // Increment VXC
      lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat, nbe, VXCs, ldvxcs, nbe_scr );
      if(not is_rks) {
        lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat_z, nbe,VXCz, ldvxcz, nbe_scr );
      }
      if(not is_uks and not is_rks) {
        lwd->inc_vxc( npts, nbf, nbe, basis_eval, submat_map, zmat_y, nbe, VXCy, ldvxcy,
          nbe_scr );
        lwd->inc_vxc( npts, nbf, nbe, basis_eval, submat_map, zmat_x, nbe, VXCx, ldvxcx,
          nbe_scr );
      }
      if(is_dks) {
        // VXC SS BEGIN

        // Vxc s
        //// xx + yy + zz
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, zmat_s_ss, nbe, VXCs_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_y_s_ss, nbe, VXCs_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_z_s_ss, nbe, VXCs_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );

        /// Vxc s im
        //// anti Zz xy-yx
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_y_z_ss, nbe, VXCs_SS_im, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_y_eval, submat_map, zmat_z_ss, nbe, VXCs_SS_im, ldvxcs_ss,
          nbe_scr, -1*RKB_factor );

        //// anti Zx yz-zy
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_z_x_ss, nbe, VXCs_SS_im, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_y_x_ss, nbe, VXCs_SS_im, ldvxcs_ss,
          nbe_scr, -1*RKB_factor );

        //// anti Zy zx-xz
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_z_eval, submat_map, zmat_y_ss, nbe, VXCs_SS_im, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_z_y_ss, nbe, VXCs_SS_im, ldvxcs_ss,
          nbe_scr, -1*RKB_factor );
        // Vxc s end


        // Vxc z
        //// zz - xx - yy
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, zmat_z_ss, nbe, VXCz_SS, ldvxcz_ss,
          nbe_scr, -1*RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_y_z_ss, nbe, VXCz_SS, ldvxcz_ss,
          nbe_scr, -1*RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_z_z_ss, nbe, VXCz_SS, ldvxcz_ss,
          nbe_scr, RKB_factor );

        //// zx + xz
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_z_x_ss, nbe, VXCz_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, zmat_x_ss, nbe, VXCz_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );

        //// zy + yz
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_z_y_ss, nbe, VXCz_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_y_y_ss, nbe, VXCz_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );

        /// Vxc z im
        //// anti Zs xy-yx
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_y_eval, submat_map, zmat_s_ss, nbe, VXCz_SS_im, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_y_s_ss, nbe, VXCz_SS_im, ldvxcs_ss,
          nbe_scr, -1*RKB_factor );
        // Vxc z end


        // Vxc x
        //// xx - yy - zz
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, zmat_x_ss, nbe, VXCx_SS, ldvxcz_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_y_x_ss, nbe, VXCx_SS, ldvxcz_ss,
          nbe_scr, -1*RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_z_x_ss, nbe, VXCx_SS, ldvxcz_ss,
          nbe_scr, -1*RKB_factor );

          // xz + zx
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_z_z_ss, nbe, VXCx_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, zmat_z_ss, nbe, VXCx_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );

          // xy + yx
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, zmat_y_ss, nbe, VXCx_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_y_y_ss, nbe, VXCx_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );

        /// Vxc x im anti
        //// anti Zs yz-zy
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_y_s_ss, nbe, VXCx_SS_im, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_z_s_ss, nbe, VXCx_SS_im, ldvxcs_ss,
          nbe_scr, -1*RKB_factor );
        // Vxc x end


        // Vxc y
        //// yy - xx - zz
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, zmat_y_ss, nbe, VXCy_SS, ldvxcy_ss,
          nbe_scr, -1*RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_y_y_ss, nbe, VXCy_SS, ldvxcy_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_z_y_ss, nbe, VXCy_SS, ldvxcy_ss,
          nbe_scr, -1*RKB_factor );

        //// yz + zy
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, xmat_z_z_ss, nbe, VXCy_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_z_eval, submat_map, xmat_y_z_ss, nbe, VXCy_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );

        //// xy + yx
        lwd->inc_vxc( npts, nbf, nbe, dbasis_y_eval, submat_map, zmat_x_ss, nbe, VXCy_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_y_x_ss, nbe, VXCy_SS, ldvxcs_ss,
          nbe_scr, RKB_factor );

        // Vxc y im anti
        //// anti Zs zx-xz
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_x_eval, submat_map, xmat_z_s_ss, nbe, VXCy_SS_im, ldvxcs_ss,
          nbe_scr, RKB_factor );
        lwd->inc_vxc_anti( npts, nbf, nbe, dbasis_z_eval, submat_map, zmat_s_ss, nbe, VXCy_SS_im, ldvxcs_ss,
          nbe_scr, -1*RKB_factor );
        // Vxc y end

      } // VXC SS END
       
    }

  } // Loop over tasks

  } // End OpenMP region


  // Set scalar return values
  *EXC  = EXC_WORK;
  *N_EL = NEL_WORK;
  *spin_N_EL = spin_NEL_WORK;

  // Prints integrated charge density (N_EL) and integrated mnorm (spin_N_EL)
  // std::cout<<"N_EL =  "<<*N_EL<<std::endl;
  // std::cout<<"spin N_EL =  "<<*spin_N_EL<<std::endl;
  // std::cout<<"EXC =  "<<*EXC<<std::endl;



  if(not is_exc_only) {
    // Symmetrize VXC
    for( int32_t j = 0;   j < nbf; ++j ) {
      for( int32_t i = j+1; i < nbf; ++i ) {
        VXCs[ j + i*ldvxcs ] = VXCs[ i + j*ldvxcs ];
      }
    }
    if(not is_rks) {
      for( int32_t j = 0;   j < nbf; ++j ) {
        for( int32_t i = j+1; i < nbf; ++i ) {
          VXCz[ j + i*ldvxcz ] = VXCz[ i + j*ldvxcz ];
        }
      }
    }
    if(not is_uks and not is_rks) {
      for( int32_t j = 0;   j < nbf; ++j ) {
        for( int32_t i = j+1; i < nbf; ++i ) {
          VXCy[ j + i*ldvxcy ] = VXCy[ i + j*ldvxcy ];
          VXCx[ j + i*ldvxcx ] = VXCx[ i + j*ldvxcx ];
        }
      }
    }
    if(is_dks) {
      for( int32_t j = 0;   j < nbf; ++j ) {
        for( int32_t i = j+1; i < nbf; ++i ) {
          VXCs_SS[ j + i*ldvxcs_ss ] = VXCs_SS[ i + j*ldvxcs_ss ];
          VXCz_SS[ j + i*ldvxcz_ss ] = VXCz_SS[ i + j*ldvxcz_ss ];
          VXCy_SS[ j + i*ldvxcy_ss ] = VXCy_SS[ i + j*ldvxcy_ss ];
          VXCx_SS[ j + i*ldvxcx_ss ] = VXCx_SS[ i + j*ldvxcx_ss ];

          // Anti Symmetrize IM[VXC]
          VXCs_SS_im[ j + i*ldvxcs_ss ] = -1. * VXCs_SS_im[ i + j*ldvxcs_ss ];
          VXCz_SS_im[ j + i*ldvxcz_ss ] = -1. * VXCz_SS_im[ i + j*ldvxcz_ss ];
          VXCy_SS_im[ j + i*ldvxcy_ss ] = -1. * VXCy_SS_im[ i + j*ldvxcy_ss ];
          VXCx_SS_im[ j + i*ldvxcx_ss ] = -1. * VXCx_SS_im[ i + j*ldvxcx_ss ];
        }
      }
    }
  }
} 



/// RKS EXC/VXC driver - delegates to generic DKS impl
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* P, int64_t ldp,
                 value_type* VXC, int64_t ldvxc,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_vxc_(m, n, P, ldp, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, nullptr, nullptr, nullptr, 
    VXC, ldvxc, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    EXC, ks_settings );

}


/// UKS EXC/VXC driver - delegates to generic DKS impl
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz,
                 value_type* VXCs, int64_t ldvxcs,
                 value_type* VXCz, int64_t ldvxcz,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_vxc_(m, n, Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, nullptr, nullptr, nullptr, 
    VXCs, ldvxcs, VXCz, ldvxcz, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    EXC, ks_settings );

}

/// GKS EXC/VXC Driver = delegates to generic DKS impl
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz,
                 const value_type* Py, int64_t ldpy,
                 const value_type* Px, int64_t ldpx,
                 value_type* VXCs, int64_t ldvxcs,
                 value_type* VXCz, int64_t ldvxcz,
                 value_type* VXCy, int64_t ldvxcy,
                 value_type* VXCx, int64_t ldvxcx,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings ) {

  eval_exc_vxc_(m, n, Ps, ldps, Pz, ldpz, Py, ldpy, Px, ldpx,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, nullptr, nullptr, nullptr, 
    VXCs, ldvxcs, VXCz, ldvxcz, VXCy, ldvxcy, VXCx, ldvxcx,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    EXC, ks_settings );
}

} // namespace GauXC::detail
