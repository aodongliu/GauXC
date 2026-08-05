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

#include "host/reference_local_host_work_driver.hpp"
#include "host/reference/weights.hpp"
#include "host/reference/collocation.hpp"

#include "host/util.hpp"
#include "host/blas.hpp"
#include <stdexcept>

#include <gauxc/basisset_map.hpp>
#include <gauxc/shell_pair.hpp>
#include <gauxc/util/unused.hpp>
#include "cpu/integral_data_types.hpp"
#include "cpu/obara_saika_integrals.hpp"
#include "cpu/chebyshev_boys_computation.hpp"
#include <gauxc/util/real_solid_harmonics.hpp>
#include "integrator_util/integral_bounds.hpp"

#include <gauxc/physcon.hpp>

namespace GauXC {

  ReferenceLocalHostWorkDriver::ReferenceLocalHostWorkDriver() {
    this->boys_table = XCPU::boys_init();
  }
  
  ReferenceLocalHostWorkDriver::~ReferenceLocalHostWorkDriver() noexcept {
    XCPU::boys_finalize(this->boys_table);
  }

  // Partition weights
  void ReferenceLocalHostWorkDriver::partition_weights( XCWeightAlg weight_alg, 
							const Molecule& mol, const MolMeta& meta, task_iterator task_begin, 
							task_iterator task_end ) {
    switch( weight_alg ) {
      case XCWeightAlg::Becke:
        reference_becke_weights_host( mol, meta, task_begin, task_end );
        break;
      case XCWeightAlg::SSF:
        reference_ssf_weights_host( mol, meta, task_begin, task_end );
        break;
      case XCWeightAlg::LKO:
        reference_lko_weights_host( mol, meta, task_begin, task_end );
        break;
      default:
        GAUXC_GENERIC_EXCEPTION("Weight Alg Not Supported");
    }
  }

  void ReferenceLocalHostWorkDriver::eval_weight_1st_deriv_contracted( 
    XCWeightAlg weight_alg, const Molecule& mol, const MolMeta& meta, 
    const XCTask& task, const double* w_times_f, double* exc_grad_w ) {
    switch( weight_alg ) {
      case XCWeightAlg::Becke:
        reference_becke_weights_1std_contraction_host( mol, meta, task, w_times_f, exc_grad_w );
        break;
      case XCWeightAlg::SSF:
        reference_ssf_weights_1std_contraction_host( mol, meta, task, w_times_f, exc_grad_w );
        break;
      default:
        GAUXC_GENERIC_EXCEPTION("Weight Alg Not Supported");
    }
  }


  // Collocation
  void ReferenceLocalHostWorkDriver::eval_collocation( size_t npts, size_t nshells, 
						       size_t nbe, const double* pts, const BasisSet<double>& basis, 
						       const int32_t* shell_list, double* basis_eval ) {
    gau2grid_collocation( npts, nshells, nbe, pts, basis, shell_list, basis_eval );
  }


  // Collocation Gradient
  void ReferenceLocalHostWorkDriver::eval_collocation_gradient( size_t npts, 
								size_t nshells, size_t nbe, const double* pts, const BasisSet<double>& basis, 
								const int32_t* shell_list, double* basis_eval, double* dbasis_x_eval, 
								double* dbasis_y_eval, double* dbasis_z_eval) {
    gau2grid_collocation_gradient(npts, nshells, nbe, pts, basis, shell_list,
				  basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
  }

  void ReferenceLocalHostWorkDriver::eval_collocation_hessian( size_t npts, 
							       size_t nshells, size_t nbe, const double* pts, const BasisSet<double>& basis, 
							       const int32_t* shell_list, double* basis_eval, double* dbasis_x_eval, 
							       double* dbasis_y_eval, double* dbasis_z_eval, double* d2basis_xx_eval, 
							       double* d2basis_xy_eval, double* d2basis_xz_eval, double* d2basis_yy_eval, 
							       double* d2basis_yz_eval, double* d2basis_zz_eval ) {
    gau2grid_collocation_hessian(npts, nshells, nbe, pts, basis, shell_list,
				 basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval,
				 d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval,
				 d2basis_zz_eval);
  }

  void ReferenceLocalHostWorkDriver::eval_collocation_der3( size_t npts,
							    size_t nshells, size_t nbe, const double* pts, const BasisSet<double>& basis, 
							     const int32_t* shell_list, double* basis_eval, double* dbasis_x_eval, 
							     double* dbasis_y_eval, double* dbasis_z_eval, double* d2basis_xx_eval, 
							     double* d2basis_xy_eval, double* d2basis_xz_eval, double* d2basis_yy_eval, 
							     double* d2basis_yz_eval, double* d2basis_zz_eval, double* d3basis_xxx_eval,
							     double* d3basis_xxy_eval, double* d3basis_xxz_eval, double* d3basis_xyy_eval,
							     double* d3basis_xyz_eval, double* d3basis_xzz_eval, double* d3basis_yyy_eval,
							     double* d3basis_yyz_eval, double* d3basis_yzz_eval, double* d3basis_zzz_eval) {
    gau2grid_collocation_der3(npts, nshells, nbe, pts, basis, shell_list,
				 basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval,
				 d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval,
				 d2basis_zz_eval, d3basis_xxx_eval, d3basis_xxy_eval, d3basis_xxz_eval,
				 d3basis_xyy_eval, d3basis_xyz_eval, d3basis_xzz_eval, d3basis_yyy_eval,
				 d3basis_yyz_eval, d3basis_yzz_eval, d3basis_zzz_eval);
  }


  // X matrix (P * B)
  void ReferenceLocalHostWorkDriver::eval_xmat( size_t npts, size_t nbf, size_t nbe, 
						const submat_map_t& submat_map, double fac, const double* P, size_t ldp, 
						const double* basis_eval, size_t ldb, double* X, size_t ldx, double* scr ) {
    const auto inbe  = static_cast<int32_t>(nbe);
    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildb  = static_cast<int32_t>(ldb);
    const auto ildx  = static_cast<int32_t>(ldx);
    const auto ildp  = static_cast<int32_t>(ldp);
    const auto* P_use = P;
    size_t ldp_use = ldp;

    if( submat_map.size() > 1 ) {
      detail::submat_set( inbf, inbf, inbe, inbe, P, ildp, scr, inbe, submat_map );
      P_use = scr;
      ldp_use = nbe;
    } else if( nbe != nbf ) {
      P_use = P + submat_map[0][0]*(ldp+1);
    }

    blas::gemm( 'N', 'N', inbe, inpts, inbe, fac, P_use, static_cast<int32_t>(ldp_use), basis_eval, ildb,
		0., X, ildx );

  }


  // U/VVar LDA (density)
  void ReferenceLocalHostWorkDriver::eval_uvvar_lda_rks( size_t npts, size_t nbe,
						     const double* basis_eval, const double* X, size_t ldx, double* den_eval) {

    const auto inbe = static_cast<int32_t>(nbe);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioff = size_t(i) * ldx;
      const auto*   X_i = X + ioff;
      den_eval[i] = blas::dot( inbe, basis_eval + ioff, 1, X_i, 1 );

    }    

  }

  
  void ReferenceLocalHostWorkDriver::eval_uvvar_lda_uks( size_t npts, size_t nbe,
   const double* basis_eval, const double* Xs, size_t ldxs,
   const double* Xz, size_t ldxz, double* den_eval) {

    const auto inbe = static_cast<int32_t>(nbe);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioffs = size_t(i) * ldxs;
      const size_t ioffz = size_t(i) * ldxz;

      const auto*   Xs_i = Xs + ioffs;
      const auto*   Xz_i = Xz + ioffz;

      const double rhos = blas::dot( inbe, basis_eval + ioffs, 1, Xs_i, 1 );
      const double rhoz = blas::dot( inbe, basis_eval + ioffz, 1, Xz_i, 1 );
      
      den_eval[2*i]   = 0.5*(rhos + rhoz); // rho_+
      den_eval[2*i+1] = 0.5*(rhos - rhoz); // rho_-

    }
 
  }
  
  void ReferenceLocalHostWorkDriver::eval_uvvar_lda_gks( size_t npts, size_t nbe, const double* basis_eval,
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz,
    const double* Xx, size_t ldxx, const double* Xy, size_t ldxy, double* den_eval, double* K, const double dtol) {

    const auto inbe = static_cast<int32_t>(nbe);

    auto *KZ = K; // KZ // store K in the Z matrix
    auto *KY = KZ + npts;
    auto *KX = KY + npts;

    double dtolsq = dtol*dtol;

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioffs = size_t(i) * ldxs;
      const size_t ioffz = size_t(i) * ldxz;
      const size_t ioffx = size_t(i) * ldxx;
      const size_t ioffy = size_t(i) * ldxy;

      const auto*   Xs_i = Xs + ioffs;
      const auto*   Xz_i = Xz + ioffz;
      const auto*   Xx_i = Xx + ioffx;
      const auto*   Xy_i = Xy + ioffy;

      const double rhos = blas::dot( inbe, basis_eval + ioffs, 1, Xs_i, 1 );
      const double rhoz = blas::dot( inbe, basis_eval + ioffz, 1, Xz_i, 1 );
      const double rhox = blas::dot( inbe, basis_eval + ioffx, 1, Xx_i, 1 );
      const double rhoy = blas::dot( inbe, basis_eval + ioffy, 1, Xy_i, 1 );
 
      double mtemp = rhoz * rhoz + rhox * rhox + rhoy * rhoy;
      double mnorm = 0;

      if (mtemp > dtolsq) {
        mnorm = sqrt(mtemp);
        KZ[i] = rhoz / mnorm;
        KY[i] = rhoy / mnorm;
        KX[i] = rhox / mnorm;
      } else {
        mnorm = (1. / 3.) * (rhox + rhoy + rhoz);
        KZ[i] = 1. / 3.;
        KY[i] = 1. / 3.;
        KX[i] = 1. / 3.;
      }

      den_eval[2*i]   = 0.5*(rhos + mnorm); // rho_+
      den_eval[2*i+1] = 0.5*(rhos - mnorm); // rho_-

    }

  }

  void ReferenceLocalHostWorkDriver::eval_uvvar_lda_dks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz, 
    const double* Xx, size_t ldxx, const double* Xy, size_t ldxy,  
    const double* Xs_x_SS, size_t ldxs_ss, const double* Xz_x_SS, size_t ldxz_ss, 
    const double* Xx_x_SS, size_t ldxx_ss, const double* Xy_x_SS, size_t ldxy_ss, 
    const double* Xs_y_SS, const double* Xz_y_SS, 
    const double* Xx_y_SS, const double* Xy_y_SS, 
    const double* Xs_z_SS, const double* Xz_z_SS, 
    const double* Xx_z_SS, const double* Xy_z_SS, 
    const double* immat_x_z, const double* immat_y_x, const double* immat_z_y,
    const double* immat_y_z, const double* immat_z_x, const double* immat_x_y,
    const double* immat_x_s, const double* immat_y_s, const double* immat_z_s,
    double* den_eval, double* K, const double dtol ) {

    auto *KZ = K; // KZ // store K in the Z matrix
    auto *KY = KZ + npts;
    auto *KX = KY + npts;

    double dtolsq = dtol*dtol;

  
    auto flip = -1.0;

 
    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioffs = size_t(i) * ldxs;
      const size_t ioffz = size_t(i) * ldxz;
      const size_t ioffx = size_t(i) * ldxx;
      const size_t ioffy = size_t(i) * ldxy;

      const auto*   Xs_i = Xs + ioffs;
      const auto*   Xz_i = Xz + ioffz;
      const auto*   Xx_i = Xx + ioffx;
      const auto*   Xy_i = Xy + ioffy;

      const auto*   Xs_x_i_ss = Xs_x_SS + ioffs;
      const auto*   Xz_x_i_ss = Xz_x_SS + ioffz;
      const auto*   Xx_x_i_ss = Xx_x_SS + ioffx;
      const auto*   Xy_x_i_ss = Xy_x_SS + ioffy;

      const auto*   Xs_y_i_ss = Xs_y_SS + ioffs;
      const auto*   Xz_y_i_ss = Xz_y_SS + ioffz;
      const auto*   Xx_y_i_ss = Xx_y_SS + ioffx;
      const auto*   Xy_y_i_ss = Xy_y_SS + ioffy;

      const auto*   Xs_z_i_ss = Xs_z_SS + ioffs; // Re[Ps] * d/dz chi nu
      const auto*   Xz_z_i_ss = Xz_z_SS + ioffz;
      const auto*   Xx_z_i_ss = Xx_z_SS + ioffx;
      const auto*   Xy_z_i_ss = Xy_z_SS + ioffy;

      const auto*   Xz_x_i_im = immat_x_z + ioffz; // im[Pz] * d/dx chi nu
      const auto*   Xz_y_i_im = immat_y_z + ioffz;

      const auto*   Xx_y_i_im = immat_y_x + ioffx;
      const auto*   Xx_z_i_im = immat_z_x + ioffx;

      const auto*   Xy_z_i_im = immat_z_y + ioffy;
      const auto*   Xy_x_i_im = immat_x_y + ioffy;

      const auto*   Xs_x_i_im = immat_x_s + ioffs;
      const auto*   Xs_y_i_im = immat_y_s + ioffs;
      const auto*   Xs_z_i_im = immat_z_s + ioffs;

      double rhos = blas::dot( nbe, basis_eval + ioffs, 1, Xs_i, 1 );
      double rhoz = blas::dot( nbe, basis_eval + ioffz, 1, Xz_i, 1 );
      double rhox = blas::dot( nbe, basis_eval + ioffx, 1, Xx_i, 1 );
      double rhoy = blas::dot( nbe, basis_eval + ioffy, 1, Xy_i, 1 );

      // rho SS
      // s
      const double rhos_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xs_y_i_ss, 1 );
      const double rhos_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xs_x_i_ss, 1 );
      const double rhos_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xs_z_i_ss, 1 );

      // s anti
      const double rhos_z_yx = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xz_x_i_im, 1 );
      const double rhos_z_xy = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xz_y_i_im, 1 );

      const double rhos_x_yz = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xx_y_i_im, 1 );
      const double rhos_x_zy = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xx_z_i_im, 1 );

      const double rhos_y_zx = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xy_z_i_im, 1 );
      const double rhos_y_xz = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xy_x_i_im, 1 );

      auto rhos_ss_dot = rhos_yy_ss + rhos_xx_ss + rhos_zz_ss; 
      auto rhos_anti = rhos_z_xy - rhos_z_yx + rhos_x_yz - rhos_x_zy + rhos_y_zx - rhos_y_xz;
      auto rhos_ss = rhos_ss_dot - rhos_anti;

      // z
      const double rhoz_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xz_x_i_ss, 1 );
      const double rhoz_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xz_y_i_ss, 1 );       
      const double rhoz_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xz_z_i_ss, 1 );

      const double rhoz_xz_ss = blas::dot( nbe, dbasis_x_eval + ioffx, 1, Xx_z_i_ss, 1 );
      const double rhoz_zx_ss = blas::dot( nbe, dbasis_z_eval + ioffx, 1, Xx_x_i_ss, 1 );

      const double rhoz_yz_ss = blas::dot( nbe, dbasis_y_eval + ioffx, 1, Xy_z_i_ss, 1 );
      const double rhoz_zy_ss = blas::dot( nbe, dbasis_z_eval + ioffx, 1, Xy_y_i_ss, 1 );

      // z anti
      const double rhoz_s_yx = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double rhoz_s_xy = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xs_y_i_im, 1 );

      const auto rhoz_ss_dot = rhoz_zz_ss - rhoz_yy_ss - rhoz_xx_ss;
      const auto rhoz_ss_cross = rhoz_xz_ss + rhoz_zx_ss + rhoz_yz_ss + rhoz_zy_ss; 
      const auto rhoz_ss_anti = rhoz_s_xy - rhoz_s_yx;
      const auto rhoz_ss = rhoz_ss_dot + rhoz_ss_anti + rhoz_ss_cross;

      // x
      const double rhox_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xx_x_i_ss, 1 );
      const double rhox_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xx_y_i_ss, 1 );       
      const double rhox_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xx_z_i_ss, 1 );

      const double rhox_xz_ss = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xz_z_i_ss, 1 );
      const double rhox_zx_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xz_x_i_ss, 1 );

      const double rhox_yx_ss = blas::dot( nbe, dbasis_x_eval + ioffy, 1, Xy_y_i_ss, 1 );
      const double rhox_xy_ss = blas::dot( nbe, dbasis_y_eval + ioffy, 1, Xy_x_i_ss, 1 );

      const auto rhox_ss_dot = rhox_xx_ss - rhox_zz_ss - rhox_yy_ss;
      const auto rhox_ss_cross = rhox_xz_ss +rhox_zx_ss + rhox_yx_ss + rhox_xy_ss;

      // x anti
      const double rhox_zy_im = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xs_y_i_im, 1 );
      const double rhox_yz_im = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xs_z_i_im, 1 );   
      
      const auto rhox_ss_anti = rhox_yz_im - rhox_zy_im;
      const auto rhox_ss = rhox_ss_dot + rhox_ss_anti + rhox_ss_cross;

      // y
      const double rhoy_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xy_x_i_ss, 1 );
      const double rhoy_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xy_y_i_ss, 1 );       
      const double rhoy_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xy_z_i_ss, 1 );

      const double rhoy_zy_ss = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xz_z_i_ss, 1 );
      const double rhoy_yz_ss = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xz_y_i_ss, 1 );

      const double rhoy_yx_ss = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xx_y_i_ss, 1 );
      const double rhoy_xy_ss = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xx_x_i_ss, 1 );

      const auto rhoy_ss_dot =  rhoy_yy_ss - rhoy_xx_ss - rhoy_zz_ss;
      const auto rhoy_ss_cross = rhoy_xy_ss + rhoy_yx_ss + rhoy_yz_ss + rhoy_zy_ss; 

      // y anti

      const double rhoy_zx_im = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double rhoy_xz_im = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xs_z_i_im, 1 );   
    
      const auto rhoy_ss_anti = rhoy_zx_im - rhoy_xz_im;
      const auto rhoy_ss = rhoy_ss_dot + rhoy_ss_anti + rhoy_ss_cross;

      rhos += RKB_factor * rhos_ss;
      rhoz += RKB_factor * rhoz_ss; 
      rhox += RKB_factor * rhox_ss;
      rhoy += RKB_factor * rhoy_ss;

      double mtemp = rhoz * rhoz + rhox * rhox + rhoy * rhoy;
      double mnorm = 0;

      if (mtemp > dtolsq) {
        mnorm = sqrt(mtemp);
        KZ[i] = rhoz / mnorm;
        KY[i] = rhoy / mnorm;
        KX[i] = rhox / mnorm;
      } else {
        mnorm = (1. / 3.) * (rhox + rhoy + rhoz);
        KZ[i] = 1. / 3.;
        KY[i] = 1. / 3.;
        KX[i] = 1. / 3.;
      }

      den_eval[2*i]   = 0.5*(rhos + mnorm); // rho_+
      den_eval[2*i+1] = 0.5*(rhos - mnorm); // rho_-

    }

  }



  void ReferenceLocalHostWorkDriver::eval_uvvar_gga_rks( size_t npts, size_t nbe,
						     const double* basis_eval, const double* dbasis_x_eval,
						     const double *dbasis_y_eval, const double* dbasis_z_eval, const double* X,
						     size_t ldx, double* den_eval, double* dden_x_eval, double* dden_y_eval,
						     double* dden_z_eval, double* gamma ) {

    const auto inbe = static_cast<int32_t>(nbe);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioff = size_t(i) * ldx;
      const auto*   X_i = X + ioff;

      den_eval[i] = blas::dot( inbe, basis_eval + ioff, 1, X_i, 1 );

      const auto dx = 2. * blas::dot( inbe, dbasis_x_eval + ioff, 1, X_i, 1 );
      const auto dy = 2. * blas::dot( inbe, dbasis_y_eval + ioff, 1, X_i, 1 );
      const auto dz = 2. * blas::dot( inbe, dbasis_z_eval + ioff, 1, X_i, 1 );

      dden_x_eval[i] = dx;
      dden_y_eval[i] = dy;
      dden_z_eval[i] = dz;

      gamma[i] = dx*dx + dy*dy + dz*dz;

    }
  }

void ReferenceLocalHostWorkDriver::eval_uvvar_gga_uks( size_t npts, size_t nbe,
  const double* basis_eval, const double* dbasis_x_eval,
  const double *dbasis_y_eval, const double* dbasis_z_eval, const double* Xs,
  size_t ldxs, const double* Xz, size_t ldxz,
  double* den_eval, double* dden_x_eval, double* dden_y_eval,
  double* dden_z_eval, double* gamma ) {

   const auto inbe = static_cast<int32_t>(nbe);

   for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioffs = size_t(i) * ldxs;
      const size_t ioffz = size_t(i) * ldxz;

      const auto*   Xs_i = Xs + ioffs;
      const auto*   Xz_i = Xz + ioffz;

      double rhos = blas::dot( inbe, basis_eval + ioffs, 1, Xs_i, 1 ); // S density
      double rhoz = blas::dot( inbe, basis_eval + ioffz, 1, Xz_i, 1 ); // Z density


      den_eval[2*i]   = 0.5*(rhos + rhoz); // rho_+
      den_eval[2*i+1] = 0.5*(rhos - rhoz); // rho_-

      const auto dndx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffs, 1, Xs_i, 1 );
      const auto dndy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffs, 1, Xs_i, 1 );
      const auto dndz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffs, 1, Xs_i, 1 );

      const auto dMzdx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffz, 1, Xz_i, 1 );
      const auto dMzdy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffz, 1, Xz_i, 1 );
      const auto dMzdz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffz, 1, Xz_i, 1 );

      dden_x_eval[2*i] = dndx; // dn / dx
      dden_y_eval[2*i] = dndy; // dn / dy
      dden_z_eval[2*i] = dndz; // dn / dz

      dden_x_eval[2*i+1] = dMzdx; // dMz / dx
      dden_y_eval[2*i+1] = dMzdy; // dMz / dy
      dden_z_eval[2*i+1] = dMzdz; // dMz / dz

      // (del n).(del n)
      const auto dn_sq  = dndx*dndx + dndy*dndy + dndz*dndz;
      // (del Mz).(del Mz)
      const auto dMz_sq = dMzdx*dMzdx + dMzdy*dMzdy + dMzdz*dMzdz;
      // (del n).(del Mz)
      const auto dn_dMz = dndx*dMzdx + dndy*dMzdy + dndz*dMzdz;

      gamma[3*i  ] = 0.25*(dn_sq + dMz_sq) + 0.5*dn_dMz;
      gamma[3*i+1] = 0.25*(dn_sq - dMz_sq);
      gamma[3*i+2] = 0.25*(dn_sq + dMz_sq) - 0.5*dn_dMz;
    }

}


void ReferenceLocalHostWorkDriver::eval_uvvar_mgga_rks( size_t npts, size_t nbe,
  const double* basis_eval, const double* dbasis_x_eval,
  const double *dbasis_y_eval, const double* dbasis_z_eval, const double* lbasis_eval,
  const double* X, size_t ldx, const double* mmat_x, const double* mmat_y,
  const double* mmat_z, [[maybe_unused]] size_t ldm,
  double* den_eval, double* dden_x_eval, double* dden_y_eval,
  double* dden_z_eval, double* gamma, double* tau, double* lapl ) {

   const auto inbe = static_cast<int32_t>(nbe);

   for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioff = size_t(i) * ldx;
      const auto*   X_i = X + ioff;

      den_eval[i] = blas::dot( inbe, basis_eval + ioff, 1, X_i, 1 );

      const auto dx = 2. * blas::dot( inbe, dbasis_x_eval + ioff, 1, X_i, 1 );
      const auto dy = 2. * blas::dot( inbe, dbasis_y_eval + ioff, 1, X_i, 1 );
      const auto dz = 2. * blas::dot( inbe, dbasis_z_eval + ioff, 1, X_i, 1 );

      dden_x_eval[i] = dx;
      dden_y_eval[i] = dy;
      dden_z_eval[i] = dz;

      gamma[i] = dx*dx + dy*dy + dz*dz;

      tau[i]  = 0.5*blas::dot( inbe, dbasis_x_eval + ioff, 1, mmat_x + ioff, 1);
      tau[i] += 0.5*blas::dot( inbe, dbasis_y_eval + ioff, 1, mmat_y + ioff, 1);
      tau[i] += 0.5*blas::dot( inbe, dbasis_z_eval + ioff, 1, mmat_z + ioff, 1);

      if (lapl != nullptr)
        lapl[i]  = 2. * blas::dot( inbe, lbasis_eval + ioff, 1, X_i, 1) + 4. * tau[i];

   }
}

void ReferenceLocalHostWorkDriver::eval_uvvar_mgga_uks( size_t npts, size_t nbe,
  const double* basis_eval, const double* dbasis_x_eval,
  const double *dbasis_y_eval, const double* dbasis_z_eval, const double* lbasis_eval,
  const double* Xs, size_t ldxs, const double* Xz, size_t ldxz,
  const double* mmat_xs, const double* mmat_ys, const double* mmat_zs, [[maybe_unused]] size_t ldms,
  const double* mmat_xz, const double* mmat_yz, const double* mmat_zz, [[maybe_unused]] size_t ldmz,
  double* den_eval, double* dden_x_eval, double* dden_y_eval,
  double* dden_z_eval, double* gamma, double* tau, double* lapl ) {

   const auto inbe = static_cast<int32_t>(nbe);

   for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioffs = size_t(i) * ldxs;
      const size_t ioffz = size_t(i) * ldxz;

      const auto*   Xs_i = Xs + ioffs;
      const auto*   Xz_i = Xz + ioffz;

      double rhos = blas::dot( inbe, basis_eval + ioffs, 1, Xs_i, 1 ); // S density
      double rhoz = blas::dot( inbe, basis_eval + ioffz, 1, Xz_i, 1 ); // Z density


      den_eval[2*i]   = 0.5*(rhos + rhoz); // rho_+
      den_eval[2*i+1] = 0.5*(rhos - rhoz); // rho_-

      const auto dndx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffs, 1, Xs_i, 1 );
      const auto dndy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffs, 1, Xs_i, 1 );
      const auto dndz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffs, 1, Xs_i, 1 );

      const auto dMzdx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffz, 1, Xz_i, 1 );
      const auto dMzdy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffz, 1, Xz_i, 1 );
      const auto dMzdz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffz, 1, Xz_i, 1 );

      dden_x_eval[2*i] = dndx; // dn / dx
      dden_y_eval[2*i] = dndy; // dn / dy
      dden_z_eval[2*i] = dndz; // dn / dz

      dden_x_eval[2*i+1] = dMzdx; // dMz / dx
      dden_y_eval[2*i+1] = dMzdy; // dMz / dy
      dden_z_eval[2*i+1] = dMzdz; // dMz / dz

      // (del n).(del n)
      const auto dn_sq  = dndx*dndx + dndy*dndy + dndz*dndz;
      // (del Mz).(del Mz)
      const auto dMz_sq = dMzdx*dMzdx + dMzdy*dMzdy + dMzdz*dMzdz;
      // (del n).(del Mz)
      const auto dn_dMz = dndx*dMzdx + dndy*dMzdy + dndz*dMzdz;

      gamma[3*i  ] = 0.25*(dn_sq + dMz_sq) + 0.5*dn_dMz;
      gamma[3*i+1] = 0.25*(dn_sq - dMz_sq);
      gamma[3*i+2] = 0.25*(dn_sq + dMz_sq) - 0.5*dn_dMz;

      auto taus  = 0.5*blas::dot( inbe, dbasis_x_eval + ioffs, 1, mmat_xs + ioffs, 1);
           taus += 0.5*blas::dot( inbe, dbasis_y_eval + ioffs, 1, mmat_ys + ioffs, 1);
           taus += 0.5*blas::dot( inbe, dbasis_z_eval + ioffs, 1, mmat_zs + ioffs, 1);
      auto tauz  = 0.5*blas::dot( inbe, dbasis_x_eval + ioffz, 1, mmat_xz + ioffz, 1);
           tauz += 0.5*blas::dot( inbe, dbasis_y_eval + ioffz, 1, mmat_yz + ioffz, 1);
           tauz += 0.5*blas::dot( inbe, dbasis_z_eval + ioffz, 1, mmat_zz + ioffz, 1);

      tau[2*i]   = 0.5*(taus + tauz);
      tau[2*i+1] = 0.5*(taus - tauz);

      if (lapl != nullptr) {
        auto lapls = 2. * blas::dot( inbe, lbasis_eval + ioffs, 1, Xs_i, 1) + 4. * taus;
        auto laplz = 2. * blas::dot( inbe, lbasis_eval + ioffz, 1, Xz_i, 1) + 4. * tauz;

        lapl[2*i]   = 0.5*(lapls + laplz);
        lapl[2*i+1] = 0.5*(lapls - laplz);
      }

   }
}



void ReferenceLocalHostWorkDriver::eval_uvvar_gga_gks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eval, const double *dbasis_y_eval,
    const double* dbasis_z_eval, const double* Xs, size_t ldxs,
    const double* Xz, size_t ldxz, const double* Xx, size_t ldxx,
    const double* Xy, size_t ldxy, double* den_eval,
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, double* gamma, double* K, double* H, const double dtol) {

   const auto inbe = static_cast<int32_t>(nbe);

   auto *KZ = K; // KZ // store K in the Z matrix
   auto *KY = KZ + npts;
   auto *KX = KY + npts;

   auto *HZ = H; // KZ // store K in the Z matrix
   auto *HY = HZ + npts;
   auto *HX = HY + npts;

   double dtolsq = dtol*dtol;

   for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioffs = size_t(i) * ldxs;
      const size_t ioffz = size_t(i) * ldxz;
      const size_t ioffx = size_t(i) * ldxx;
      const size_t ioffy = size_t(i) * ldxy;

      const auto*   Xs_i = Xs + ioffs;
      const auto*   Xz_i = Xz + ioffz;
      const auto*   Xx_i = Xx + ioffx;
      const auto*   Xy_i = Xy + ioffy;

      const double rhos = blas::dot( inbe, basis_eval + ioffs, 1, Xs_i, 1 );
      const double rhoz = blas::dot( inbe, basis_eval + ioffz, 1, Xz_i, 1 );
      const double rhox = blas::dot( inbe, basis_eval + ioffx, 1, Xx_i, 1 );
      const double rhoy = blas::dot( inbe, basis_eval + ioffy, 1, Xy_i, 1 );

      const auto dndx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffs, 1, Xs_i, 1 );
      const auto dndy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffs, 1, Xs_i, 1 );
      const auto dndz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffs, 1, Xs_i, 1 );

      const auto dMzdx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffz, 1, Xz_i, 1 );
      const auto dMzdy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffz, 1, Xz_i, 1 );
      const auto dMzdz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffz, 1, Xz_i, 1 );

      const auto dMxdx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffx, 1, Xx_i, 1 );
      const auto dMxdy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffx, 1, Xx_i, 1 );
      const auto dMxdz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffx, 1, Xx_i, 1 );

      const auto dMydx =
        2. * blas::dot( inbe, dbasis_x_eval + ioffy, 1, Xy_i, 1 );
      const auto dMydy =
        2. * blas::dot( inbe, dbasis_y_eval + ioffy, 1, Xy_i, 1 );
      const auto dMydz =
        2. * blas::dot( inbe, dbasis_z_eval + ioffy, 1, Xy_i, 1 );


      dden_x_eval[4 * i] = dndx;
      dden_y_eval[4 * i] = dndy;
      dden_z_eval[4 * i] = dndz;

      dden_x_eval[4 * i + 1] = dMzdx;
      dden_y_eval[4 * i + 1] = dMzdy;
      dden_z_eval[4 * i + 1] = dMzdz;

      dden_x_eval[4 * i + 2] = dMydx;
      dden_y_eval[4 * i + 2] = dMydy;
      dden_z_eval[4 * i + 2] = dMydz;

      dden_x_eval[4 * i + 3] = dMxdx;
      dden_y_eval[4 * i + 3] = dMxdy;
      dden_z_eval[4 * i + 3] = dMxdz;

      double mtemp = rhoz * rhoz + rhox * rhox + rhoy * rhoy;
      double mnorm = 0;

      auto dels_dot_dels = dndx * dndx + dndy * dndy + dndz * dndz;
      auto delz_dot_delz = dMzdx * dMzdx + dMzdy * dMzdy + dMzdz * dMzdz;
      auto delx_dot_delx = dMxdx * dMxdx + dMxdy * dMxdy + dMxdz * dMxdz;
      auto dely_dot_dely = dMydx * dMydx + dMydy * dMydy + dMydz * dMydz;

      auto dels_dot_delz = dndx * dMzdx + dndy * dMzdy + dndz * dMzdz;
      auto dels_dot_delx = dndx * dMxdx + dndy * dMxdy + dndz * dMxdz;
      auto dels_dot_dely = dndx * dMydx + dndy * dMydy + dndz * dMydz;

      auto sum = delz_dot_delz + delx_dot_delx + dely_dot_dely;
      auto s_sum =
          dels_dot_delz * rhoz + dels_dot_delx * rhox + dels_dot_dely * rhoy;

      double sign = 1.;
      if (std::signbit(s_sum))
        sign = -1.;

      if (mtemp > dtolsq) {
        mnorm = sqrt(mtemp);
        auto sqsum2 =
          sqrt(dels_dot_delz * dels_dot_delz + dels_dot_delx * dels_dot_delx +
               dels_dot_dely * dels_dot_dely);
        KZ[i] = rhoz / mnorm;
        KY[i] = rhoy / mnorm;
        KX[i] = rhox / mnorm;
        HZ[i] = sign * dels_dot_delz / sqsum2;
        HY[i] = sign * dels_dot_dely / sqsum2;
        HX[i] = sign * dels_dot_delx / sqsum2;

        gamma[3 * i] = 0.25 * (dels_dot_dels + sum) + 0.5 * sign * sqsum2;
        gamma[3 * i + 1] = 0.25 * (dels_dot_dels - sum);
        gamma[3 * i + 2] = 0.25 * (dels_dot_dels + sum) - 0.5 * sign * sqsum2;
      } else {
        mnorm = (1. / 3.) * (rhox + rhoy + rhoz);
        
        KZ[i] = 1. / 3.;
        KY[i] = 1. / 3.;
        KX[i] = 1. / 3.;

        HZ[i] = sign / 3.;
        HY[i] = sign / 3.;
        HX[i] = sign / 3.;

        auto dels_dot_delms = (1./3.)*(dels_dot_delx + dels_dot_dely + dels_dot_delz);
        gamma[3 * i] = 0.25 * (dels_dot_dels + sum) + 0.5 * sign * dels_dot_delms;
        gamma[3 * i + 1] = 0.25 * (dels_dot_dels - sum);
        gamma[3 * i + 2] = 0.25 * (dels_dot_dels + sum) - 0.5 * sign * dels_dot_delms;
      }
      
      den_eval[2 * i] = 0.5 * (rhos + mnorm);
      den_eval[2 * i + 1] = 0.5 * (rhos - mnorm);
      
    }

}
  

void ReferenceLocalHostWorkDriver::eval_uvvar_gga_dks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    const double* d2basis_xx_eval, const double* d2basis_xy_eval, const double* d2basis_xz_eval, 
    const double* d2basis_yy_eval, const double* d2basis_yz_eval, const double* d2basis_zz_eval,
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz, 
    const double* Xx, size_t ldxx, const double* Xy, size_t ldxy,  
    const double* Xs_x_SS, size_t ldxs_ss, const double* Xz_x_SS, size_t ldxz_ss, 
    const double* Xx_x_SS, size_t ldxx_ss, const double* Xy_x_SS, size_t ldxy_ss, 
    const double* Xs_y_SS, const double* Xz_y_SS, 
    const double* Xx_y_SS, const double* Xy_y_SS, 
    const double* Xs_z_SS, const double* Xz_z_SS, 
    const double* Xx_z_SS, const double* Xy_z_SS, 
    const double* immat_x_z, const double* immat_y_x, const double* immat_z_y,
    const double* immat_y_z, const double* immat_z_x, const double* immat_x_y,
    const double* immat_x_s, const double* immat_y_s, const double* immat_z_s,
    double* den_eval, double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, 
    double* gamma, double* K, double* H, const double dtol ) {

   auto *KZ = K; // KZ // store K in the Z matrix
   auto *KY = KZ + npts;
   auto *KX = KY + npts;

   auto *HZ = H; // KZ // store K in the Z matrix
   auto *HY = HZ + npts;
   auto *HX = HY + npts;

   double dtolsq = dtol*dtol;

   for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const size_t ioffs = size_t(i) * ldxs;
      const size_t ioffz = size_t(i) * ldxz;
      const size_t ioffx = size_t(i) * ldxx;
      const size_t ioffy = size_t(i) * ldxy;

      const auto*   Xs_i = Xs + ioffs;
      const auto*   Xz_i = Xz + ioffz;
      const auto*   Xx_i = Xx + ioffx;
      const auto*   Xy_i = Xy + ioffy;

      const auto*   Xs_x_i_ss = Xs_x_SS + ioffs;
      const auto*   Xz_x_i_ss = Xz_x_SS + ioffz;
      const auto*   Xx_x_i_ss = Xx_x_SS + ioffx;
      const auto*   Xy_x_i_ss = Xy_x_SS + ioffy;

      const auto*   Xs_y_i_ss = Xs_y_SS + ioffs;
      const auto*   Xz_y_i_ss = Xz_y_SS + ioffz;
      const auto*   Xx_y_i_ss = Xx_y_SS + ioffx;
      const auto*   Xy_y_i_ss = Xy_y_SS + ioffy;

      const auto*   Xs_z_i_ss = Xs_z_SS + ioffs; // Re[Ps] * d/dz chi nu
      const auto*   Xz_z_i_ss = Xz_z_SS + ioffz;
      const auto*   Xx_z_i_ss = Xx_z_SS + ioffx;
      const auto*   Xy_z_i_ss = Xy_z_SS + ioffy;

      const auto*   Xz_x_i_im = immat_x_z + ioffz; // im[Pz] * d/dx chi nu
      const auto*   Xz_y_i_im = immat_y_z + ioffz;

      const auto*   Xx_y_i_im = immat_y_x + ioffx;
      const auto*   Xx_z_i_im = immat_z_x + ioffx;

      const auto*   Xy_z_i_im = immat_z_y + ioffy;
      const auto*   Xy_x_i_im = immat_x_y + ioffy;

      const auto*   Xs_x_i_im = immat_x_s + ioffs;
      const auto*   Xs_y_i_im = immat_y_s + ioffs;
      const auto*   Xs_z_i_im = immat_z_s + ioffs;

      // rho LL
      double rhos = blas::dot( nbe, basis_eval + ioffs, 1, Xs_i, 1 );
      double rhoz = blas::dot( nbe, basis_eval + ioffz, 1, Xz_i, 1 );
      double rhox = blas::dot( nbe, basis_eval + ioffx, 1, Xx_i, 1 );
      double rhoy = blas::dot( nbe, basis_eval + ioffy, 1, Xy_i, 1 );

      // rho SS

      // s
      const double rhos_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xs_y_i_ss, 1 );
      const double rhos_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xs_x_i_ss, 1 );
      const double rhos_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xs_z_i_ss, 1 );

      // s anti
      const double rhos_z_yx = blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xz_x_i_im, 1 );
      const double rhos_z_xy = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xz_y_i_im, 1 );

      const double rhos_x_yz = blas::dot( nbe, dbasis_z_eval + ioffx, 1, Xx_y_i_im, 1 );
      const double rhos_x_zy = blas::dot( nbe, dbasis_y_eval + ioffx, 1, Xx_z_i_im, 1 );

      const double rhos_y_zx = blas::dot( nbe, dbasis_x_eval + ioffy, 1, Xy_z_i_im, 1 );
      const double rhos_y_xz = blas::dot( nbe, dbasis_z_eval + ioffy, 1, Xy_x_i_im, 1 );


      auto rhos_ss = rhos_yy_ss + rhos_xx_ss + rhos_zz_ss; 
      auto rhos_anti = rhos_z_xy - rhos_z_yx + rhos_x_yz - rhos_x_zy + rhos_y_zx - rhos_y_xz;
      rhos_ss -= rhos_anti;

      // z
      const double rhoz_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xz_x_i_ss, 1 );
      const double rhoz_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xz_y_i_ss, 1 );       
      const double rhoz_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xz_z_i_ss, 1 );

      const double rhoz_xz_ss = blas::dot( nbe, dbasis_x_eval + ioffx, 1, Xx_z_i_ss, 1 );
      const double rhoz_zx_ss = blas::dot( nbe, dbasis_z_eval + ioffx, 1, Xx_x_i_ss, 1 );

      const double rhoz_yz_ss = blas::dot( nbe, dbasis_y_eval + ioffy, 1, Xy_z_i_ss, 1 );
      const double rhoz_zy_ss = blas::dot( nbe, dbasis_z_eval + ioffy, 1, Xy_y_i_ss, 1 );

      const double rhoz_s_yx = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double rhoz_s_xy = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xs_y_i_im, 1 );

      const auto rhoz_ss_dot = rhoz_zz_ss - rhoz_yy_ss - rhoz_xx_ss;
      const auto rhoz_ss_cross = rhoz_xz_ss + rhoz_zx_ss + rhoz_yz_ss + rhoz_zy_ss; 
      const auto rhoz_ss_anti = rhoz_s_xy - rhoz_s_yx;
      const auto rhoz_ss = rhoz_ss_dot + rhoz_ss_anti + rhoz_ss_cross;

      // x
      const double rhox_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xx_x_i_ss, 1 );
      const double rhox_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xx_y_i_ss, 1 );       
      const double rhox_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xx_z_i_ss, 1 );

      const double rhox_xz_ss = blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xz_z_i_ss, 1 );
      const double rhox_zx_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xz_x_i_ss, 1 );

      const double rhox_yx_ss = blas::dot( nbe, dbasis_x_eval + ioffy, 1, Xy_y_i_ss, 1 );
      const double rhox_xy_ss = blas::dot( nbe, dbasis_y_eval + ioffy, 1, Xy_x_i_ss, 1 );

      const auto rhox_ss_dot = rhox_xx_ss - rhox_zz_ss - rhox_yy_ss;
      const auto rhox_ss_cross = rhox_xz_ss +rhox_zx_ss + rhox_yx_ss + rhox_xy_ss;

      // x anti

      const double rhox_zy = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xs_y_i_im, 1 );
      const double rhox_yz = blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xs_z_i_im, 1 );   
      
      const auto rhox_ss_anti = rhox_yz - rhox_zy;
      const auto rhox_ss = rhox_ss_dot + rhox_ss_anti + rhox_ss_cross;

      // y
      const double rhoy_xx_ss = blas::dot( nbe, dbasis_x_eval + ioffy, 1, Xy_x_i_ss, 1 );
      const double rhoy_yy_ss = blas::dot( nbe, dbasis_y_eval + ioffy, 1, Xy_y_i_ss, 1 );       
      const double rhoy_zz_ss = blas::dot( nbe, dbasis_z_eval + ioffy, 1, Xy_z_i_ss, 1 );

      const double rhoy_zy_ss = blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xz_z_i_ss, 1 );
      const double rhoy_yz_ss = blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xz_y_i_ss, 1 );

      const double rhoy_yx_ss = blas::dot( nbe, dbasis_x_eval + ioffx, 1, Xx_y_i_ss, 1 );
      const double rhoy_xy_ss = blas::dot( nbe, dbasis_y_eval + ioffx, 1, Xx_x_i_ss, 1 );

      const auto rhoy_ss_dot =  rhoy_yy_ss - rhoy_xx_ss - rhoy_zz_ss;
      const auto rhoy_ss_cross = rhoy_xy_ss + rhoy_yx_ss + rhoy_yz_ss + rhoy_zy_ss; 

      // y anti

      const double rhoy_zx = blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double rhoy_xz = blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xs_z_i_im, 1 );   
    
      const auto rhoy_ss_anti = rhoy_zx - rhoy_xz;
      const auto rhoy_ss = rhoy_ss_dot + rhoy_ss_anti + rhoy_ss_cross;


      // // total rho (LL + SS)
     rhos += RKB_factor * rhos_ss;
     rhoz += RKB_factor * rhoz_ss; 
     rhox += RKB_factor * rhox_ss;
     rhoy += RKB_factor * rhoy_ss;


      // // dV/dk LL
      auto dndx =
        2. * blas::dot( nbe, dbasis_x_eval + ioffs, 1, Xs_i, 1 );
      auto dndy =
        2. * blas::dot( nbe, dbasis_y_eval + ioffs, 1, Xs_i, 1 );
      auto dndz =
        2. * blas::dot( nbe, dbasis_z_eval + ioffs, 1, Xs_i, 1 );

      auto dMzdx =
        2. * blas::dot( nbe, dbasis_x_eval + ioffz, 1, Xz_i, 1 );
      auto dMzdy =
        2. * blas::dot( nbe, dbasis_y_eval + ioffz, 1, Xz_i, 1 );
      auto dMzdz =
        2. * blas::dot( nbe, dbasis_z_eval + ioffz, 1, Xz_i, 1 );

      auto dMxdx =
        2. * blas::dot( nbe, dbasis_x_eval + ioffx, 1, Xx_i, 1 );
      auto dMxdy =
        2. * blas::dot( nbe, dbasis_y_eval + ioffx, 1, Xx_i, 1 );
      auto dMxdz =
        2. * blas::dot( nbe, dbasis_z_eval + ioffx, 1, Xx_i, 1 );

      auto dMydx =
        2. * blas::dot( nbe, dbasis_x_eval + ioffy, 1, Xy_i, 1 );
      auto dMydy =
        2. * blas::dot( nbe, dbasis_y_eval + ioffy, 1, Xy_i, 1 );
      auto dMydz =
        2. * blas::dot( nbe, dbasis_z_eval + ioffy, 1, Xy_i, 1 );

      // // dV/dk SS
      // dndx SS
      const auto dndx_xxx_ss =
        blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xs_x_i_ss, 1 );
      const auto dndx_yxy_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xs_y_i_ss, 1 );
      const auto dndx_zxz_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xs_z_i_ss, 1 );

      auto dndx_ss = dndx_xxx_ss + dndx_yxy_ss + dndx_zxz_ss;

      const double dndx_z_yx = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xz_x_i_im, 1 );
      const double dndx_z_xy = blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xz_y_i_im, 1 );

      const double dndx_x_yz = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xx_y_i_im, 1 );
      const double dndx_x_zy = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xx_z_i_im, 1 );

      const double dndx_y_zx = blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xy_z_i_im, 1 );
      const double dndx_y_xz = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xy_x_i_im, 1 );

      const auto dndx_anti = dndx_z_xy - dndx_z_yx + dndx_x_yz - dndx_x_zy + dndx_y_zx - dndx_y_xz;
      dndx_ss -= dndx_anti;

      // dndy SS
      const auto dndy_xyx_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xs_x_i_ss, 1 );
      const auto dndy_yyy_ss =
        blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xs_y_i_ss, 1 );
      const auto dndy_zyz_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xs_z_i_ss, 1 );

      auto dndy_ss = dndy_xyx_ss + dndy_yyy_ss + dndy_zyz_ss;

      const double dndy_z_yx = blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xz_x_i_im, 1 );
      const double dndy_z_xy = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xz_y_i_im, 1 );
      const double dndy_x_yz = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xx_y_i_im, 1 );
      const double dndy_x_zy = blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xx_z_i_im, 1 );
      const double dndy_y_zx = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xy_z_i_im, 1 );
      const double dndy_y_xz = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xy_x_i_im, 1 );

      const auto dndy_anti = dndy_z_xy - dndy_z_yx + dndy_x_yz - dndy_x_zy + dndy_y_zx - dndy_y_xz;
      dndy_ss -= dndy_anti;

      // dndz SS
      const auto dndz_xzx_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xs_x_i_ss, 1 );
      const auto dndz_yzy_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xs_y_i_ss, 1 );
      const auto dndz_zzz_ss =
        blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xs_z_i_ss, 1 );

      auto dndz_ss = dndz_xzx_ss + dndz_yzy_ss + dndz_zzz_ss;

      const double dndz_z_yx = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xz_x_i_im, 1 );
      const double dndz_z_xy = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xz_y_i_im, 1 );
      const double dndz_x_yz = blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xx_y_i_im, 1 );
      const double dndz_x_zy = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xx_z_i_im, 1 );
      const double dndz_y_zx = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xy_z_i_im, 1 );
      const double dndz_y_xz = blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xy_x_i_im, 1 );

      const auto dndz_anti = dndz_z_xy - dndz_z_yx + dndz_x_yz - dndz_x_zy + dndz_y_zx - dndz_y_xz;
      dndz_ss -= dndz_anti;

//////. dMz SS

      /// dMzdx SS
      const auto dMzdx_xxx_ss =
        blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xz_x_i_ss, 1 );
      const auto dMzdx_yxy_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xz_y_i_ss, 1 );
      const auto dMzdx_zxz_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xz_z_i_ss, 1 );

      auto dMzdx_ss = dMzdx_zxz_ss - dMzdx_xxx_ss - dMzdx_yxy_ss;

      const double dMzdx_xz_ss = blas::dot( nbe, d2basis_xx_eval + ioffx, 1, Xx_z_i_ss, 1 );
      const double dMzdx_zx_ss = blas::dot( nbe, d2basis_xz_eval + ioffx, 1, Xx_x_i_ss, 1 );
      const double dMzdx_yz_ss = blas::dot( nbe, d2basis_xy_eval + ioffx, 1, Xy_z_i_ss, 1 );
      const double dMzdx_zy_ss = blas::dot( nbe, d2basis_xz_eval + ioffx, 1, Xy_y_i_ss, 1 );
      const double dMzdx_s_yx = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double dMzdx_s_xy = blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xs_y_i_im, 1 );

      const auto dMzdx_cross = dMzdx_xz_ss + dMzdx_zx_ss + dMzdx_yz_ss + dMzdx_zy_ss;
      const auto dMzdx_anti = dMzdx_s_xy - dMzdx_s_yx;

      dMzdx_ss += dMzdx_cross + dMzdx_anti;

      /// dMzdy SS
      const auto dMzdy_xyx_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xz_x_i_ss, 1 );
      const auto dMzdy_yyy_ss =
        blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xz_y_i_ss, 1 );
      const auto dMzdy_zyz_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xz_z_i_ss, 1 );


      auto dMzdy_ss = dMzdy_zyz_ss - dMzdy_xyx_ss - dMzdy_yyy_ss;

      const double dMzdy_xz_ss = blas::dot( nbe, d2basis_xy_eval + ioffx, 1, Xx_z_i_ss, 1 );
      const double dMzdy_zx_ss = blas::dot( nbe, d2basis_yz_eval + ioffx, 1, Xx_x_i_ss, 1 );
      const double dMzdy_yz_ss = blas::dot( nbe, d2basis_yy_eval + ioffx, 1, Xy_z_i_ss, 1 );
      const double dMzdy_zy_ss = blas::dot( nbe, d2basis_yz_eval + ioffx, 1, Xy_y_i_ss, 1 );
      const double dMzdy_s_yx = blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double dMzdy_s_xy = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xs_y_i_im, 1 );

      const auto dMzdy_cross = dMzdy_xz_ss + dMzdy_zx_ss + dMzdy_yz_ss + dMzdy_zy_ss;
      const auto dMzdy_anti = dMzdy_s_xy - dMzdy_s_yx;

      dMzdy_ss += dMzdy_cross + dMzdy_anti;

      /// dMzdz SS
      const auto dMzdz_xzx_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xz_x_i_ss, 1 );
      const auto dMzdz_yzy_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xz_y_i_ss, 1 );
      const auto dMzdz_zzz_ss =
        blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xz_z_i_ss, 1 );

      auto dMzdz_ss = dMzdz_zzz_ss - dMzdz_xzx_ss - dMzdz_yzy_ss;

      const double dMzdz_xz_ss = blas::dot( nbe, d2basis_xz_eval + ioffx, 1, Xx_z_i_ss, 1 );
      const double dMzdz_zx_ss = blas::dot( nbe, d2basis_zz_eval + ioffx, 1, Xx_x_i_ss, 1 );
      const double dMzdz_yz_ss = blas::dot( nbe, d2basis_yz_eval + ioffx, 1, Xy_z_i_ss, 1 );
      const double dMzdz_zy_ss = blas::dot( nbe, d2basis_zz_eval + ioffx, 1, Xy_y_i_ss, 1 );
      const double dMzdz_s_yx = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double dMzdz_s_xy = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xs_y_i_im, 1 );

      const auto dMzdz_cross = dMzdz_xz_ss + dMzdz_zx_ss + dMzdz_yz_ss + dMzdz_zy_ss;
      const auto dMzdz_anti = dMzdz_s_xy - dMzdz_s_yx;

      dMzdz_ss += dMzdz_cross + dMzdz_anti;

      // //////  dMx SS

      /// dMxdz SS
      const auto dMxdx_xxx_ss =
        blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xx_x_i_ss, 1 );
      const auto dMxdx_yxy_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xx_y_i_ss, 1 );
      const auto dMxdx_zxz_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xx_z_i_ss, 1 );


      auto dMxdx_ss = dMxdx_xxx_ss - dMxdx_yxy_ss - dMxdx_zxz_ss;

      const double dMxdx_xz_ss = blas::dot( nbe, d2basis_xx_eval + ioffz, 1, Xz_z_i_ss, 1 );
      const double dMxdx_zx_ss = blas::dot( nbe, d2basis_xz_eval + ioffz, 1, Xz_x_i_ss, 1 );
      const double dMxdx_yx_ss = blas::dot( nbe, d2basis_xx_eval + ioffy, 1, Xy_y_i_ss, 1 );
      const double dMxdx_xy_ss = blas::dot( nbe, d2basis_xy_eval + ioffy, 1, Xy_x_i_ss, 1 );
      const double dMxdx_zy_im = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xs_y_i_im, 1 );
      const double dMxdx_yz_im = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xs_z_i_im, 1 );  

      const auto dMxdx_cross = dMxdx_xz_ss + dMxdx_zx_ss + dMxdx_yx_ss + dMxdx_xy_ss;
      const auto dMxdx_anti = dMxdx_yz_im - dMxdx_zy_im;

      dMxdx_ss += dMxdx_cross  + dMxdx_anti;

      /// dMxdy. SS
      const auto dMxdy_xyx_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xx_x_i_ss, 1 );
      const auto dMxdy_yyy_ss =
        blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xx_y_i_ss, 1 );
      const auto dMxdy_zyz_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xx_z_i_ss, 1 );

      auto dMxdy_ss = dMxdy_xyx_ss - dMxdy_yyy_ss - dMxdy_zyz_ss;

      const double dMxdy_xz_ss = blas::dot( nbe, d2basis_xy_eval + ioffz, 1, Xz_z_i_ss, 1 );
      const double dMxdy_zx_ss = blas::dot( nbe, d2basis_yz_eval + ioffz, 1, Xz_x_i_ss, 1 );
      const double dMxdy_yx_ss = blas::dot( nbe, d2basis_xy_eval + ioffy, 1, Xy_y_i_ss, 1 );
      const double dMxdy_xy_ss = blas::dot( nbe, d2basis_yy_eval + ioffy, 1, Xy_x_i_ss, 1 );
      const double dMxdy_zy_im = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xs_y_i_im, 1 );
      const double dMxdy_yz_im = blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xs_z_i_im, 1 );  

      const auto dMxdy_cross = dMxdy_xz_ss + dMxdy_zx_ss + dMxdy_yx_ss + dMxdy_xy_ss;
      const auto dMxdy_anti = dMxdy_yz_im - dMxdy_zy_im;

      dMxdy_ss += dMxdy_cross  + dMxdy_anti;

/// dMxdz. SS
      const auto dMxdz_xzx_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xx_x_i_ss, 1 );
      const auto dMxdz_yzy_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xx_y_i_ss, 1 );
      const auto dMxdz_zzz_ss =
        blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xx_z_i_ss, 1 );

      auto dMxdz_ss = dMxdz_xzx_ss - dMxdz_yzy_ss - dMxdz_zzz_ss;

      const double dMxdz_xz_ss = blas::dot( nbe, d2basis_xz_eval + ioffz, 1, Xz_z_i_ss, 1 );
      const double dMxdz_zx_ss = blas::dot( nbe, d2basis_zz_eval + ioffz, 1, Xz_x_i_ss, 1 );
      const double dMxdz_yx_ss = blas::dot( nbe, d2basis_xz_eval + ioffy, 1, Xy_y_i_ss, 1 );
      const double dMxdz_xy_ss = blas::dot( nbe, d2basis_yz_eval + ioffy, 1, Xy_x_i_ss, 1 );
      const double dMxdz_zy_im = blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xs_y_i_im, 1 );
      const double dMxdz_yz_im = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xs_z_i_im, 1 );  

      const auto dMxdz_cross = dMxdz_xz_ss + dMxdz_zx_ss + dMxdz_yx_ss + dMxdz_xy_ss;
      const auto dMxdz_anti = dMxdz_yz_im - dMxdz_zy_im;

      dMxdz_ss += dMxdz_cross  + dMxdz_anti;

      //////. dMy SS

      // dMydx SS
      const auto dMydx_xxx_ss =
        blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xy_x_i_ss, 1 );
      const auto dMydx_yxy_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xy_y_i_ss, 1 );
      const auto dMydx_zxz_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xy_z_i_ss, 1 );

      auto dMydx_ss = dMydx_yxy_ss - dMydx_xxx_ss - dMydx_zxz_ss;

      const double dMydx_zy_ss = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xz_z_i_ss, 1 );
      const double dMydx_yz_ss = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xz_y_i_ss, 1 );
      const double dMydx_yx_ss = blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xx_y_i_ss, 1 );
      const double dMydx_xy_ss = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xx_x_i_ss, 1 );
      const double dMydx_zx_im = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double dMydx_xz_im = blas::dot( nbe, d2basis_xx_eval + ioffs, 1, Xs_z_i_im, 1 ); 

      const auto dMydx_cross = dMydx_zy_ss + dMydx_yz_ss + dMydx_yx_ss + dMydx_xy_ss;
      const auto dMydx_anti = dMydx_zx_im - dMydx_xz_im;

      dMydx_ss += dMydx_cross  + dMydx_anti;

      /// dMydy
      const auto dMydy_xyx_ss =
        blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xy_x_i_ss, 1 );
      const auto dMydy_yyy_ss =
        blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xy_y_i_ss, 1 );
      const auto dMydy_zyz_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xy_z_i_ss, 1 );

      auto dMydy_ss = dMydy_yyy_ss - dMydy_xyx_ss - dMydy_zyz_ss;

      const double dMydy_zy_ss = blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xz_z_i_ss, 1 );
      const double dMydy_yz_ss = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xz_y_i_ss, 1 );
      const double dMydy_yx_ss = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xx_y_i_ss, 1 );
      const double dMydy_xy_ss = blas::dot( nbe, d2basis_yy_eval + ioffs, 1, Xx_x_i_ss, 1 );
      const double dMydy_zx_im = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double dMydy_xz_im = blas::dot( nbe, d2basis_xy_eval + ioffs, 1, Xs_z_i_im, 1 ); 

      const auto dMydy_cross = dMydy_zy_ss + dMydy_yz_ss + dMydy_yx_ss + dMydy_xy_ss;
      const auto dMydy_anti = dMydy_zx_im - dMydy_xz_im;

      dMydy_ss += dMydy_cross  + dMydy_anti;

      //// dMydz
      const auto dMydz_xzx_ss =
        blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xy_x_i_ss, 1 );
      const auto dMydz_yzy_ss =
        blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xy_y_i_ss, 1 );
      const auto dMydz_zzz_ss =
        blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xy_z_i_ss, 1 );

      auto dMydz_ss = dMydz_yzy_ss - dMydz_xzx_ss - dMydz_zzz_ss;

      const double dMydz_zy_ss = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xz_z_i_ss, 1 );
      const double dMydz_yz_ss = blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xz_y_i_ss, 1 );
      const double dMydz_yx_ss = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xx_y_i_ss, 1 );
      const double dMydz_xy_ss = blas::dot( nbe, d2basis_yz_eval + ioffs, 1, Xx_x_i_ss, 1 );
      const double dMydz_zx_im = blas::dot( nbe, d2basis_zz_eval + ioffs, 1, Xs_x_i_im, 1 );
      const double dMydz_xz_im = blas::dot( nbe, d2basis_xz_eval + ioffs, 1, Xs_z_i_im, 1 ); 

      const auto dMydz_cross = dMydz_zy_ss + dMydz_yz_ss + dMydz_yx_ss + dMydz_xy_ss;
      const auto dMydz_anti = dMydz_zx_im - dMydz_xz_im;

      dMydz_ss += dMydz_cross  + dMydz_anti;

      // // Form total density gradients (LL + SS)
      dndx +=  2 * RKB_factor * dndx_ss;
      dndy +=  2 * RKB_factor * dndy_ss;
      dndz +=  2 * RKB_factor * dndz_ss;
      dMzdx += 2 * RKB_factor * dMzdx_ss;
      dMzdy += 2 * RKB_factor * dMzdy_ss;
      dMzdz += 2 * RKB_factor * dMzdz_ss;
      dMydx += 2 * RKB_factor * dMydx_ss;
      dMydy += 2 * RKB_factor * dMydy_ss;
      dMydz += 2 * RKB_factor * dMydz_ss;
      dMxdx += 2 * RKB_factor * dMxdx_ss;
      dMxdy += 2 * RKB_factor * dMxdy_ss;
      dMxdz += 2 * RKB_factor * dMxdz_ss;

      // Store Pauli Spin Density Gradients in the derivative density eval objects (dden_k_eval)
      dden_x_eval[4 * i] = dndx;
      dden_y_eval[4 * i] = dndy;
      dden_z_eval[4 * i] = dndz;

      dden_x_eval[4 * i + 1] = dMzdx;
      dden_y_eval[4 * i + 1] = dMzdy;
      dden_z_eval[4 * i + 1] = dMzdz;

      dden_x_eval[4 * i + 2] = dMydx;
      dden_y_eval[4 * i + 2] = dMydy;
      dden_z_eval[4 * i + 2] = dMydz;

      dden_x_eval[4 * i + 3] = dMxdx;
      dden_y_eval[4 * i + 3] = dMxdy;
      dden_z_eval[4 * i + 3] = dMxdz;

     // rho_k dot rho_k
      double mtemp = rhoz * rhoz + rhox * rhox + rhoy * rhoy;
      double mnorm = 0;

      auto dels_dot_dels = dndx * dndx + dndy * dndy + dndz * dndz;
      auto delz_dot_delz = dMzdx * dMzdx + dMzdy * dMzdy + dMzdz * dMzdz;
      auto delx_dot_delx = dMxdx * dMxdx + dMxdy * dMxdy + dMxdz * dMxdz;
      auto dely_dot_dely = dMydx * dMydx + dMydy * dMydy + dMydz * dMydz;
      auto dels_dot_delz = dndx * dMzdx + dndy * dMzdy + dndz * dMzdz;
      auto dels_dot_delx = dndx * dMxdx + dndy * dMxdy + dndz * dMxdz;
      auto dels_dot_dely = dndx * dMydx + dndy * dMydy + dndz * dMydz;

      auto sum = delz_dot_delz + delx_dot_delx + dely_dot_dely;
      auto s_sum =
          dels_dot_delz * rhoz + dels_dot_delx * rhox + dels_dot_dely * rhoy;

      double sign = 1.;
      if (std::signbit(s_sum))
        sign = -1.;
      
      if (mtemp > dtolsq) {
        mnorm = sqrt(mtemp);
        auto sqsum2 =
          sqrt(dels_dot_delz * dels_dot_delz + dels_dot_delx * dels_dot_delx +
               dels_dot_dely * dels_dot_dely);
        KZ[i] = rhoz / mnorm;
        KY[i] = rhoy / mnorm;
        KX[i] = rhox / mnorm;
        HZ[i] = sign * dels_dot_delz / sqsum2;
        HY[i] = sign * dels_dot_dely / sqsum2;
        HX[i] = sign * dels_dot_delx / sqsum2;

        gamma[3 * i] = 0.25 * (dels_dot_dels + sum) + 0.5 * sign * sqsum2;
        gamma[3 * i + 1] = 0.25 * (dels_dot_dels - sum);
        gamma[3 * i + 2] = 0.25 * (dels_dot_dels + sum) - 0.5 * sign * sqsum2;
      } else {
        mnorm = (1. / 3.) * (rhox + rhoy + rhoz);
        
        KZ[i] = 1. / 3.;
        KY[i] = 1. / 3.;
        KX[i] = 1. / 3.;

        HZ[i] = sign / 3.;
        HY[i] = sign / 3.;
        HX[i] = sign / 3.;

        auto dels_dot_delms = (1./3.)*(dels_dot_delx + dels_dot_dely + dels_dot_delz);
        gamma[3 * i] = 0.25 * (dels_dot_dels + sum) + 0.5 * sign * dels_dot_delms;
        gamma[3 * i + 1] = 0.25 * (dels_dot_dels - sum);
        gamma[3 * i + 2] = 0.25 * (dels_dot_dels + sum) - 0.5 * sign * dels_dot_delms;
      }

      // NON-COLLINEAR
      den_eval[2 * i] = 0.5 * (rhos + mnorm);
      den_eval[2 * i + 1] = 0.5 * (rhos - mnorm);

      // gamma[3 * i] = 0.25 * (dels_dot_dels + sum) + 0.5 * sign * sqsum2;
      // gamma[3 * i + 1] = 0.25 * (dels_dot_dels - sum);
      // gamma[3 * i + 2] = 0.25 * (dels_dot_dels + sum) - 0.5 * sign * sqsum2;


      // COLLINEAR TESTING
      // den_eval[2 * i] = 0.5 * (rhos + rhoz);
      // den_eval[2 * i + 1] = 0.5 * (rhos - rhoz);

      // gamma[3 * i] = 0.25 * (dels_dot_dels +2*dels_dot_delz + delz_dot_delz);
      // gamma[3 * i + 1] = 0.25 * (dels_dot_dels - delz_dot_delz);
      // gamma[3 * i + 2] = 0.25 * (dels_dot_dels - 2*dels_dot_delz + delz_dot_delz);
    }
}


  // Eval Z Matrix LDA VXC
  void ReferenceLocalHostWorkDriver::eval_zmat_lda_vxc_rks( size_t npts, size_t nbf,
							const double* vrho, const double* basis_eval, double* Z, size_t ldz ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildz  = static_cast<int32_t>(ldz);

    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Z, ildz );

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      auto* z_col = Z + i*ldz;

      const double fact = 0.5 * vrho[i];
      GauXC::blas::scal( inbf, fact, z_col, 1 );

    }

  }

  // Eval Z Matrix LDA VXC
  void ReferenceLocalHostWorkDriver::eval_zmat_lda_vxc_uks( size_t npts, size_t nbf,
              const double* vrho, const double* basis_eval, double* Zs, size_t ldzs,
              double* Zz, size_t ldzz ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildzs = static_cast<int32_t>(ldzs);
    const auto ildzz = static_cast<int32_t>(ldzz);

    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zs, ildzs);
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zz, ildzz);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      auto* zs_col = Zs + i*ldzs;
      auto* zz_col = Zz + i*ldzz;

      const double factp = 0.5 * vrho[2*i];
      const double factm = 0.5 * vrho[2*i+1];

      //eq. 56 https://doi.org/10.1140/epjb/e2018-90170-1
      GauXC::blas::scal( inbf, 0.5*(factp + factm), zs_col, 1 );
      GauXC::blas::scal( inbf, 0.5*(factp - factm), zz_col, 1 );

    }


  }

void ReferenceLocalHostWorkDriver::eval_zmat_lda_vxc_gks( size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, double* Zs, size_t ldzs, double* Zz, size_t ldzz,
    double* Zx, size_t ldzx,double* Zy, size_t ldzy, double *K ) {

  const auto inbe  = static_cast<int32_t>(nbe);
  const auto inpts = static_cast<int32_t>(npts);
  const auto ildzs = static_cast<int32_t>(ldzs);
  const auto ildzz = static_cast<int32_t>(ldzz);
  const auto ildzx = static_cast<int32_t>(ldzx);
  const auto ildzy = static_cast<int32_t>(ldzy);

  auto *KZ = K; // KZ // store K in the Z matrix
  auto *KY = KZ + npts;
  auto *KX = KY + npts;

    blas::lacpy( 'A', inbe, inpts, basis_eval, inbe, Zs, ildzs);
    blas::lacpy( 'A', inbe, inpts, basis_eval, inbe, Zz, ildzz);
    blas::lacpy( 'A', inbe, inpts, basis_eval, inbe, Zx, ildzx);
    blas::lacpy( 'A', inbe, inpts, basis_eval, inbe, Zy, ildzy);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      auto* zs_col = Zs + i*ldzs;
      auto* zz_col = Zz + i*ldzz;
      auto* zx_col = Zx + i*ldzx;
      auto* zy_col = Zy + i*ldzy;

      const double factp = 0.5 * vrho[2*i];
      const double factm = 0.5 * vrho[2*i+1];
      const double factor = 0.5 * (factp - factm);

      //eq. 56 https://doi.org/10.1140/epjb/e2018-90170-1
      GauXC::blas::scal( inbe, 0.5*(factp + factm), zs_col, 1 );
      GauXC::blas::scal( inbe, KZ[i] * factor, zz_col, 1 );
      GauXC::blas::scal( inbe, KX[i] * factor, zx_col, 1 );
      GauXC::blas::scal( inbe, KY[i] * factor, zy_col, 1 );

    }

}


void ReferenceLocalHostWorkDriver::eval_zmat_lda_vxc_dks(
    size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz, double* Zx, size_t ldzx, double* Zy, size_t ldzy,
    double* Zs_x_SS, size_t ldzs_ss, double* Zz_x_SS, size_t ldzz_ss, 
    double* Zx_x_SS, size_t ldzx_ss, double* Zy_x_SS, size_t ldzy_ss, 
    double* Zs_y_SS, double* Zz_y_SS, 
    double* Zx_y_SS, double* Zy_y_SS, 
    double* Zs_z_SS, double* Zz_z_SS, 
    double* Zx_z_SS, double* Zy_z_SS,
    double* K ) {

  auto *KZ = K; // KZ // store K in the Z matrix
  auto *KY = KZ + npts;
  auto *KX = KY + npts;

    blas::lacpy( 'A', nbe, npts, basis_eval, nbe, Zs, ldzs);
    blas::lacpy( 'A', nbe, npts, basis_eval, nbe, Zz, ldzz);
    blas::lacpy( 'A', nbe, npts, basis_eval, nbe, Zx, ldzx);
    blas::lacpy( 'A', nbe, npts, basis_eval, nbe, Zy, ldzy);


    // SS x
    blas::lacpy( 'A', nbe, npts, dbasis_x_eval, nbe, Zs_x_SS, ldzs);
    blas::lacpy( 'A', nbe, npts, dbasis_x_eval, nbe, Zz_x_SS, ldzz);
    blas::lacpy( 'A', nbe, npts, dbasis_x_eval, nbe, Zx_x_SS, ldzx);
    blas::lacpy( 'A', nbe, npts, dbasis_x_eval, nbe, Zy_x_SS, ldzy); 

    // SS y
    blas::lacpy( 'A', nbe, npts, dbasis_y_eval, nbe, Zs_y_SS, ldzs);
    blas::lacpy( 'A', nbe, npts, dbasis_y_eval, nbe, Zz_y_SS, ldzz);
    blas::lacpy( 'A', nbe, npts, dbasis_y_eval, nbe, Zx_y_SS, ldzx);
    blas::lacpy( 'A', nbe, npts, dbasis_y_eval, nbe, Zy_y_SS, ldzy); 

    // SS z
    blas::lacpy( 'A', nbe, npts, dbasis_z_eval, nbe, Zs_z_SS, ldzs);
    blas::lacpy( 'A', nbe, npts, dbasis_z_eval, nbe, Zz_z_SS, ldzz);
    blas::lacpy( 'A', nbe, npts, dbasis_z_eval, nbe, Zx_z_SS, ldzx);
    blas::lacpy( 'A', nbe, npts, dbasis_z_eval, nbe, Zy_z_SS, ldzy); 

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      auto* zs_col = Zs + i*ldzs;
      auto* zz_col = Zz + i*ldzz;
      auto* zx_col = Zx + i*ldzx;
      auto* zy_col = Zy + i*ldzy;

            // SS
      auto* zs_x_ss_col = Zs_x_SS + i*ldzs;
      auto* zz_x_ss_col = Zz_x_SS + i*ldzz;
      auto* zx_x_ss_col = Zx_x_SS + i*ldzx;
      auto* zy_x_ss_col = Zy_x_SS + i*ldzy;

      auto* zs_y_ss_col = Zs_y_SS + i*ldzs;
      auto* zz_y_ss_col = Zz_y_SS + i*ldzz;
      auto* zx_y_ss_col = Zx_y_SS + i*ldzx;
      auto* zy_y_ss_col = Zy_y_SS + i*ldzy;

      auto* zs_z_ss_col = Zs_z_SS + i*ldzs;
      auto* zz_z_ss_col = Zz_z_SS + i*ldzz;
      auto* zx_z_ss_col = Zx_z_SS + i*ldzx;
      auto* zy_z_ss_col = Zy_z_SS + i*ldzy;
      

      const double factp = 0.5 * vrho[2*i];
      const double factm = 0.5 * vrho[2*i+1];
      const double factor = 0.5 * (factp - factm);

      //eq. 56 https://doi.org/10.1140/epjb/e2018-90170-1
      GauXC::blas::scal( nbe, 0.5*(factp + factm), zs_col, 1 );
      GauXC::blas::scal( nbe, KZ[i] * factor, zz_col, 1 );
      GauXC::blas::scal( nbe, KX[i] * factor, zx_col, 1 );
      GauXC::blas::scal( nbe, KY[i] * factor, zy_col, 1 );

            // SS
      GauXC::blas::scal( nbe, 0.5*(factp + factm), zs_x_ss_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( nbe, KZ[i]*factor, zz_x_ss_col, 1 );
      GauXC::blas::scal( nbe, KX[i]*factor, zx_x_ss_col, 1 );
      GauXC::blas::scal( nbe, KY[i]*factor, zy_x_ss_col, 1 );

      GauXC::blas::scal( nbe, 0.5*(factp + factm), zs_y_ss_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( nbe, KZ[i]*factor, zz_y_ss_col, 1 );
      GauXC::blas::scal( nbe, KX[i]*factor, zx_y_ss_col, 1 );
      GauXC::blas::scal( nbe, KY[i]*factor, zy_y_ss_col, 1 );

      GauXC::blas::scal( nbe, 0.5*(factp + factm), zs_z_ss_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( nbe, KZ[i]*factor, zz_z_ss_col, 1 );
      GauXC::blas::scal( nbe, KX[i]*factor, zx_z_ss_col, 1 );
      GauXC::blas::scal( nbe, KY[i]*factor, zy_z_ss_col, 1 );
   
    }

}

  // Eval Z Matrix GGA VXC
  void ReferenceLocalHostWorkDriver::eval_zmat_gga_vxc_rks( size_t npts, size_t nbf,
							const double* vrho, const double* vgamma, const double* basis_eval,
							const double* dbasis_x_eval, const double* dbasis_y_eval,
							const double* dbasis_z_eval, const double* dden_x_eval,
							const double* dden_y_eval, const double* dden_z_eval, double* Z, size_t ldz ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);

    if( ldz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Z, inbf );

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * inbf;

      auto* z_col    = Z + ioff;
      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;

      const auto lda_fact = 0.5 * vrho[i];
      blas::scal( inbf, lda_fact, z_col, 1 );

      const auto gga_fact = 2. * vgamma[i];
      const auto x_fact = gga_fact * dden_x_eval[i];
      const auto y_fact = gga_fact * dden_y_eval[i];
      const auto z_fact = gga_fact * dden_z_eval[i];

      blas::axpy( inbf, x_fact, bf_x_col, 1, z_col, 1 );
      blas::axpy( inbf, y_fact, bf_y_col, 1, z_col, 1 );
      blas::axpy( inbf, z_fact, bf_z_col, 1, z_col, 1 );

    }

  }

  void ReferenceLocalHostWorkDriver::eval_zmat_gga_vxc_uks( size_t npts, size_t nbf,
              const double* vrho, const double* vgamma, const double* basis_eval,
              const double* dbasis_x_eval, const double* dbasis_y_eval,
              const double* dbasis_z_eval, const double* dden_x_eval,
              const double* dden_y_eval, const double* dden_z_eval, double* Zs,
              size_t ldzs, double* Zz, size_t ldzz ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildzs = static_cast<int32_t>(ldzs);
    const auto ildzz = static_cast<int32_t>(ldzz);

    if( ldzs != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zs, ildzs);
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zz, ildzz);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * inbf;

      auto* zs_col = Zs + ioff;
      auto* zz_col = Zz + ioff;
      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;

      const double factp = 0.5 * vrho[2*i];
      const double factm = 0.5 * vrho[2*i+1];

      GauXC::blas::scal( inbf, 0.5*(factp + factm), zs_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( inbf, 0.5*(factp - factm), zz_col, 1 );

      const auto gga_fact_pp = vgamma[3*i];
      const auto gga_fact_pm = vgamma[3*i+1];
      const auto gga_fact_mm = vgamma[3*i+2];

      const auto gga_fact_1 = 0.5*(gga_fact_pp + gga_fact_pm + gga_fact_mm);
      const auto gga_fact_2 = 0.5*(gga_fact_pp - gga_fact_mm);
      const auto gga_fact_3 = 0.5*(gga_fact_pp - gga_fact_pm + gga_fact_mm);

      const auto x_fact_s = gga_fact_1 * dden_x_eval[2*i] + gga_fact_2 * dden_x_eval[2*i+1];
      const auto y_fact_s = gga_fact_1 * dden_y_eval[2*i] + gga_fact_2 * dden_y_eval[2*i+1];
      const auto z_fact_s = gga_fact_1 * dden_z_eval[2*i] + gga_fact_2 * dden_z_eval[2*i+1];

      const auto x_fact_z = gga_fact_3 * dden_x_eval[2*i+1] + gga_fact_2 * dden_x_eval[2*i];
      const auto y_fact_z = gga_fact_3 * dden_y_eval[2*i+1] + gga_fact_2 * dden_y_eval[2*i];
      const auto z_fact_z = gga_fact_3 * dden_z_eval[2*i+1] + gga_fact_2 * dden_z_eval[2*i];

      blas::axpy( inbf, x_fact_s, bf_x_col, 1, zs_col, 1 );
      blas::axpy( inbf, y_fact_s, bf_y_col, 1, zs_col, 1 );
      blas::axpy( inbf, z_fact_s, bf_z_col, 1, zs_col, 1 );

      blas::axpy( inbf, x_fact_z, bf_x_col, 1, zz_col, 1 );
      blas::axpy( inbf, y_fact_z, bf_y_col, 1, zz_col, 1 );
      blas::axpy( inbf, z_fact_z, bf_z_col, 1, zz_col, 1 );

    }
  }

  // Eval Z Matrix MGGA VXC
  void ReferenceLocalHostWorkDriver::eval_zmat_mgga_vxc_rks( size_t npts, size_t nbf,
              const double* vrho, const double* vgamma, const double* vlapl,
              const double* basis_eval,
              const double* dbasis_x_eval, const double* dbasis_y_eval,
              const double* dbasis_z_eval, const double* lbasis_eval,
              const double* dden_x_eval,
              const double* dden_y_eval, const double* dden_z_eval, double* Z, size_t ldz ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);

    if( ldz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Z, inbf );

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * inbf;

      auto* z_col    = Z + ioff;
      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;

      const auto lda_fact = 0.5 * vrho[i];
      blas::scal( inbf, lda_fact, z_col, 1 );

      const auto gga_fact = 2. * vgamma[i];
      const auto x_fact = gga_fact * dden_x_eval[i];
      const auto y_fact = gga_fact * dden_y_eval[i];
      const auto z_fact = gga_fact * dden_z_eval[i];

      blas::axpy( inbf, x_fact, bf_x_col, 1, z_col, 1 );
      blas::axpy( inbf, y_fact, bf_y_col, 1, z_col, 1 );
      blas::axpy( inbf, z_fact, bf_z_col, 1, z_col, 1 );

      if ( vlapl != nullptr ) {
  auto* lbf_col = lbasis_eval + ioff;
        const auto lapl_fact = vlapl[i];
        blas::axpy( inbf, lapl_fact, lbf_col, 1, z_col, 1 );
      }

    }

  }

void ReferenceLocalHostWorkDriver::eval_zmat_mgga_vxc_uks( size_t npts, size_t nbf,
              const double* vrho, const double* vgamma, const double* vlapl,
        const double* basis_eval,
              const double* dbasis_x_eval, const double* dbasis_y_eval,
              const double* dbasis_z_eval, const double* lbasis_eval,
        const double* dden_x_eval,
              const double* dden_y_eval, const double* dden_z_eval, double* Zs,
              size_t ldzs, double* Zz, size_t ldzz ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildzs = static_cast<int32_t>(ldzs);
    const auto ildzz = static_cast<int32_t>(ldzz);

    if( ldzs != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zs, ildzs);
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zz, ildzz);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * inbf;

      auto* zs_col = Zs + ioff;
      auto* zz_col = Zz + ioff;
      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;
      auto* lbf_col = lbasis_eval + ioff;

      const double factp = 0.5 * vrho[2*i];
      const double factm = 0.5 * vrho[2*i+1];

      GauXC::blas::scal( inbf, 0.5*(factp + factm), zs_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( inbf, 0.5*(factp - factm), zz_col, 1 );

      const auto gga_fact_pp = vgamma[3*i];
      const auto gga_fact_pm = vgamma[3*i+1];
      const auto gga_fact_mm = vgamma[3*i+2];

      const auto gga_fact_1 = 0.5*(gga_fact_pp + gga_fact_pm + gga_fact_mm);
      const auto gga_fact_2 = 0.5*(gga_fact_pp - gga_fact_mm);
      const auto gga_fact_3 = 0.5*(gga_fact_pp - gga_fact_pm + gga_fact_mm);

      const auto x_fact_s = gga_fact_1 * dden_x_eval[2*i] + gga_fact_2 * dden_x_eval[2*i+1];
      const auto y_fact_s = gga_fact_1 * dden_y_eval[2*i] + gga_fact_2 * dden_y_eval[2*i+1];
      const auto z_fact_s = gga_fact_1 * dden_z_eval[2*i] + gga_fact_2 * dden_z_eval[2*i+1];

      const auto x_fact_z = gga_fact_3 * dden_x_eval[2*i+1] + gga_fact_2 * dden_x_eval[2*i];
      const auto y_fact_z = gga_fact_3 * dden_y_eval[2*i+1] + gga_fact_2 * dden_y_eval[2*i];
      const auto z_fact_z = gga_fact_3 * dden_z_eval[2*i+1] + gga_fact_2 * dden_z_eval[2*i];


      blas::axpy( inbf, x_fact_s, bf_x_col, 1, zs_col, 1 );
      blas::axpy( inbf, y_fact_s, bf_y_col, 1, zs_col, 1 );
      blas::axpy( inbf, z_fact_s, bf_z_col, 1, zs_col, 1 );

      blas::axpy( inbf, x_fact_z, bf_x_col, 1, zz_col, 1 );
      blas::axpy( inbf, y_fact_z, bf_y_col, 1, zz_col, 1 );
      blas::axpy( inbf, z_fact_z, bf_z_col, 1, zz_col, 1 );

      if (vlapl != nullptr) {
        const auto lfactp = vlapl[2*i];
        const auto lfactm = vlapl[2*i+1];
        blas::axpy( inbf, 0.5*(lfactp + lfactm), lbf_col, 1, zs_col, 1);
        blas::axpy( inbf, 0.5*(lfactp - lfactm), lbf_col, 1, zz_col, 1);
      }

    }
  }

  void ReferenceLocalHostWorkDriver::eval_mmat_mgga_vxc_rks(size_t npts, size_t nbf,
              const double* vtau, const double* vlapl,
              const double* dbasis_x_eval, const double* dbasis_y_eval,
              const double* dbasis_z_eval,
              double* mmat_x, double* mmat_y, double* mmat_z, size_t ldm ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildm  = static_cast<int32_t>(ldm);

    if( ldm != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));

    blas::lacpy( 'A', inbf, inpts, dbasis_x_eval, inbf, mmat_x, ildm);
    blas::lacpy( 'A', inbf, inpts, dbasis_y_eval, inbf, mmat_y, ildm);
    blas::lacpy( 'A', inbf, inpts, dbasis_z_eval, inbf, mmat_z, ildm);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * inbf;
      auto* mmat_x_col = mmat_x + ioff;
      auto* mmat_y_col = mmat_y + ioff;
      auto* mmat_z_col = mmat_z + ioff;
      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;

      const auto tfact = 0.25 * vtau[i];

      blas::scal( inbf, tfact, mmat_x_col, 1);
      blas::scal( inbf, tfact, mmat_y_col, 1);
      blas::scal( inbf, tfact, mmat_z_col, 1);

      if ( vlapl != nullptr ) {
        const auto lfact = vlapl[i];
        blas::axpy( inbf, lfact, bf_x_col, 1, mmat_x_col, 1);
        blas::axpy( inbf, lfact, bf_y_col, 1, mmat_y_col, 1);
        blas::axpy( inbf, lfact, bf_z_col, 1, mmat_z_col, 1);
      }
    }
  }

void ReferenceLocalHostWorkDriver::eval_mmat_mgga_vxc_uks(size_t npts, size_t nbf,
              const double* vtau, const double* vlapl,
              const double* dbasis_x_eval, const double* dbasis_y_eval,
              const double* dbasis_z_eval,
              double* mmat_xs, double* mmat_ys, double* mmat_zs, size_t ldms,
              double* mmat_xz, double* mmat_yz, double* mmat_zz, size_t ldmz) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildms = static_cast<int32_t>(ldms);
    const auto ildmz = static_cast<int32_t>(ldmz);

    if( ldms != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldmz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));

    blas::lacpy( 'A', inbf, inpts, dbasis_x_eval, inbf, mmat_xs, ildms);
    blas::lacpy( 'A', inbf, inpts, dbasis_y_eval, inbf, mmat_ys, ildms);
    blas::lacpy( 'A', inbf, inpts, dbasis_z_eval, inbf, mmat_zs, ildms);
    blas::lacpy( 'A', inbf, inpts, dbasis_x_eval, inbf, mmat_xz, ildmz);
    blas::lacpy( 'A', inbf, inpts, dbasis_y_eval, inbf, mmat_yz, ildmz);
    blas::lacpy( 'A', inbf, inpts, dbasis_z_eval, inbf, mmat_zz, ildmz);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * inbf;
      auto* xs_col = mmat_xs + ioff;
      auto* ys_col = mmat_ys + ioff;
      auto* zs_col = mmat_zs + ioff;
      auto* xz_col = mmat_xz + ioff;
      auto* yz_col = mmat_yz + ioff;
      auto* zz_col = mmat_zz + ioff;
      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;

      const auto tfactp = 0.25 * vtau[2*i];
      const auto tfactm = 0.25 * vtau[2*i+1];
      const auto tfacts = 0.5*(tfactp + tfactm);
      const auto tfactz = 0.5*(tfactp - tfactm);

      blas::scal( inbf, tfacts, xs_col, 1);
      blas::scal( inbf, tfacts, ys_col, 1);
      blas::scal( inbf, tfacts, zs_col, 1);
      blas::scal( inbf, tfactz, xz_col, 1);
      blas::scal( inbf, tfactz, yz_col, 1);
      blas::scal( inbf, tfactz, zz_col, 1);

      if ( vlapl != nullptr ) {
        const auto lfactp = vlapl[2*i];
        const auto lfactm = vlapl[2*i+1];
  const auto lfacts = 0.5*(lfactp + lfactm);
  const auto lfactz = 0.5*(lfactp - lfactm);
        blas::axpy( inbf, lfacts, bf_x_col, 1, xs_col, 1);
        blas::axpy( inbf, lfacts, bf_y_col, 1, ys_col, 1);
        blas::axpy( inbf, lfacts, bf_z_col, 1, zs_col, 1);
        blas::axpy( inbf, lfactz, bf_x_col, 1, xz_col, 1);
        blas::axpy( inbf, lfactz, bf_y_col, 1, yz_col, 1);
        blas::axpy( inbf, lfactz, bf_z_col, 1, zz_col, 1);
      }

    }
  }


void ReferenceLocalHostWorkDriver::eval_zmat_gga_vxc_gks( size_t npts, size_t nbf, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz, double* Zx, size_t ldzx,
    double* Zy, size_t ldzy, double* K, double* H ) {

    const auto inbf  = static_cast<int32_t>(nbf);
    const auto inpts = static_cast<int32_t>(npts);
    const auto ildzs = static_cast<int32_t>(ldzs);
    const auto ildzz = static_cast<int32_t>(ldzz);
    const auto ildzx = static_cast<int32_t>(ldzx);
    const auto ildzy = static_cast<int32_t>(ldzy);

    auto *KZ = K; // KZ // store K in the Z matrix
    auto *KY = KZ + npts;
    auto *KX = KY + npts;

    auto *HZ = H; // KZ // store K in the Z matrix
    auto *HY = HZ + npts;
    auto *HX = HY + npts;

    if( ldzs != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzx != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzy != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));

    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zs, ildzs);
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zz, ildzz);
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zx, ildzx);
    blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zy, ildzy);

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * inbf;

      auto* zs_col = Zs + ioff;
      auto* zz_col = Zz + ioff;
      auto* zx_col = Zx + ioff;
      auto* zy_col = Zy + ioff;

      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;

      const double factp = 0.5 * vrho[2*i];
      const double factm = 0.5 * vrho[2*i+1];
      const double factor = 0.5 * (factp - factm);

      GauXC::blas::scal( inbf, 0.5*(factp + factm), zs_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( inbf, KZ[i]*factor, zz_col, 1 );
      GauXC::blas::scal( inbf, KX[i]*factor, zx_col, 1 );
      GauXC::blas::scal( inbf, KY[i]*factor, zy_col, 1 );

      const auto gga_fact_pp = vgamma[3 * i];
      const auto gga_fact_pm = vgamma[3 * i + 1];
      const auto gga_fact_mm = vgamma[3 * i + 2];

      const auto gga_fact_1 = 0.5 * (gga_fact_pp + gga_fact_pm + gga_fact_mm);
      const auto gga_fact_2 = 0.5 * (gga_fact_pp - gga_fact_mm);
      const auto gga_fact_3 = 0.5 * (gga_fact_pp - gga_fact_pm + gga_fact_mm);

      const auto x_fact_s = gga_fact_1 * dden_x_eval[4 * i] +
                            gga_fact_2 * (HZ[i] * dden_x_eval[4 * i + 1] +
                                          HY[i] * dden_x_eval[4 * i + 2] +
                                          HX[i] * dden_x_eval[4 * i + 3]);
      const auto y_fact_s = gga_fact_1 * dden_y_eval[4 * i] +
                            gga_fact_2 * (HZ[i] * dden_y_eval[4 * i + 1] +
                                          HY[i] * dden_y_eval[4 * i + 2] +
                                          HX[i] * dden_y_eval[4 * i + 3]);
      const auto z_fact_s = gga_fact_1 * dden_z_eval[4 * i] +
                            gga_fact_2 * (HZ[i] * dden_z_eval[4 * i + 1] +
                                          HY[i] * dden_z_eval[4 * i + 2] +
                                          HX[i] * dden_z_eval[4 * i + 3]);

      const auto x_fact_z = gga_fact_3 * dden_x_eval[4 * i + 1] +
                            gga_fact_2 * HZ[i] * dden_x_eval[4 * i];
      const auto y_fact_z = gga_fact_3 * dden_y_eval[4 * i + 1] +
                            gga_fact_2 * HZ[i] * dden_y_eval[4 * i];
      const auto z_fact_z = gga_fact_3 * dden_z_eval[4 * i + 1] +
                            gga_fact_2 * HZ[i] * dden_z_eval[4 * i];

      const auto x_fact_x = gga_fact_3 * dden_x_eval[4 * i + 3] +
                            gga_fact_2 * HX[i] * dden_x_eval[4 * i];
      const auto y_fact_x = gga_fact_3 * dden_y_eval[4 * i + 3] +
                            gga_fact_2 * HX[i] * dden_y_eval[4 * i];
      const auto z_fact_x = gga_fact_3 * dden_z_eval[4 * i + 3] +
                            gga_fact_2 * HX[i] * dden_z_eval[4 * i];

      const auto x_fact_y = gga_fact_3 * dden_x_eval[4 * i + 2] +
                            gga_fact_2 * HY[i] * dden_x_eval[4 * i];
      const auto y_fact_y = gga_fact_3 * dden_y_eval[4 * i + 2] +
                            gga_fact_2 * HY[i] * dden_y_eval[4 * i];
      const auto z_fact_y = gga_fact_3 * dden_z_eval[4 * i + 2] +
                            gga_fact_2 * HY[i] * dden_z_eval[4 * i];


      blas::axpy(inbf, x_fact_s, bf_x_col, 1, zs_col, 1);
      blas::axpy(inbf, y_fact_s, bf_y_col, 1, zs_col, 1);
      blas::axpy(inbf, z_fact_s, bf_z_col, 1, zs_col, 1);

      blas::axpy(inbf, x_fact_z, bf_x_col, 1, zz_col, 1);
      blas::axpy(inbf, y_fact_z, bf_y_col, 1, zz_col, 1);
      blas::axpy(inbf, z_fact_z, bf_z_col, 1, zz_col, 1);

      blas::axpy(inbf, x_fact_x, bf_x_col, 1, zx_col, 1);
      blas::axpy(inbf, y_fact_x, bf_y_col, 1, zx_col, 1);
      blas::axpy(inbf, z_fact_x, bf_z_col, 1, zx_col, 1);

      blas::axpy(inbf, x_fact_y, bf_x_col, 1, zy_col, 1);
      blas::axpy(inbf, y_fact_y, bf_y_col, 1, zy_col, 1);
      blas::axpy(inbf, z_fact_y, bf_z_col, 1, zy_col, 1);

    }

}

void ReferenceLocalHostWorkDriver::eval_zmat_gga_vxc_dks( size_t npts, size_t nbf, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* d2basis_xx_eval, const double* d2basis_xy_eval, const double* d2basis_xz_eval, 
    const double* d2basis_yy_eval, const double* d2basis_yz_eval, const double* d2basis_zz_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz, double* Zx, size_t ldzx, double* Zy, size_t ldzy,
    double* Zs_x_SS, size_t ldzs_ss, double* Zz_x_SS, size_t ldzz_ss, 
    double* Zx_x_SS, size_t ldzx_ss, double* Zy_x_SS, size_t ldzy_ss, 
    double* Zs_y_SS, double* Zz_y_SS, 
    double* Zx_y_SS, double* Zy_y_SS, 
    double* Zs_z_SS, double* Zz_z_SS, 
    double* Zx_z_SS, double* Zy_z_SS,
    double* K, double* H ) {

    auto *KZ = K; // KZ // store K in the Z matrix
    auto *KY = KZ + npts;
    auto *KX = KY + npts;

    auto *HZ = H; // KZ // store K in the Z matrix
    auto *HY = HZ + npts;
    auto *HX = HY + npts;

    if( ldzs != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzx != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
    if( ldzy != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));

    blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zs, ldzs);
    blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zz, ldzz);
    blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zx, ldzx);
    blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zy, ldzy);   

    // SS x
    blas::lacpy( 'A', nbf, npts, dbasis_x_eval, nbf, Zs_x_SS, ldzs);
    blas::lacpy( 'A', nbf, npts, dbasis_x_eval, nbf, Zz_x_SS, ldzz);
    blas::lacpy( 'A', nbf, npts, dbasis_x_eval, nbf, Zx_x_SS, ldzx);
    blas::lacpy( 'A', nbf, npts, dbasis_x_eval, nbf, Zy_x_SS, ldzy); 

    // SS y
    blas::lacpy( 'A', nbf, npts, dbasis_y_eval, nbf, Zs_y_SS, ldzs);
    blas::lacpy( 'A', nbf, npts, dbasis_y_eval, nbf, Zz_y_SS, ldzz);
    blas::lacpy( 'A', nbf, npts, dbasis_y_eval, nbf, Zx_y_SS, ldzx);
    blas::lacpy( 'A', nbf, npts, dbasis_y_eval, nbf, Zy_y_SS, ldzy); 

    // SS z
    blas::lacpy( 'A', nbf, npts, dbasis_z_eval, nbf, Zs_z_SS, ldzs);
    blas::lacpy( 'A', nbf, npts, dbasis_z_eval, nbf, Zz_z_SS, ldzz);
    blas::lacpy( 'A', nbf, npts, dbasis_z_eval, nbf, Zx_z_SS, ldzx);
    blas::lacpy( 'A', nbf, npts, dbasis_z_eval, nbf, Zy_z_SS, ldzy); 

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      const int32_t ioff = i * nbf;

      auto* zs_col = Zs + ioff;
      auto* zz_col = Zz + ioff;
      auto* zx_col = Zx + ioff;
      auto* zy_col = Zy + ioff;

      // SS
      auto* zs_x_ss_col = Zs_x_SS + ioff;
      auto* zz_x_ss_col = Zz_x_SS + ioff;
      auto* zx_x_ss_col = Zx_x_SS + ioff;
      auto* zy_x_ss_col = Zy_x_SS + ioff;

      auto* zs_y_ss_col = Zs_y_SS + ioff;
      auto* zz_y_ss_col = Zz_y_SS + ioff;
      auto* zx_y_ss_col = Zx_y_SS + ioff;
      auto* zy_y_ss_col = Zy_y_SS + ioff;

      auto* zs_z_ss_col = Zs_z_SS + ioff;
      auto* zz_z_ss_col = Zz_z_SS + ioff;
      auto* zx_z_ss_col = Zx_z_SS + ioff;
      auto* zy_z_ss_col = Zy_z_SS + ioff;


      auto* bf_x_col = dbasis_x_eval + ioff;
      auto* bf_y_col = dbasis_y_eval + ioff;
      auto* bf_z_col = dbasis_z_eval + ioff;

      // SS
      auto* bf_xx_col = d2basis_xx_eval + ioff;
      auto* bf_xy_col = d2basis_xy_eval + ioff;
      auto* bf_xz_col = d2basis_xz_eval + ioff;

      auto* bf_yx_col = d2basis_xy_eval + ioff;
      auto* bf_yy_col = d2basis_yy_eval + ioff;
      auto* bf_yz_col = d2basis_yz_eval + ioff;

      auto* bf_zx_col = d2basis_xz_eval + ioff;
      auto* bf_zy_col = d2basis_yz_eval + ioff;
      auto* bf_zz_col = d2basis_zz_eval + ioff;

      const double factp = 0.5 * vrho[2*i];
      const double factm = 0.5 * vrho[2*i+1];
      const double factor = 0.5 * (factp - factm);

      GauXC::blas::scal( nbf, 0.5*(factp + factm), zs_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( nbf, KZ[i]*factor, zz_col, 1 );
      GauXC::blas::scal( nbf, KX[i]*factor, zx_col, 1 );
      GauXC::blas::scal( nbf, KY[i]*factor, zy_col, 1 );

      // SS
      GauXC::blas::scal( nbf, 0.5*(factp + factm), zs_x_ss_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( nbf, KZ[i]*factor, zz_x_ss_col, 1 );
      GauXC::blas::scal( nbf, KX[i]*factor, zx_x_ss_col, 1 );
      GauXC::blas::scal( nbf, KY[i]*factor, zy_x_ss_col, 1 );

      GauXC::blas::scal( nbf, 0.5*(factp + factm), zs_y_ss_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( nbf, KZ[i]*factor, zz_y_ss_col, 1 );
      GauXC::blas::scal( nbf, KX[i]*factor, zx_y_ss_col, 1 );
      GauXC::blas::scal( nbf, KY[i]*factor, zy_y_ss_col, 1 );

      GauXC::blas::scal( nbf, 0.5*(factp + factm), zs_z_ss_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
      GauXC::blas::scal( nbf, KZ[i]*factor, zz_z_ss_col, 1 );
      GauXC::blas::scal( nbf, KX[i]*factor, zx_z_ss_col, 1 );
      GauXC::blas::scal( nbf, KY[i]*factor, zy_z_ss_col, 1 );

      const auto gga_fact_pp = vgamma[3 * i];
      const auto gga_fact_pm = vgamma[3 * i + 1];
      const auto gga_fact_mm = vgamma[3 * i + 2];

      const auto gga_fact_1 = 0.5 * (gga_fact_pp + gga_fact_pm + gga_fact_mm);
      const auto gga_fact_2 = 0.5 * (gga_fact_pp - gga_fact_mm);
      const auto gga_fact_3 = 0.5 * (gga_fact_pp - gga_fact_pm + gga_fact_mm);

      const auto x_fact_s = gga_fact_1 * dden_x_eval[4 * i] +
                            gga_fact_2 * (HZ[i] * dden_x_eval[4 * i + 1] +
                                          HY[i] * dden_x_eval[4 * i + 2] +
                                          HX[i] * dden_x_eval[4 * i + 3]);
      const auto y_fact_s = gga_fact_1 * dden_y_eval[4 * i] +
                            gga_fact_2 * (HZ[i] * dden_y_eval[4 * i + 1] +
                                          HY[i] * dden_y_eval[4 * i + 2] +
                                          HX[i] * dden_y_eval[4 * i + 3]);
      const auto z_fact_s = gga_fact_1 * dden_z_eval[4 * i] +
                            gga_fact_2 * (HZ[i] * dden_z_eval[4 * i + 1] +
                                          HY[i] * dden_z_eval[4 * i + 2] +
                                          HX[i] * dden_z_eval[4 * i + 3]);

      const auto x_fact_z = gga_fact_3 * dden_x_eval[4 * i + 1] +
                            gga_fact_2 * HZ[i] * dden_x_eval[4 * i];
      const auto y_fact_z = gga_fact_3 * dden_y_eval[4 * i + 1] +
                            gga_fact_2 * HZ[i] * dden_y_eval[4 * i];
      const auto z_fact_z = gga_fact_3 * dden_z_eval[4 * i + 1] +
                            gga_fact_2 * HZ[i] * dden_z_eval[4 * i];

      const auto x_fact_x = gga_fact_3 * dden_x_eval[4 * i + 3] +
                            gga_fact_2 * HX[i] * dden_x_eval[4 * i];
      const auto y_fact_x = gga_fact_3 * dden_y_eval[4 * i + 3] +
                            gga_fact_2 * HX[i] * dden_y_eval[4 * i];
      const auto z_fact_x = gga_fact_3 * dden_z_eval[4 * i + 3] +
                            gga_fact_2 * HX[i] * dden_z_eval[4 * i];

      const auto x_fact_y = gga_fact_3 * dden_x_eval[4 * i + 2] +
                            gga_fact_2 * HY[i] * dden_x_eval[4 * i];
      const auto y_fact_y = gga_fact_3 * dden_y_eval[4 * i + 2] +
                            gga_fact_2 * HY[i] * dden_y_eval[4 * i];
      const auto z_fact_y = gga_fact_3 * dden_z_eval[4 * i + 2] +
                            gga_fact_2 * HY[i] * dden_z_eval[4 * i];

      // LL
      blas::axpy(nbf, x_fact_s, bf_x_col, 1, zs_col, 1);
      blas::axpy(nbf, y_fact_s, bf_y_col, 1, zs_col, 1);
      blas::axpy(nbf, z_fact_s, bf_z_col, 1, zs_col, 1);

      blas::axpy(nbf, x_fact_z, bf_x_col, 1, zz_col, 1);
      blas::axpy(nbf, y_fact_z, bf_y_col, 1, zz_col, 1);
      blas::axpy(nbf, z_fact_z, bf_z_col, 1, zz_col, 1);

      blas::axpy(nbf, x_fact_x, bf_x_col, 1, zx_col, 1);
      blas::axpy(nbf, y_fact_x, bf_y_col, 1, zx_col, 1);
      blas::axpy(nbf, z_fact_x, bf_z_col, 1, zx_col, 1);

      blas::axpy(nbf, x_fact_y, bf_x_col, 1, zy_col, 1);
      blas::axpy(nbf, y_fact_y, bf_y_col, 1, zy_col, 1);
      blas::axpy(nbf, z_fact_y, bf_z_col, 1, zy_col, 1);

      // SS dX
      blas::axpy(nbf, x_fact_s, bf_xx_col, 1, zs_x_ss_col, 1);
      blas::axpy(nbf, y_fact_s, bf_yx_col, 1, zs_x_ss_col, 1);
      blas::axpy(nbf, z_fact_s, bf_zx_col, 1, zs_x_ss_col, 1);

      blas::axpy(nbf, x_fact_z, bf_xx_col, 1, zz_x_ss_col, 1);
      blas::axpy(nbf, y_fact_z, bf_yx_col, 1, zz_x_ss_col, 1);
      blas::axpy(nbf, z_fact_z, bf_zx_col, 1, zz_x_ss_col, 1);

      blas::axpy(nbf, x_fact_x, bf_xx_col, 1, zx_x_ss_col, 1);
      blas::axpy(nbf, y_fact_x, bf_yx_col, 1, zx_x_ss_col, 1);
      blas::axpy(nbf, z_fact_x, bf_zx_col, 1, zx_x_ss_col, 1);

      blas::axpy(nbf, x_fact_y, bf_xx_col, 1, zy_x_ss_col, 1);
      blas::axpy(nbf, y_fact_y, bf_yx_col, 1, zy_x_ss_col, 1);
      blas::axpy(nbf, z_fact_y, bf_zx_col, 1, zy_x_ss_col, 1);

      // SS dY
      blas::axpy(nbf, x_fact_s, bf_xy_col, 1, zs_y_ss_col, 1);
      blas::axpy(nbf, y_fact_s, bf_yy_col, 1, zs_y_ss_col, 1);
      blas::axpy(nbf, z_fact_s, bf_zy_col, 1, zs_y_ss_col, 1);

      blas::axpy(nbf, x_fact_z, bf_xy_col, 1, zz_y_ss_col, 1);
      blas::axpy(nbf, y_fact_z, bf_yy_col, 1, zz_y_ss_col, 1);
      blas::axpy(nbf, z_fact_z, bf_zy_col, 1, zz_y_ss_col, 1);

      blas::axpy(nbf, x_fact_x, bf_xy_col, 1, zx_y_ss_col, 1);
      blas::axpy(nbf, y_fact_x, bf_yy_col, 1, zx_y_ss_col, 1);
      blas::axpy(nbf, z_fact_x, bf_zy_col, 1, zx_y_ss_col, 1);

      blas::axpy(nbf, x_fact_y, bf_xy_col, 1, zy_y_ss_col, 1);
      blas::axpy(nbf, y_fact_y, bf_yy_col, 1, zy_y_ss_col, 1);
      blas::axpy(nbf, z_fact_y, bf_zy_col, 1, zy_y_ss_col, 1);

      // SS dZ
      blas::axpy(nbf, x_fact_s, bf_xz_col, 1, zs_z_ss_col, 1);
      blas::axpy(nbf, y_fact_s, bf_yz_col, 1, zs_z_ss_col, 1);
      blas::axpy(nbf, z_fact_s, bf_zz_col, 1, zs_z_ss_col, 1);

      blas::axpy(nbf, x_fact_z, bf_xz_col, 1, zz_z_ss_col, 1);
      blas::axpy(nbf, y_fact_z, bf_yz_col, 1, zz_z_ss_col, 1);
      blas::axpy(nbf, z_fact_z, bf_zz_col, 1, zz_z_ss_col, 1);

      blas::axpy(nbf, x_fact_x, bf_xz_col, 1, zx_z_ss_col, 1);
      blas::axpy(nbf, y_fact_x, bf_yz_col, 1, zx_z_ss_col, 1);
      blas::axpy(nbf, z_fact_x, bf_zz_col, 1, zx_z_ss_col, 1);

      blas::axpy(nbf, x_fact_y, bf_xz_col, 1, zy_z_ss_col, 1);
      blas::axpy(nbf, y_fact_y, bf_yz_col, 1, zy_z_ss_col, 1);
      blas::axpy(nbf, z_fact_y, bf_zz_col, 1, zy_z_ss_col, 1);

    }
}

void ReferenceLocalHostWorkDriver::eval_tmat_lda_vxc_rks( size_t npts, const double* v2rho2, const double* trho, double* A){
	for( int32_t i = 0; i < (int32_t)npts; ++i ) 
		A[i] = v2rho2[i] * trho[i];
}

void ReferenceLocalHostWorkDriver::eval_tmat_lda_vxc_uks( size_t npts, const double* v2rho2, const double* trho, double* A){
	for( int32_t i = 0; i < (int32_t)npts; ++i ) {
		A[2*i] = v2rho2[3*i] * trho[2*i] + v2rho2[3*i+1] * trho[2*i+1];
		A[2*i+1] = v2rho2[3*i+1] * trho[2*i] + v2rho2[3*i+2] * trho[2*i+1];
	}
}

void ReferenceLocalHostWorkDriver::eval_tmat_gga_vxc_rks( size_t npts, const double* vgamma, 
  const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
  const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B ){

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {


    //calculate trial gamma
    const auto tgamma = tdden_x_eval[i] * dden_x_eval[i] + tdden_y_eval[i] * dden_y_eval[i] + tdden_z_eval[i] * dden_z_eval[i];

    A[i] = v2rho2[i] * trho[i] + 2 * v2rhogamma[i] * tgamma;

    auto B_coef = v2rhogamma[i] * trho[i] + 2 * v2gamma2[i] * tgamma;

    B[i * 3]     = 2 * B_coef * dden_x_eval[i] + 2 * vgamma[i] * tdden_x_eval[i];
    B[i * 3 + 1] = 2 * B_coef * dden_y_eval[i] + 2 * vgamma[i] * tdden_y_eval[i];
    B[i * 3 + 2] = 2 * B_coef * dden_z_eval[i] + 2 * vgamma[i] * tdden_z_eval[i];

  }
}


void ReferenceLocalHostWorkDriver::eval_tmat_gga_vxc_uks( size_t npts, const double* vgamma, 
  const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
  const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B ){

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    // convert dden_x_eval, dden_y_eval, dden_z_eval to two-spinor representation
    const auto dden_x_eval_a = 0.5 * (dden_x_eval[2*i] + dden_x_eval[2*i+1]);
    const auto dden_x_eval_b = 0.5 * (dden_x_eval[2*i] - dden_x_eval[2*i+1]);
    const auto dden_y_eval_a = 0.5 * (dden_y_eval[2*i] + dden_y_eval[2*i+1]);
    const auto dden_y_eval_b = 0.5 * (dden_y_eval[2*i] - dden_y_eval[2*i+1]);
    const auto dden_z_eval_a = 0.5 * (dden_z_eval[2*i] + dden_z_eval[2*i+1]);
    const auto dden_z_eval_b = 0.5 * (dden_z_eval[2*i] - dden_z_eval[2*i+1]);
    // convert tdden_x_eval, tdden_y_eval, tdden_z_eval to two-spinor representation
    const auto tdden_x_eval_a = 0.5 * (tdden_x_eval[2*i] + tdden_x_eval[2*i+1]);
    const auto tdden_x_eval_b = 0.5 * (tdden_x_eval[2*i] - tdden_x_eval[2*i+1]);
    const auto tdden_y_eval_a = 0.5 * (tdden_y_eval[2*i] + tdden_y_eval[2*i+1]);
    const auto tdden_y_eval_b = 0.5 * (tdden_y_eval[2*i] - tdden_y_eval[2*i+1]);
    const auto tdden_z_eval_a = 0.5 * (tdden_z_eval[2*i] + tdden_z_eval[2*i+1]);
    const auto tdden_z_eval_b = 0.5 * (tdden_z_eval[2*i] - tdden_z_eval[2*i+1]);

    //calculate trial gamma
    const auto tgamma_aa = tdden_x_eval_a * dden_x_eval_a + tdden_y_eval_a * dden_y_eval_a + tdden_z_eval_a * dden_z_eval_a;
    const auto tgamma_ab = tdden_x_eval_a * dden_x_eval_b + tdden_y_eval_a * dden_y_eval_b + tdden_z_eval_a * dden_z_eval_b
                        + tdden_x_eval_b * dden_x_eval_a + tdden_y_eval_b * dden_y_eval_a + tdden_z_eval_b * dden_z_eval_a;
    const auto tgamma_bb = tdden_x_eval_b * dden_x_eval_b + tdden_y_eval_b * dden_y_eval_b + tdden_z_eval_b * dden_z_eval_b;
    const auto trho_a = trho[2*i];
    const auto trho_b = trho[2*i+1];

    const auto v2rho2_a_a = v2rho2[3*i];
    const auto v2rho2_a_b = v2rho2[3*i+1];
    const auto v2rho2_b_b = v2rho2[3*i+2];
    const auto v2rhogamma_a_aa = v2rhogamma[6*i];
    const auto v2rhogamma_a_ab = v2rhogamma[6*i+1];
    const auto v2rhogamma_a_bb = v2rhogamma[6*i+2];
    const auto v2rhogamma_b_aa = v2rhogamma[6*i+3];
    const auto v2rhogamma_b_ab = v2rhogamma[6*i+4];
    const auto v2rhogamma_b_bb = v2rhogamma[6*i+5];
    const auto v2gamma2_aa_aa = v2gamma2[6*i];
    const auto v2gamma2_aa_ab = v2gamma2[6*i+1];
    const auto v2gamma2_aa_bb = v2gamma2[6*i+2];
    const auto v2gamma2_ab_ab = v2gamma2[6*i+3];
    const auto v2gamma2_ab_bb = v2gamma2[6*i+4];
    const auto v2gamma2_bb_bb = v2gamma2[6*i+5];
    const auto vgamma_aa = vgamma[3*i];
    const auto vgamma_ab = vgamma[3*i+1];
    const auto vgamma_bb = vgamma[3*i+2];

    A[2 * i] = v2rho2_a_a * trho_a + 2 * v2rhogamma_a_aa * tgamma_aa + v2rhogamma_a_ab * tgamma_ab +
             v2rho2_a_b * trho_b + 2 * v2rhogamma_a_bb * tgamma_bb;
    A[2 * i + 1] = v2rho2_b_b * trho_b + 2 * v2rhogamma_b_bb * tgamma_bb + v2rhogamma_b_ab * tgamma_ab +
             v2rho2_a_b * trho_a + 2 * v2rhogamma_b_aa * tgamma_aa;

    auto B_coef1 = v2rhogamma_a_aa * trho_a + 2 * v2gamma2_aa_aa * tgamma_aa + v2gamma2_aa_ab * tgamma_ab +
             v2rhogamma_b_aa * trho_b + 2 * v2gamma2_aa_bb * tgamma_bb;
    auto B_coef2 = v2rhogamma_a_ab * trho_a + 2 * v2gamma2_aa_ab * tgamma_aa + v2gamma2_ab_ab * tgamma_ab +
             v2rhogamma_b_ab * trho_b + 2 * v2gamma2_ab_bb * tgamma_bb;

    B[i * 6]     = 2 * B_coef1 * dden_x_eval_a + B_coef2 * dden_x_eval_b + 2 * vgamma_aa * tdden_x_eval_a + vgamma_ab * tdden_x_eval_b;
    B[i * 6 + 1] = 2 * B_coef1 * dden_y_eval_a + B_coef2 * dden_y_eval_b + 2 * vgamma_aa * tdden_y_eval_a + vgamma_ab * tdden_y_eval_b;
    B[i * 6 + 2] = 2 * B_coef1 * dden_z_eval_a + B_coef2 * dden_z_eval_b + 2 * vgamma_aa * tdden_z_eval_a + vgamma_ab * tdden_z_eval_b;

    B_coef1 = v2rhogamma_b_bb * trho_b + 2 * v2gamma2_bb_bb * tgamma_bb + v2gamma2_ab_bb * tgamma_ab +
             v2rhogamma_a_bb * trho_a + 2 * v2gamma2_aa_bb * tgamma_aa;
    B_coef2 = v2rhogamma_b_ab * trho_b + 2 * v2gamma2_ab_bb * tgamma_bb + v2gamma2_ab_ab * tgamma_ab +
             v2rhogamma_a_ab * trho_a + 2 * v2gamma2_aa_ab * tgamma_aa;

    B[i * 6 + 3] = 2 * B_coef1 * dden_x_eval_b + B_coef2 * dden_x_eval_a + 2 * vgamma_bb * tdden_x_eval_b + vgamma_ab * tdden_x_eval_a;
    B[i * 6 + 4] = 2 * B_coef1 * dden_y_eval_b + B_coef2 * dden_y_eval_a + 2 * vgamma_bb * tdden_y_eval_b + vgamma_ab * tdden_y_eval_a;
    B[i * 6 + 5] = 2 * B_coef1 * dden_z_eval_b + B_coef2 * dden_z_eval_a + 2 * vgamma_bb * tdden_z_eval_b + vgamma_ab * tdden_z_eval_a;
  }
}


void ReferenceLocalHostWorkDriver::eval_tmat_mgga_vxc_rks( size_t npts, const double* vgamma,
  const double* v2rho2, const double* v2rhogamma, [[maybe_unused]] const double* v2rholapl, const double* v2rhotau,
  const double* v2gamma2, [[maybe_unused]] const double* v2gammalapl, const double* v2gammatau,
  [[maybe_unused]] const double* v2lapl2, [[maybe_unused]] const double* v2lapltau, const double* v2tau2, 
  const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C){

    for( int32_t i = 0; i < (int32_t)npts; ++i ) {

      //calculate trial gamma
      const auto tgamma = tdden_x_eval[i] * dden_x_eval[i] + tdden_y_eval[i] * dden_y_eval[i] + tdden_z_eval[i] * dden_z_eval[i];
  
      A[i] = v2rho2[i] * trho[i] + 2 * v2rhogamma[i] * tgamma + v2rhotau[i] * ttau[i];
      C[i] = v2rhotau[i] * trho[i] + 2 * v2gammatau[i] * tgamma + v2tau2[i] * ttau[i];
  
      auto B_coef = v2rhogamma[i] * trho[i] + 2 * v2gamma2[i] * tgamma + v2gammatau[i] * ttau[i];
  
      B[i * 3]     = 2 * B_coef * dden_x_eval[i] + 2 * vgamma[i] * tdden_x_eval[i];
      B[i * 3 + 1] = 2 * B_coef * dden_y_eval[i] + 2 * vgamma[i] * tdden_y_eval[i];
      B[i * 3 + 2] = 2 * B_coef * dden_z_eval[i] + 2 * vgamma[i] * tdden_z_eval[i];
  
    }

}


void ReferenceLocalHostWorkDriver::eval_tmat_mgga_vxc_uks( size_t npts, const double* vgamma, 
  const double* v2rho2, const double* v2rhogamma, const double* v2rholapl, const double* v2rhotau, 
  const double* v2gamma2, const double* v2gammalapl, const double* v2gammatau,
  const double* v2lapl2, const double* v2lapltau, const double* v2tau2, 
  const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C){

  // Laplacian is not supported now
  if( v2rholapl != nullptr ||  v2gammalapl != nullptr ||  v2lapltau != nullptr ||  v2lapl2 != nullptr )
      GAUXC_GENERIC_EXCEPTION(std::string("Laplacian not supported"));

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    // convert dden_x_eval, dden_y_eval, dden_z_eval to two-spinor representation
    const auto dden_x_eval_a = 0.5 * (dden_x_eval[2*i] + dden_x_eval[2*i+1]);
    const auto dden_x_eval_b = 0.5 * (dden_x_eval[2*i] - dden_x_eval[2*i+1]);
    const auto dden_y_eval_a = 0.5 * (dden_y_eval[2*i] + dden_y_eval[2*i+1]);
    const auto dden_y_eval_b = 0.5 * (dden_y_eval[2*i] - dden_y_eval[2*i+1]);
    const auto dden_z_eval_a = 0.5 * (dden_z_eval[2*i] + dden_z_eval[2*i+1]);
    const auto dden_z_eval_b = 0.5 * (dden_z_eval[2*i] - dden_z_eval[2*i+1]);
    // convert tdden_x_eval, tdden_y_eval, tdden_z_eval to two-spinor representation
    const auto tdden_x_eval_a = 0.5 * (tdden_x_eval[2*i] + tdden_x_eval[2*i+1]);
    const auto tdden_x_eval_b = 0.5 * (tdden_x_eval[2*i] - tdden_x_eval[2*i+1]);
    const auto tdden_y_eval_a = 0.5 * (tdden_y_eval[2*i] + tdden_y_eval[2*i+1]);
    const auto tdden_y_eval_b = 0.5 * (tdden_y_eval[2*i] - tdden_y_eval[2*i+1]);
    const auto tdden_z_eval_a = 0.5 * (tdden_z_eval[2*i] + tdden_z_eval[2*i+1]);
    const auto tdden_z_eval_b = 0.5 * (tdden_z_eval[2*i] - tdden_z_eval[2*i+1]);

    //calculate trial gamma
    const auto tgamma_aa = tdden_x_eval_a * dden_x_eval_a + tdden_y_eval_a * dden_y_eval_a + tdden_z_eval_a * dden_z_eval_a;
    const auto tgamma_ab = tdden_x_eval_a * dden_x_eval_b + tdden_y_eval_a * dden_y_eval_b + tdden_z_eval_a * dden_z_eval_b
                         + tdden_x_eval_b * dden_x_eval_a + tdden_y_eval_b * dden_y_eval_a + tdden_z_eval_b * dden_z_eval_a;
    const auto tgamma_bb = tdden_x_eval_b * dden_x_eval_b + tdden_y_eval_b * dden_y_eval_b + tdden_z_eval_b * dden_z_eval_b;
    const auto trho_a = trho[2*i];
    const auto trho_b = trho[2*i+1];
    const auto ttau_a = ttau[2*i];
    const auto ttau_b = ttau[2*i+1];

    const auto v2rho2_a_a = v2rho2[3*i];
    const auto v2rho2_a_b = v2rho2[3*i+1];
    const auto v2rho2_b_b = v2rho2[3*i+2];
    const auto v2rhogamma_a_aa = v2rhogamma[6*i];
    const auto v2rhogamma_a_ab = v2rhogamma[6*i+1];
    const auto v2rhogamma_a_bb = v2rhogamma[6*i+2];
    const auto v2rhogamma_b_aa = v2rhogamma[6*i+3];
    const auto v2rhogamma_b_ab = v2rhogamma[6*i+4];
    const auto v2rhogamma_b_bb = v2rhogamma[6*i+5];
    const auto v2gamma2_aa_aa = v2gamma2[6*i];
    const auto v2gamma2_aa_ab = v2gamma2[6*i+1];
    const auto v2gamma2_aa_bb = v2gamma2[6*i+2];
    const auto v2gamma2_ab_ab = v2gamma2[6*i+3];
    const auto v2gamma2_ab_bb = v2gamma2[6*i+4];
    const auto v2gamma2_bb_bb = v2gamma2[6*i+5];
    const auto vgamma_aa = vgamma[3*i];
    const auto vgamma_ab = vgamma[3*i+1];
    const auto vgamma_bb = vgamma[3*i+2];
    const auto v2rhotau_a_a = v2rhotau[4*i];
    const auto v2rhotau_a_b = v2rhotau[4*i+1];
    const auto v2rhotau_b_a = v2rhotau[4*i+2];
    const auto v2rhotau_b_b = v2rhotau[4*i+3];
    const auto v2tau2_a_a = v2tau2[3*i];
    const auto v2tau2_a_b = v2tau2[3*i+1];
    const auto v2tau2_b_b = v2tau2[3*i+2];
    const auto v2gammatau_aa_a = v2gammatau[6*i];
    const auto v2gammatau_aa_b = v2gammatau[6*i+1];
    const auto v2gammatau_ab_a = v2gammatau[6*i+2];
    const auto v2gammatau_ab_b = v2gammatau[6*i+3];
    const auto v2gammatau_bb_a = v2gammatau[6*i+4];
    const auto v2gammatau_bb_b = v2gammatau[6*i+5];

  
    A[2 * i] =     v2rho2_a_a * trho_a + 2 * v2rhogamma_a_aa * tgamma_aa + v2rhogamma_a_ab * tgamma_ab + v2rhotau_a_a * ttau_a
                +  v2rho2_a_b * trho_b + 2 * v2rhogamma_a_bb * tgamma_bb + v2rhotau_a_b * ttau_b;
    A[2 * i + 1] = v2rho2_b_b * trho_b + 2 * v2rhogamma_b_bb * tgamma_bb + v2rhogamma_b_ab * tgamma_ab + v2rhotau_b_b * ttau_b
                +  v2rho2_a_b * trho_a + 2 * v2rhogamma_b_aa * tgamma_aa + v2rhotau_b_a * ttau_a;

    C[2 * i] =     v2rhotau_a_a * trho_a + 2 * v2gammatau_aa_a * tgamma_aa + v2gammatau_ab_a * tgamma_ab + v2tau2_a_a * ttau_a
                +  v2rhotau_b_a * trho_b + 2 * v2gammatau_bb_a * tgamma_bb + v2tau2_a_b * ttau_b;
    C[2 * i + 1] = v2rhotau_b_b * trho_b + 2 * v2gammatau_bb_b * tgamma_bb + v2gammatau_ab_b * tgamma_ab + v2tau2_b_b * ttau_b
                +  v2rhotau_a_b * trho_a + 2 * v2gammatau_aa_b * tgamma_aa + v2tau2_a_b * ttau_a;

    auto B_coef1 = v2rhogamma_a_aa * trho_a + 2 * v2gamma2_aa_aa * tgamma_aa + v2gamma2_aa_ab * tgamma_ab + v2gammatau_aa_a * ttau_a
                +  v2rhogamma_b_aa * trho_b + 2 * v2gamma2_aa_bb * tgamma_bb + v2gammatau_aa_b * ttau_b;
    auto B_coef2 = v2rhogamma_a_ab * trho_a + 2 * v2gamma2_aa_ab * tgamma_aa + v2gamma2_ab_ab * tgamma_ab + v2gammatau_ab_a * ttau_a
                +  v2rhogamma_b_ab * trho_b + 2 * v2gamma2_ab_bb * tgamma_bb + v2gammatau_ab_b * ttau_b;

    B[i * 6]     = 2 * B_coef1 * dden_x_eval_a + B_coef2 * dden_x_eval_b + 2 * vgamma_aa * tdden_x_eval_a + vgamma_ab * tdden_x_eval_b;
    B[i * 6 + 1] = 2 * B_coef1 * dden_y_eval_a + B_coef2 * dden_y_eval_b + 2 * vgamma_aa * tdden_y_eval_a + vgamma_ab * tdden_y_eval_b;
    B[i * 6 + 2] = 2 * B_coef1 * dden_z_eval_a + B_coef2 * dden_z_eval_b + 2 * vgamma_aa * tdden_z_eval_a + vgamma_ab * tdden_z_eval_b;

    B_coef1 = v2rhogamma_b_bb * trho_b + 2 * v2gamma2_bb_bb * tgamma_bb + v2gamma2_ab_bb * tgamma_ab + v2gammatau_bb_b * ttau_b
            + v2rhogamma_a_bb * trho_a + 2 * v2gamma2_aa_bb * tgamma_aa + v2gammatau_bb_a * ttau_a;
    B_coef2 = v2rhogamma_b_ab * trho_b + 2 * v2gamma2_ab_bb * tgamma_bb + v2gamma2_ab_ab * tgamma_ab + v2gammatau_ab_b * ttau_b
            + v2rhogamma_a_ab * trho_a + 2 * v2gamma2_aa_ab * tgamma_aa + v2gammatau_ab_a * ttau_a;

    B[i * 6 + 3] = 2 * B_coef1 * dden_x_eval_b + B_coef2 * dden_x_eval_a + 2 * vgamma_bb * tdden_x_eval_b + vgamma_ab * tdden_x_eval_a;
    B[i * 6 + 4] = 2 * B_coef1 * dden_y_eval_b + B_coef2 * dden_y_eval_a + 2 * vgamma_bb * tdden_y_eval_b + vgamma_ab * tdden_y_eval_a;
    B[i * 6 + 5] = 2 * B_coef1 * dden_z_eval_b + B_coef2 * dden_z_eval_a + 2 * vgamma_bb * tdden_z_eval_b + vgamma_ab * tdden_z_eval_a;

  }
}


// Eval Z Matrix LDA VXC for two-spinors
void ReferenceLocalHostWorkDriver::eval_zmat_lda_vxc_uks_ts( size_t npts, size_t nbf,
  const double* vrho, const double* basis_eval, double* Za, size_t ldza,
  double* Zb, size_t ldzb ) {

  const auto inbf  = static_cast<int32_t>(nbf);
  const auto inpts = static_cast<int32_t>(npts);
  const auto ildza = static_cast<int32_t>(ldza);
  const auto ildzb = static_cast<int32_t>(ldzb);

  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Za, ildza);
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zb, ildzb);
  for( int32_t i = 0; i < (int32_t)npts; ++i ) {
  //eq. 56 https://doi.org/10.1140/epjb/e2018-90170-1
  GauXC::blas::scal( inbf, 0.5 * vrho[2*i], Za + i*ldza, 1 );
  GauXC::blas::scal( inbf, 0.5 * vrho[2*i+1], Zb + i*ldzb, 1 );
  }
}

void ReferenceLocalHostWorkDriver::eval_Bvec_gga_vxc_rks_ts( size_t npts, const double* vgamma, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B ){

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {
    B[i*3]   = 2 * vgamma[i] * dden_x_eval[i];
    B[i*3+1] = 2 * vgamma[i] * dden_y_eval[i];
    B[i*3+2] = 2 * vgamma[i]* dden_z_eval[i]; 
  }
}

void ReferenceLocalHostWorkDriver::eval_zmat_gga_vxc_rks_ts( size_t npts, size_t nbf,
  const double* A, const double* B, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval,
  const double* dbasis_z_eval, double* Z,
  size_t ldz) {

  const auto inbf  = static_cast<int32_t>(nbf);
  const auto inpts = static_cast<int32_t>(npts);
  const auto ildz  = static_cast<int32_t>(ldz);

  if( ldz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Z, ildz);

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    const int32_t ioff = i * inbf;

    auto* z_col = Z + ioff;
    auto* bf_x_col = dbasis_x_eval + ioff;
    auto* bf_y_col = dbasis_y_eval + ioff;
    auto* bf_z_col = dbasis_z_eval + ioff;

    GauXC::blas::scal( inbf, 0.5*A[i], z_col, 1 );

    blas::axpy( inbf, B[i*3],   bf_x_col, 1, z_col, 1 );
    blas::axpy( inbf, B[i*3+1], bf_y_col, 1, z_col, 1 );
    blas::axpy( inbf, B[i*3+2], bf_z_col, 1, z_col, 1 );

  }
}


void ReferenceLocalHostWorkDriver::eval_Bvec_gga_vxc_uks_ts( size_t npts, const double* vgamma, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B ){


  for( int32_t i = 0; i < (int32_t)npts; ++i ) {
    const auto gga_fact_aa = vgamma[3*i];
    const auto gga_fact_ab = vgamma[3*i+1];
    const auto gga_fact_bb = vgamma[3*i+2];

    // dden_x_eval, dden_y_eval, dden_z_eval are all still in Pauli representation
    // so we need to convert them to the two spinor representation
    const auto dden_x_eval_a = 0.5 * (dden_x_eval[2*i] + dden_x_eval[2*i+1]);
    const auto dden_x_eval_b = 0.5 * (dden_x_eval[2*i] - dden_x_eval[2*i+1]);
    const auto dden_y_eval_a = 0.5 * (dden_y_eval[2*i] + dden_y_eval[2*i+1]);
    const auto dden_y_eval_b = 0.5 * (dden_y_eval[2*i] - dden_y_eval[2*i+1]);
    const auto dden_z_eval_a = 0.5 * (dden_z_eval[2*i] + dden_z_eval[2*i+1]);
    const auto dden_z_eval_b = 0.5 * (dden_z_eval[2*i] - dden_z_eval[2*i+1]);

    B[i*6]   = 2 * gga_fact_aa * dden_x_eval_a + gga_fact_ab * dden_x_eval_b;
    B[i*6+1] = 2 * gga_fact_aa * dden_y_eval_a + gga_fact_ab * dden_y_eval_b;
    B[i*6+2] = 2 * gga_fact_aa * dden_z_eval_a + gga_fact_ab * dden_z_eval_b;
    
    B[i*6+3] = 2 * gga_fact_bb * dden_x_eval_b + gga_fact_ab * dden_x_eval_a;
    B[i*6+4] = 2 * gga_fact_bb * dden_y_eval_b + gga_fact_ab * dden_y_eval_a;
    B[i*6+5] = 2 * gga_fact_bb * dden_z_eval_b + gga_fact_ab * dden_z_eval_a;
  }
}
void ReferenceLocalHostWorkDriver::eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbf,
  const double* A, const double* B, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval,
  const double* dbasis_z_eval, double* Za,
  size_t ldza, double* Zb, size_t ldzb ) {

  const auto inbf  = static_cast<int32_t>(nbf);
  const auto inpts = static_cast<int32_t>(npts);
  const auto ildza = static_cast<int32_t>(ldza);
  const auto ildzb = static_cast<int32_t>(ldzb);

  if( ldza != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  if( ldzb != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Za, ildza);
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zb, ildzb);

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    const int32_t ioff = i * inbf;

    auto* za_col = Za + ioff;
    auto* zb_col = Zb + ioff;
    auto* bf_x_col = dbasis_x_eval + ioff;
    auto* bf_y_col = dbasis_y_eval + ioff;
    auto* bf_z_col = dbasis_z_eval + ioff;

    GauXC::blas::scal( inbf, 0.5*A[2*i], za_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
    GauXC::blas::scal( inbf, 0.5*A[2*i+1], zb_col, 1 );

    blas::axpy( inbf, B[i*6],   bf_x_col, 1, za_col, 1 );
    blas::axpy( inbf, B[i*6+1], bf_y_col, 1, za_col, 1 );
    blas::axpy( inbf, B[i*6+2], bf_z_col, 1, za_col, 1 );

    blas::axpy( inbf, B[i*6+3], bf_x_col, 1, zb_col, 1 );
    blas::axpy( inbf, B[i*6+4], bf_y_col, 1, zb_col, 1 );
    blas::axpy( inbf, B[i*6+5], bf_z_col, 1, zb_col, 1 );

  }
}


void ReferenceLocalHostWorkDriver::eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbf,
  const double* vrho, const double* vgamma, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval,
  const double* dbasis_z_eval, const double* dden_x_eval,
  const double* dden_y_eval, const double* dden_z_eval, double* Za,
  size_t ldza, double* Zb, size_t ldzb ) {

  const auto inbf  = static_cast<int32_t>(nbf);
  const auto inpts = static_cast<int32_t>(npts);
  const auto ildza = static_cast<int32_t>(ldza);
  const auto ildzb = static_cast<int32_t>(ldzb);

  if( ldza != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  if( ldzb != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Za, ildza);
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zb, ildzb);

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    const int32_t ioff = i * inbf;

    auto* za_col = Za + ioff;
    auto* zb_col = Zb + ioff;
    auto* bf_x_col = dbasis_x_eval + ioff;
    auto* bf_y_col = dbasis_y_eval + ioff;
    auto* bf_z_col = dbasis_z_eval + ioff;

    GauXC::blas::scal( inbf, 0.5*vrho[2*i], za_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
    GauXC::blas::scal( inbf, 0.5*vrho[2*i+1], zb_col, 1 );

    const auto gga_fact_aa = vgamma[3*i];
    const auto gga_fact_ab = vgamma[3*i+1];
    const auto gga_fact_bb = vgamma[3*i+2];

    // dden_x_eval, dden_y_eval, dden_z_eval are all still in Pauli representation
    // so we need to convert them to the two spinor representation
    const auto dden_x_eval_a = 0.5 * (dden_x_eval[2*i] + dden_x_eval[2*i+1]);
    const auto dden_x_eval_b = 0.5 * (dden_x_eval[2*i] - dden_x_eval[2*i+1]);
    const auto dden_y_eval_a = 0.5 * (dden_y_eval[2*i] + dden_y_eval[2*i+1]);
    const auto dden_y_eval_b = 0.5 * (dden_y_eval[2*i] - dden_y_eval[2*i+1]);
    const auto dden_z_eval_a = 0.5 * (dden_z_eval[2*i] + dden_z_eval[2*i+1]);
    const auto dden_z_eval_b = 0.5 * (dden_z_eval[2*i] - dden_z_eval[2*i+1]);

    const auto x_fact_a = 2 * gga_fact_aa * dden_x_eval_a + gga_fact_ab * dden_x_eval_b;
    const auto y_fact_a = 2 * gga_fact_aa * dden_y_eval_a + gga_fact_ab * dden_y_eval_b;
    const auto z_fact_a = 2 * gga_fact_aa * dden_z_eval_a + gga_fact_ab * dden_z_eval_b;

    const auto x_fact_b = 2 * gga_fact_bb * dden_x_eval_b + gga_fact_ab * dden_x_eval_a;
    const auto y_fact_b = 2 * gga_fact_bb * dden_y_eval_b + gga_fact_ab * dden_y_eval_a;
    const auto z_fact_b = 2 * gga_fact_bb * dden_z_eval_b + gga_fact_ab * dden_z_eval_a;

    blas::axpy( inbf, x_fact_a, bf_x_col, 1, za_col, 1 );
    blas::axpy( inbf, y_fact_a, bf_y_col, 1, za_col, 1 );
    blas::axpy( inbf, z_fact_a, bf_z_col, 1, za_col, 1 );

    blas::axpy( inbf, x_fact_b, bf_x_col, 1, zb_col, 1 );
    blas::axpy( inbf, y_fact_b, bf_y_col, 1, zb_col, 1 );
    blas::axpy( inbf, z_fact_b, bf_z_col, 1, zb_col, 1 );

  }
}

void ReferenceLocalHostWorkDriver::eval_zmat_mgga_vxc_uks_ts( size_t npts, size_t nbf,
              const double* vrho, const double* vgamma, const double* vlapl,
        const double* basis_eval,
              const double* dbasis_x_eval, const double* dbasis_y_eval,
              const double* dbasis_z_eval, const double* lbasis_eval,
        const double* dden_x_eval,
              const double* dden_y_eval, const double* dden_z_eval, double* Za,
              size_t ldza, double* Zb, size_t ldzb ) {

  const auto inbf  = static_cast<int32_t>(nbf);
  const auto inpts = static_cast<int32_t>(npts);
  const auto ildza = static_cast<int32_t>(ldza);
  const auto ildzb = static_cast<int32_t>(ldzb);

  if( ldza != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  if( ldzb != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Za, ildza);
  blas::lacpy( 'A', inbf, inpts, basis_eval, inbf, Zb, ildzb);

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    const int32_t ioff = i * inbf;

    auto* za_col = Za + ioff;
    auto* zb_col = Zb + ioff;
    auto* bf_x_col = dbasis_x_eval + ioff;
    auto* bf_y_col = dbasis_y_eval + ioff;
    auto* bf_z_col = dbasis_z_eval + ioff;
    auto* lbf_col = lbasis_eval + ioff;

    GauXC::blas::scal( inbf, 0.5*vrho[2*i], za_col, 1 ); //additional 0.5 is from eq 56 in petrone 2018 eur phys journal b "an efficent implementation of .. "
    GauXC::blas::scal( inbf, 0.5*vrho[2*i+1], zb_col, 1 );

    // dden_x_eval, dden_y_eval, dden_z_eval are all still in Pauli representation
    // so we need to convert them to the two spinor representation
    const auto dden_x_eval_a = 0.5 * (dden_x_eval[2*i] + dden_x_eval[2*i+1]);
    const auto dden_x_eval_b = 0.5 * (dden_x_eval[2*i] - dden_x_eval[2*i+1]);
    const auto dden_y_eval_a = 0.5 * (dden_y_eval[2*i] + dden_y_eval[2*i+1]);
    const auto dden_y_eval_b = 0.5 * (dden_y_eval[2*i] - dden_y_eval[2*i+1]);
    const auto dden_z_eval_a = 0.5 * (dden_z_eval[2*i] + dden_z_eval[2*i+1]);
    const auto dden_z_eval_b = 0.5 * (dden_z_eval[2*i] - dden_z_eval[2*i+1]);

    const auto gga_fact_aa = vgamma[3*i];
    const auto gga_fact_ab = vgamma[3*i+1];
    const auto gga_fact_bb = vgamma[3*i+2];

    const auto x_fact_a = 2 * gga_fact_aa * dden_x_eval_a + gga_fact_ab * dden_x_eval_b;
    const auto y_fact_a = 2 * gga_fact_aa * dden_y_eval_a + gga_fact_ab * dden_y_eval_b;
    const auto z_fact_a = 2 * gga_fact_aa * dden_z_eval_a + gga_fact_ab * dden_z_eval_b;

    const auto x_fact_b = 2 * gga_fact_bb * dden_x_eval_b + gga_fact_ab * dden_x_eval_a;
    const auto y_fact_b = 2 * gga_fact_bb * dden_y_eval_b + gga_fact_ab * dden_y_eval_a;
    const auto z_fact_b = 2 * gga_fact_bb * dden_z_eval_b + gga_fact_ab * dden_z_eval_a;

    blas::axpy( inbf, x_fact_a, bf_x_col, 1, za_col, 1 );
    blas::axpy( inbf, y_fact_a, bf_y_col, 1, za_col, 1 );
    blas::axpy( inbf, z_fact_a, bf_z_col, 1, za_col, 1 );

    blas::axpy( inbf, x_fact_b, bf_x_col, 1, zb_col, 1 );
    blas::axpy( inbf, y_fact_b, bf_y_col, 1, zb_col, 1 );
    blas::axpy( inbf, z_fact_b, bf_z_col, 1, zb_col, 1 );

    if (vlapl != nullptr) {
      blas::axpy( inbf, vlapl[2*i],     lbf_col, 1, za_col, 1);
      blas::axpy( inbf, vlapl[2*i + 1], lbf_col, 1, zb_col, 1);
    }

  }
}
void ReferenceLocalHostWorkDriver::eval_mmat_mgga_vxc_uks_ts(size_t npts, size_t nbf,
        const double* vtau, const double* vlapl,
        const double* dbasis_x_eval, const double* dbasis_y_eval,
        const double* dbasis_z_eval,
        double* mmat_xa, double* mmat_ya, double* mmat_za, size_t ldma,
        double* mmat_xb, double* mmat_yb, double* mmat_zb, size_t ldmb) {

  const auto inbf  = static_cast<int32_t>(nbf);
  const auto inpts = static_cast<int32_t>(npts);
  const auto ildma = static_cast<int32_t>(ldma);
  const auto ildmb = static_cast<int32_t>(ldmb);

  if( ldma != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  if( ldmb != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));

  blas::lacpy( 'A', inbf, inpts, dbasis_x_eval, inbf, mmat_xa, ildma);
  blas::lacpy( 'A', inbf, inpts, dbasis_y_eval, inbf, mmat_ya, ildma);
  blas::lacpy( 'A', inbf, inpts, dbasis_z_eval, inbf, mmat_za, ildma);
  blas::lacpy( 'A', inbf, inpts, dbasis_x_eval, inbf, mmat_xb, ildmb);
  blas::lacpy( 'A', inbf, inpts, dbasis_y_eval, inbf, mmat_yb, ildmb);
  blas::lacpy( 'A', inbf, inpts, dbasis_z_eval, inbf, mmat_zb, ildmb);

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    const int32_t ioff = i * inbf;
    auto* xa_col = mmat_xa + ioff;
    auto* ya_col = mmat_ya + ioff;
    auto* za_col = mmat_za + ioff;
    auto* xb_col = mmat_xb + ioff;
    auto* yb_col = mmat_yb + ioff;
    auto* zb_col = mmat_zb + ioff;
    auto* bf_x_col = dbasis_x_eval + ioff;
    auto* bf_y_col = dbasis_y_eval + ioff;
    auto* bf_z_col = dbasis_z_eval + ioff;

    const auto tfacta = 0.25 * vtau[2*i];
    const auto tfactb = 0.25 * vtau[2*i+1];

    blas::scal( inbf, tfacta, xa_col, 1);
    blas::scal( inbf, tfacta, ya_col, 1);
    blas::scal( inbf, tfacta, za_col, 1);
    blas::scal( inbf, tfactb, xb_col, 1);
    blas::scal( inbf, tfactb, yb_col, 1);
    blas::scal( inbf, tfactb, zb_col, 1);

    if ( vlapl != nullptr ) {
      const auto lfacta = vlapl[2*i];
      const auto lfactb = vlapl[2*i+1];
      blas::axpy( inbf, lfacta, bf_x_col, 1, xa_col, 1);
      blas::axpy( inbf, lfacta, bf_y_col, 1, ya_col, 1);
      blas::axpy( inbf, lfacta, bf_z_col, 1, za_col, 1);
      blas::axpy( inbf, lfactb, bf_x_col, 1, xb_col, 1);
      blas::axpy( inbf, lfactb, bf_y_col, 1, yb_col, 1);
      blas::axpy( inbf, lfactb, bf_z_col, 1, zb_col, 1);
    }

  }
}






  // Increment VXC by Z
  void ReferenceLocalHostWorkDriver::inc_vxc( size_t npts, size_t nbf, size_t nbe, 
					      const double* basis_eval, const submat_map_t& submat_map, const double* Z,
					      size_t ldz, double* VXC, size_t ldvxc, double* scr, const double factor ) {
      
      const auto inbe  = static_cast<int32_t>(nbe);
      const auto inbf  = static_cast<int32_t>(nbf);
      const auto inpts = static_cast<int32_t>(npts);
      const auto ildz    = static_cast<int32_t>(ldz);
      const auto ildvxc  = static_cast<int32_t>(ldvxc);

      blas::syr2k('L', 'N', inbe, inpts, factor, basis_eval, inbe, Z, ildz, 0., scr, inbe );

      detail::inc_by_submat_atomic( inbf, inbf, inbe, inbe, VXC, ildvxc, scr, inbe, submat_map );

  }

  // Increment anti-symmetric part of VXC by Z
  void ReferenceLocalHostWorkDriver::inc_vxc_anti( size_t npts, size_t nbf, size_t nbe, 
					      const double* basis_eval, const submat_map_t& submat_map, const double* Z,
					      size_t ldz, double* VXC, size_t ldvxc, double* scr, const double factor ) {

      blas::gemm( 'N', 'T', nbe, nbe, npts, factor, basis_eval, nbe, Z, ldz, 0., scr, nbe );
      blas::gemm( 'N', 'T', nbe, nbe, npts, factor,  Z, ldz, basis_eval, nbe, -1., scr, nbe );

      detail::inc_by_submat_atomic( nbf, nbf, nbe, nbe, VXC, ldvxc, scr, nbe, submat_map );

  }

  // Increment K by G
  void ReferenceLocalHostWorkDriver::inc_exx_k( size_t npts, size_t nbf, 
						size_t nbe_bra, size_t nbe_ket, const double* basis_eval, 
						const submat_map_t& submat_map_bra, const submat_map_t& submat_map_ket, 
						const double* G, size_t ldg, double* K, size_t ldk, double* scr ) {

      const auto inbe_bra = static_cast<int32_t>(nbe_bra);
      const auto inbe_ket = static_cast<int32_t>(nbe_ket);
      const auto inbf     = static_cast<int32_t>(nbf);
      const auto inpts    = static_cast<int32_t>(npts);
      const auto ildg     = static_cast<int32_t>(ldg);
      const auto ildk     = static_cast<int32_t>(ldk);

      blas::gemm( 'N', 'T', inbe_bra, inbe_ket, inpts, 1., basis_eval, inbe_bra,
		  G, ildg, 0., scr, inbe_bra );

      detail::inc_by_submat_atomic( inbf, inbf, inbe_bra, inbe_ket, K, ildk, scr, inbe_bra,
			     submat_map_bra, submat_map_ket );

  }


  // Construct F = P * B (P non-square, TODO: should merge with XMAT)
  void ReferenceLocalHostWorkDriver::eval_exx_fmat( size_t npts, size_t nbf, 
						    size_t nbe_bra, size_t nbe_ket, const submat_map_t& submat_map_bra,
						    const submat_map_t& submat_map_ket, const double* P, size_t ldp,
						    const double* basis_eval, size_t ldb, double* F, size_t ldf,
						    double* scr ) {

    const auto inbe_bra = static_cast<int32_t>(nbe_bra);
    const auto inbe_ket = static_cast<int32_t>(nbe_ket);
    const auto inbf     = static_cast<int32_t>(nbf);
    const auto inpts    = static_cast<int32_t>(npts);
    const auto ildb     = static_cast<int32_t>(ldb);
    const auto ildf     = static_cast<int32_t>(ldf);
    const auto ildp     = static_cast<int32_t>(ldp);
    const auto* P_use = P;
    size_t ldp_use = ldp;

    if( submat_map_bra.size() > 1 or submat_map_ket.size() > 1 ) {
      detail::submat_set( inbf, inbf, inbe_bra, inbe_ket, P, ildp,
			  scr, inbe_bra, submat_map_bra, submat_map_ket );
      P_use = scr;
      ldp_use = nbe_bra;
    } else {
      P_use = P + submat_map_ket[0][0]*ldp + submat_map_bra[0][0];
    }

    blas::gemm( 'N', 'N', inbe_bra, inpts, inbe_ket, 1., P_use, static_cast<int32_t>(ldp_use), basis_eval,
		ildb, 0., F, ildf );

  }

  // Construct G(mu,i) = w(i) * A(mu,nu,i) * F(nu, i)
  void ReferenceLocalHostWorkDriver::eval_exx_gmat( size_t npts, size_t nshells, 
    size_t nshell_pairs, size_t nbe, const double* points, const double* weights, 
    const BasisSet<double>& basis, const ShellPairCollection<double>& shpairs, 
    const BasisSetMap& basis_map, const int32_t* shell_list, 
    const std::pair<int32_t,int32_t>* shell_pair_list, 
    const double* X, size_t ldx, double* G, size_t ldg ) {

    util::unused(basis_map);

    const auto inpts = static_cast<int32_t>(npts);
    const auto ildx  = static_cast<int32_t>(ldx);
    const auto ildg  = static_cast<int32_t>(ldg);
    XCPU::point* _points = 
      reinterpret_cast<XCPU::point*>(const_cast<double*>(points));
    std::vector<double> _points_transposed(3 * npts);

    for(size_t i = 0; i < npts; ++i) {
      _points_transposed[i + 0 * npts] = _points[i].x;
      _points_transposed[i + 1 * npts] = _points[i].y;
      _points_transposed[i + 2 * npts] = _points[i].z;
    }

  
    // Set G to zero
    for( size_t j = 0; j < npts; ++j )
    for( size_t i = 0; i < nbe;  ++i ) {
	    G[i + j*ldg] = 0.;
    }


    // Spherical Harmonic Transformer
    util::SphericalHarmonicTransform sph_trans(5);

    const bool any_pure = std::any_of( shell_list, shell_list + nshells,
				       [&](const auto& i){ return basis.at(i).pure(); } );
    
    const size_t nbe_cart =
      basis.nbf_cart_subset( shell_list, shell_list + nshells );
    const auto inbe_cart = static_cast<int32_t>(nbe_cart);

    std::vector<double> X_cart, G_cart;
    if( any_pure ){
      X_cart.resize( nbe_cart * npts );
      G_cart.resize( nbe_cart * npts, 0. );

      // Transform X into cartesian
      int ioff = 0;
      int ioff_cart = 0;
      for( auto i = 0ul; i < nshells; ++i ) {
        const auto ish = shell_list[i];
        const auto& shell      = basis.at(ish);
        const int shell_l       = shell.l();
        const int shell_sz      = shell.size();
        const int shell_cart_sz = shell.cart_size();
        
        if( shell.pure() and shell_l > 0 ) {
          sph_trans.itform_bra_cm( shell_l, inpts, X + ioff, ildx,
        			   X_cart.data() + ioff_cart, inbe_cart );
        } else {
          blas::lacpy( 'A', shell_sz, inpts, X + ioff, ildx,
        	       X_cart.data() + ioff_cart, inbe_cart );
        }
        ioff += shell_sz;
        ioff_cart += shell_cart_sz;
      }
    }

    const auto* X_use = any_pure ? X_cart.data() : X;
    auto*       G_use = any_pure ? G_cart.data() : G;
    const auto ldx_use = any_pure ? nbe_cart : ldx;
    const auto ldg_use = any_pure ? nbe_cart : ldg;

    std::vector<double> X_cart_rm( nbe_cart*npts,0. ), 
                        G_cart_rm( nbe_cart*npts,0. );
    for( auto i = 0ul; i < nbe_cart; ++i )
    for( auto j = 0ul; j < npts;     ++j ) {
      X_cart_rm[i*npts + j] = X_use[i + j*ldx_use];
    }


    std::map<size_t,size_t> cou_offsets_map;
    std::vector<size_t> cou_cart_sizes(nshells);
    cou_cart_sizes[0] = 0;
    cou_offsets_map[shell_list[0]] = 0;
    for(size_t i = 1; i < nshells; ++i) {
      cou_cart_sizes[i] = cou_cart_sizes[i-1] +
        basis.at(shell_list[i-1]).cart_size();
      cou_offsets_map[shell_list[i]] = cou_cart_sizes[i];
    }

    {
#if 0
    //size_t ioff_cart = 0;
    for( auto i = 0ul; i < nshells; ++i ) {
      const auto ish        = shell_list[i];
      const auto& bra       = basis[ish];
      const int bra_cart_sz = bra.cart_size();
      const size_t ioff_cart = cou_cart_sizes[i] * npts;
      XCPU::point bra_origin{bra.O()[0],bra.O()[1],bra.O()[2]};

      //size_t joff_cart = 0;
      for( auto j = 0ul; j <= i; ++j ) {
      //for( auto j = i; j < nshells; ++j ) {
        const auto jsh        = shell_list[j];
        const auto& ket       = basis[jsh];
        const int ket_cart_sz = ket.cart_size();
        const size_t joff_cart = cou_cart_sizes[j] * npts;
        XCPU::point ket_origin{ket.O()[0],ket.O()[1],ket.O()[2]};
        if(!need_sp(ish,jsh)) continue;

        auto sh_pair = shpairs.at(ish,jsh);
        auto prim_pair_data = sh_pair.prim_pairs();
        auto nprim_pair     = sh_pair.nprim_pairs();
        
        XCPU::compute_integral_shell_pair( ish == jsh,
        				   npts, _points_transposed.data(),
        				   bra.l(), ket.l(), bra_origin, ket_origin,
        				   nprim_pair, prim_pair_data,
        				   X_cart_rm.data()+ioff_cart, X_cart_rm.data()+joff_cart, npts,
        				   G_cart_rm.data()+ioff_cart, G_cart_rm.data()+joff_cart, npts,
        				   const_cast<double*>(weights), this->boys_table );
        
        //joff_cart += ket_cart_sz * npts;
      }
	
      //ioff_cart += bra_cart_sz * npts;
    }
#else
    for( auto ij = 0ul; ij < nshell_pairs; ++ij ) {
      auto [ish,jsh] = shell_pair_list[ij];
      //std::cout << "SHP " << ij << " " << i << " " << j << " " << nshells << std::endl;

     
      // Bra
      const auto& bra      = basis.at(ish);
      const auto ioff_cart = cou_offsets_map.at(ish) * npts;
      XCPU::point bra_origin{bra.O()[0],bra.O()[1],bra.O()[2]};

      // Ket
      const auto& ket      = basis.at(jsh);
      const auto joff_cart = cou_offsets_map.at(jsh) * npts;
      XCPU::point ket_origin{ket.O()[0],ket.O()[1],ket.O()[2]};

      auto sh_pair = shpairs.at(ish,jsh);
      auto prim_pair_data = sh_pair.prim_pairs();
      auto nprim_pair     = sh_pair.nprim_pairs();
      
      XCPU::compute_integral_shell_pair( ish == jsh,
      				   npts, _points_transposed.data(),
      				   bra.l(), ket.l(), bra_origin, ket_origin,
      				   static_cast<int>(nprim_pair), prim_pair_data,
      				   X_cart_rm.data()+ioff_cart, X_cart_rm.data()+joff_cart, inpts,
      				   G_cart_rm.data()+ioff_cart, G_cart_rm.data()+joff_cart, inpts,
      				   const_cast<double*>(weights), this->boys_table );
    }
#endif
    }
   
    for( auto i = 0ul; i < nbe_cart; ++i )
    for( auto j = 0ul; j < npts;     ++j ) {
	    G_use[i + j*ldg_use] = G_cart_rm[i*npts + j];
    }
  
    // Transform G back to spherical
    if( any_pure ) {
      size_t ioff = 0;
      size_t ioff_cart = 0;
      for( auto i = 0ul; i < nshells; ++i ) {
        const auto ish = shell_list[i];
        const auto& shell      = basis.at(ish);
        const int shell_l       = shell.l();
        const int shell_sz      = shell.size();
        const int shell_cart_sz = shell.cart_size();
        
        if( shell.pure() and shell_l > 0 ) {
          sph_trans.tform_bra_cm( shell_l, inpts, G_cart.data() + ioff_cart, inbe_cart,
        			  G + ioff, ildg );
        } else {
          blas::lacpy( 'A', shell_sz, inpts, G_cart.data() + ioff_cart, inbe_cart,
        	       G + ioff, ildg );
        }
        ioff += shell_sz;
        ioff_cart += shell_cart_sz;
      }
    }

  } // GMAT

}
