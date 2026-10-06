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
#include "local_device_work_driver_pimpl.hpp"
#include <stdexcept>

namespace GauXC {

LocalDeviceWorkDriver::LocalDeviceWorkDriver() : 
  pimpl_(nullptr) { }
LocalDeviceWorkDriver::LocalDeviceWorkDriver(pimpl_type&& ptr) :
  pimpl_( std::move(ptr) ){ }

LocalDeviceWorkDriver::~LocalDeviceWorkDriver() noexcept = default;

LocalDeviceWorkDriver::LocalDeviceWorkDriver( LocalDeviceWorkDriver&& other ) noexcept :
  pimpl_(std::move(other.pimpl_)) { }

#define throw_if_invalid_pimpl(ptr) \
  if(not ptr) GAUXC_PIMPL_NOT_INITIALIZED()



#define FWD_TO_PIMPL(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data);                                    \
}
#define FWD_TO_PIMPL_BOOL(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, bool b ) { \
  throw_if_invalid_pimpl(pimpl_);                                       \
  pimpl_->NAME(device_data, b);                                         \
}

#define FWD_TO_PIMPL_DEN_ID(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, density_id den ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, den);                               \
}

#define FWD_TO_PIMPL_DEN_ID_BOOL(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, density_id den, bool b ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, den, b);                               \
}

#define FWD_TO_PIMPL_BOOL_DEN_ID(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, bool b, density_id den ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, b, den);                               \
}

#define FWD_TO_PIMPL_KS_SCHEME(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, integrator_ks_scheme track ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, track);                               \
}
#define FWD_TO_PIMPL_KS_SCHEME_BOOL(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, integrator_ks_scheme track, bool b ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, track, b);                               \
}
#define FWD_TO_PIMPL_KS_SCHEME_BOOL_BOOL(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, integrator_ks_scheme track, bool b1, bool b2 ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, track, b1, b2);                               \
}
#define FWD_TO_PIMPL_KS_SCHEME_DEN_ID(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, integrator_ks_scheme track, density_id den ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, track, den);                               \
}
#define FWD_TO_PIMPL_KS_SCHEME_BOOL_DEN_ID(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, integrator_ks_scheme track, bool b, density_id den ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, track, b, den);                               \
}
#define FWD_TO_PIMPL_RANGE(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, point_work_range range ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, range);                               \
}
#define FWD_TO_PIMPL_DEN_ID_RANGE(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, density_id den, task_work_range range ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, den, range);                               \
}
#define FWD_TO_PIMPL_KS_SCHEME_DEN_ID_RANGE(NAME) \
void LocalDeviceWorkDriver::NAME( XCDeviceData* device_data, integrator_ks_scheme track, density_id den, task_work_range range ) { \
  throw_if_invalid_pimpl(pimpl_);                               \
  pimpl_->NAME(device_data, track, den, range);                               \
}

FWD_TO_PIMPL(partition_weights)         // Partition weights

FWD_TO_PIMPL(eval_collocation)          // Collocation
FWD_TO_PIMPL(eval_collocation_gradient) // Collocation Gradient
FWD_TO_PIMPL(eval_collocation_hessian)  // Collocation Hessian
FWD_TO_PIMPL(eval_collocation_laplacian)  // Collocation Laplacian
FWD_TO_PIMPL(eval_collocation_lapgrad)  // Collocation Laplacian gradient


FWD_TO_PIMPL_KS_SCHEME(eval_uvars_lda)            // U variables LDA (rho)
FWD_TO_PIMPL_KS_SCHEME(eval_uvars_gga)            // U variables GGA (gamma)
FWD_TO_PIMPL_KS_SCHEME_BOOL(eval_uvars_mgga)      // U variables MGGA (tau, lapl)
FWD_TO_PIMPL_DEN_ID_RANGE(eval_vvars_lda)          // V variables LDA (density), per work range
FWD_TO_PIMPL_DEN_ID_RANGE(eval_vvars_gga)          // V variables GGA (density + grad), per work range
FWD_TO_PIMPL_DEN_ID_BOOL(eval_vvars_mgga)         // V variables MGGA (density + grad + tau + lapl)

FWD_TO_PIMPL_KS_SCHEME(eval_tmat_lda)            // T variables LDA (trho)
FWD_TO_PIMPL_KS_SCHEME(eval_tmat_gga)            // T variables GGA (tgamma)
FWD_TO_PIMPL_KS_SCHEME_BOOL(eval_tmat_mgga)      // T variables MGGA (ttau, tlapl)
FWD_TO_PIMPL_DEN_ID(eval_vvars_lda_trial)               // V variables LDA (trial density)
FWD_TO_PIMPL_DEN_ID(eval_vvars_gga_trial)               // V variables GGA (trial density + grad)
FWD_TO_PIMPL_DEN_ID_BOOL(eval_vvars_mgga_trial)         // V variables MGGA (trial density + grad + tau + lapl)

FWD_TO_PIMPL_KS_SCHEME_DEN_ID_RANGE(eval_zmat_lda_vxc)    // Eval Z Matrix LDA VXC, per work range
FWD_TO_PIMPL_KS_SCHEME_DEN_ID_RANGE(eval_zmat_gga_vxc)    // Eval Z Matrix GGA VXC, per work range
FWD_TO_PIMPL_KS_SCHEME_BOOL_DEN_ID(eval_zmat_mgga_vxc)   // Eval Z Matrix mGGA VXC
FWD_TO_PIMPL_KS_SCHEME_BOOL_DEN_ID(eval_mmat_mgga_vxc)   // Eval M Matrix mGGA VXC

FWD_TO_PIMPL_DEN_ID(eval_zmat_lda_fxc)         // Eval Z Matrix LDA FXC
FWD_TO_PIMPL_DEN_ID(eval_zmat_gga_fxc)         // Eval Z Matrix GGA FXC
FWD_TO_PIMPL_BOOL_DEN_ID(eval_zmat_mgga_fxc)   // Eval Z Matrix mGGA FXC
FWD_TO_PIMPL_BOOL_DEN_ID(eval_mmat_mgga_fxc)   // Eval M Matrix mGGA FXC


FWD_TO_PIMPL(eval_exx_fmat)             // Eval EXX F Matrix
//FWD_TO_PIMPL(eval_exx_gmat)           // Eval EXX G Matrix


void LocalDeviceWorkDriver::inc_exc( XCDeviceData* data, point_work_range range,
  double* accumulator ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->inc_exc(data, range, accumulator);
}

FWD_TO_PIMPL_RANGE(inc_nel)                   // Increment N_EL over a point work range

void LocalDeviceWorkDriver::inc_vxc( XCDeviceData* data, density_id den,
  bool do_m, const std::vector<task_work_range>& work_ranges ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->inc_vxc(data, den, do_m, work_ranges);
}

FWD_TO_PIMPL_DEN_ID_BOOL(inc_fxc)            // Increment FXC_I by Z

FWD_TO_PIMPL(inc_exx_k)     
FWD_TO_PIMPL_KS_SCHEME_BOOL(inc_exc_grad_lda)
FWD_TO_PIMPL_KS_SCHEME_BOOL(inc_exc_grad_gga)
FWD_TO_PIMPL_KS_SCHEME_BOOL_BOOL(inc_exc_grad_mgga)

FWD_TO_PIMPL_DEN_ID(symmetrize_vxc)
FWD_TO_PIMPL_DEN_ID(symmetrize_fxc) // Added FXC function
FWD_TO_PIMPL(symmetrize_exx_k)
FWD_TO_PIMPL(eval_exx_ek_screening_bfn_stats)

// X     = fac * P * B
// dX/dx = fac * P * dB/dx (do_grad)

void LocalDeviceWorkDriver::eval_xmat( double fac, XCDeviceData* device_data, bool do_grad, density_id den ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_xmat(fac, device_data, do_grad, den);
}
void LocalDeviceWorkDriver::save_xmat( XCDeviceData* device_data, bool do_grad, density_id den ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->save_xmat(device_data, do_grad, den);
}
void LocalDeviceWorkDriver::eval_xmat_trial( double fac, XCDeviceData* device_data, bool do_grad, density_id den ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_xmat_trial(fac, device_data, do_grad, den);
}

void LocalDeviceWorkDriver::eval_exx_gmat( XCDeviceData* device_data, 
  const BasisSetMap& basis_map) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_exx_gmat(device_data, basis_map);
}

void LocalDeviceWorkDriver::eval_kern_exc_vxc_lda( const functional_type& func,
  XCDeviceData* data, point_work_range range ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_kern_exc_vxc_lda(func,data,range);
}

void LocalDeviceWorkDriver::eval_kern_exc_vxc_gga( const functional_type& func,
  XCDeviceData* data, point_work_range range ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_kern_exc_vxc_gga(func,data,range);
}

void LocalDeviceWorkDriver::eval_kern_exc_vxc_mgga( const functional_type& func,
  XCDeviceData* data) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_kern_exc_vxc_mgga(func,data);
}

void LocalDeviceWorkDriver::eval_kern_vxc_fxc_lda( const functional_type& func,
  XCDeviceData* data) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_kern_vxc_fxc_lda(func,data);
}

void LocalDeviceWorkDriver::eval_kern_vxc_fxc_gga( const functional_type& func,
  XCDeviceData* data) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_kern_vxc_fxc_gga(func,data);
}

void LocalDeviceWorkDriver::eval_kern_vxc_fxc_mgga( const functional_type& func,
  XCDeviceData* data) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_kern_vxc_fxc_mgga(func,data);
}

std::unique_ptr<XCDeviceData> LocalDeviceWorkDriver::create_device_data(const DeviceRuntimeEnvironment& rt) {
  throw_if_invalid_pimpl(pimpl_);
  return pimpl_->create_device_data(rt);
}

bool LocalDeviceWorkDriver::supports_multiparticle() const {
  throw_if_invalid_pimpl(pimpl_);
  return pimpl_->supports_multiparticle();
}

void LocalDeviceWorkDriver::gather_pair_densities( XCDeviceData* device_data ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->gather_pair_densities(device_data);
}

void LocalDeviceWorkDriver::eval_pair_functional( const functional_type& func,
  XCDeviceData* device_data, point_work_range range ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_pair_functional(func, device_data, range);
}

void LocalDeviceWorkDriver::scatter_pair_potential( XCDeviceData* device_data ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->scatter_pair_potential(device_data);
}

void LocalDeviceWorkDriver::sum_pair_energy( XCDeviceData* device_data ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->sum_pair_energy(device_data);
}

void LocalDeviceWorkDriver::accumulate_w_times_f( XCDeviceData* device_data ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->accumulate_w_times_f(device_data);
}

void LocalDeviceWorkDriver::eval_weight_1st_deriv_contracted(
  XCDeviceData* device_data, XCWeightAlg alg, const double* w_times_f, size_t npts ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_weight_1st_deriv_contracted(device_data, alg, w_times_f, npts);
}

void LocalDeviceWorkDriver::exx_ek_shellpair_collision( double eps_E, double eps_K, 
  XCDeviceData* device_data, host_task_iterator tb, host_task_iterator te,
  const ShellPairCollection<double>& shpairs ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->exx_ek_shellpair_collision( eps_E, eps_K, device_data, tb, te, shpairs );
}

void LocalDeviceWorkDriver::eval_weight_1st_deriv_contracted( XCDeviceData* device_data, XCWeightAlg alg ) {
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_weight_1st_deriv_contracted(device_data, alg);
}

}
