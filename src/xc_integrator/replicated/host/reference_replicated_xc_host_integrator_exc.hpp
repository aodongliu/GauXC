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

namespace GauXC  {
namespace detail {
// dks
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
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
             value_type* EXC, const IntegratorSettingsXC& ks_settings ) {

  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P Must Have Same Dimension as Basis");
  if( ldps and ldps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDP");
  if( ldpz and ldpz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPZ");
  if( ldpy and ldpy < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPY");
  if( ldpx and ldpx < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPX");


  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Temporary electron count to judge integrator accuracy
  value_type N_EL;
  value_type spin_N_EL;


  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    //exc_vxc_local_work_( P, ldp, VXC, ldvxc, EXC, &N_EL );
    exc_vxc_local_work_( basis, Ps, ldps, Pz, ldpz, Py, ldpy, Px, ldpx,
                         Ps_SS, ldps_ss, Pz_SS, ldpz_ss, Py_SS, ldpy_ss, Px_SS, ldpx_ss,
                         Ps_SS_imag, Pz_SS_imag, Py_SS_imag, Px_SS_imag, 
                         nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
                         nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
                         nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
                         EXC, &N_EL, &spin_N_EL, ks_settings, tasks.begin(), tasks.end());
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( EXC,   1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &spin_N_EL, 1    , ReductionOp::Sum );

  });

}
// RKS
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
             value_type* EXC, const IntegratorSettingsXC& ks_settings ) {

  eval_exc_(m, n, P, ldp, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, nullptr, nullptr, nullptr, EXC, ks_settings);

}
// UKS
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
             const value_type* Pz, int64_t ldpz,
             value_type* EXC, const IntegratorSettingsXC& ks_settings ) {

  eval_exc_(m, n, Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0, 
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, nullptr, nullptr, nullptr, EXC, ks_settings);

}
// GKS
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
             const value_type* Pz, int64_t ldpz,
             const value_type* Py, int64_t ldpy,
             const value_type* Px, int64_t ldpx,
             value_type* EXC, const IntegratorSettingsXC& ks_settings ) {

  eval_exc_(m, n, Ps, ldps, Pz, ldpz, Py, ldpy, Px, ldpx, 
    nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
    nullptr, nullptr, nullptr, nullptr, EXC, ks_settings);

}
}

}
