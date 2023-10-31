/**
 * GauXC Copyright (c) 2020-2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once
#include <gauxc/xc_integrator/replicated/replicated_xc_device_integrator.hpp>
#include "device/xc_device_data.hpp"

namespace GauXC {
namespace detail {

template <typename ValueType>
class IncoreReplicatedXCDeviceIntegrator : 
  public ReplicatedXCDeviceIntegrator<ValueType> {

  using base_type  = ReplicatedXCDeviceIntegrator<ValueType>;

public:

  using value_type = typename base_type::value_type;
  using basis_type = typename base_type::basis_type;

  using host_task_container = std::vector<XCTask>;
  using host_task_iterator  = typename host_task_container::iterator;

protected:

  void integrate_den_( int64_t m, int64_t n, const value_type* P,
                      int64_t ldp, value_type* N_EL ) override;

  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* P,
                      int64_t ldp, value_type* VXC, int64_t ldvxc,
                      value_type* EXC ) override;

  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Pscalar,
                      int64_t ldpscalar,
                      const value_type* Pz,
                      int64_t ldpz,
                      value_type* VXCscalar, int64_t ldvxcscalar,
                      value_type* VXCz, int64_t ldvxcz,
                      value_type* EXC ) override;
  
  void eval_neo_exc_vxc_( int64_t m1, int64_t n1, int64_t m2, int64_t n2, 
                         const value_type* P1s, int64_t ldp1s,
                         const value_type* P2s, int64_t ldp2s,
                         const value_type* P2z, int64_t ldp2z,
                         value_type* VXC1s, int64_t ldvxc1s,
                         value_type* VXC2s, int64_t ldvxc2s,
                         value_type* VXC2z, int64_t ldvxc2z,
                         value_type* EXC1,  value_type* EXC2 ) override;

  void eval_neo_exc_vxc_( int64_t m1, int64_t n1, int64_t m2, int64_t n2, 
                         const value_type* P1s, int64_t ldp1s,
                         const value_type* P1z, int64_t ldp1z,
                         const value_type* P2s, int64_t ldp2s,
                         const value_type* P2z, int64_t ldp2z,
                         value_type* VXC1s, int64_t ldvxc1s,
                         value_type* VXC1z, int64_t ldvxc1z,
                         value_type* VXC2s, int64_t ldvxc2s,
                         value_type* VXC2z, int64_t ldvxc2z,
                         value_type* EXC1,  value_type* EXC2 ) override;


  void eval_exc_grad_( int64_t m, int64_t n, const value_type* P,
                       int64_t ldp, value_type* EXC_GRAD ) override;

  void eval_exx_( int64_t m, int64_t n, const value_type* P,
                  int64_t ldp, value_type* K, int64_t ldk,
                  const IntegratorSettingsEXX& settings ) override;


  void integrate_den_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                            value_type *N_EL,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );

  void exc_vxc_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );

  void exc_vxc_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                            value_type* VXC, int64_t ldvxc, value_type* EXC, value_type *N_EL,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );

  void exc_vxc_local_work_( const basis_type& basis, const value_type* Pscalar, int64_t ldpscalar,
                                const value_type* Pz, int64_t ldpz,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );

  void exc_vxc_local_work_( const basis_type& basis, const value_type* Pscalar, int64_t ldpscalar,
                            const value_type* Pz, int64_t ldpz,
                            value_type* VXC, int64_t ldvxc,
                            value_type* VXCz, int64_t ldvxcz, value_type* EXC, value_type *N_EL,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );

  void eval_exc_grad_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                                  host_task_iterator task_begin, host_task_iterator task_end,
                                  XCDeviceData& device_data );

  void eval_exc_grad_local_work_( const basis_type& basis, const value_type* P,
                                  int64_t ldp, value_type* EXC_GRAD, 
                                  host_task_iterator task_begin, host_task_iterator task_end,
                                  XCDeviceData& device_data );



  void exx_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                        host_task_iterator task_begin, host_task_iterator task_end,
                        XCDeviceData& device_data, 
                        const IntegratorSettingsEXX& settings);

  void exx_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                        value_type* K, int64_t ldk,
                        host_task_iterator task_begin, host_task_iterator task_end,
                        XCDeviceData& device_data, 
                        const IntegratorSettingsEXX& settings);

  void exx_ek_screening_local_work_( const basis_type& basis, 
                        const value_type* P, int64_t ldp, 
                        XCDeviceData& device_data, 
                        const IntegratorSettingsEXX& settings);

public:

  template <typename... Args>
  IncoreReplicatedXCDeviceIntegrator( Args&&... args ) :
    base_type( std::forward<Args>(args)... ) { }

  virtual ~IncoreReplicatedXCDeviceIntegrator() noexcept;


  template <typename... Args>
  void exc_vxc_local_work(Args&&... args) {
    exc_vxc_local_work_( std::forward<Args>(args)... );
  }

};

extern template class IncoreReplicatedXCDeviceIntegrator<double>;

}
}
