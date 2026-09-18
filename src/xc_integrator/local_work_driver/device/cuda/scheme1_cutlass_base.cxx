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
#include <stdexcept>

#include "scheme1_cutlass_base.hpp"
#include "device/common/pack_submat.hpp"
#include "device/common/inc_potential.hpp"
#include "device/common/device_blas.hpp"

#include "device/cuda/kernels/cutlass_wrapper.hpp"
#include "device/cuda/kernels/syr2k_tiny.hpp"
#include "device/common/syr2k_size_classes.hpp"

namespace GauXC {

// Common implementation for eval_xmat and eval_xmat_trial
template<bool is_trial>
void AoSScheme1CUTLASSBase::eval_xmat_impl(double fac, XCDeviceData* _data, bool do_grad, density_id den_id) {
  auto* data = dynamic_cast<AoSScheme1CUTLASSBase::Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  // Pack density matrix 
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  
  double* dmat_ptr;
  if constexpr (is_trial) {
    dmat_ptr = static_stack.tden_selector(den_id);
    // now screened trial density matrix is stored in aos_stack.device_tasks[itask].nbe_scr
  } else {
    dmat_ptr = static_stack.den_selector(den_id);
  }
  
  sym_pack_submat( ntasks, aos_stack.device_tasks, dmat_ptr, 
    nbf, submat_block_size, data->device_backend_->queue() );

  auto cutlass_stack = data->cutlass_stack;
  double** dmat_array;
  if constexpr (is_trial) {
    dmat_array = cutlass_stack.tdmat_array(den_id);
  } else {
    dmat_array = cutlass_stack.dmat_array(den_id);
  }
  cutlass_gemm(
    cutlass_stack.problem_sizes_device,
    data->problem_sizes_host.data(),
    ntasks,
    cutlass_stack.bf_array_device, dmat_array,
    cutlass_stack.zmat_array_device, cutlass_stack.zmat_array_device,
    cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
    cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
    fac, 0.0,
    data->device_backend_->queue()
  );

  if(do_grad) {
    cutlass_gemm(
      cutlass_stack.problem_sizes_device,
      data->problem_sizes_host.data(),
      ntasks,
      cutlass_stack.bfx_array_device, dmat_array,
      cutlass_stack.xmat_x_array_device, cutlass_stack.xmat_x_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
      cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
      fac, 0.0,
      data->device_backend_->queue()
    );
    cutlass_gemm(
      cutlass_stack.problem_sizes_device,
      data->problem_sizes_host.data(),
      ntasks,
      cutlass_stack.bfy_array_device, dmat_array,
      cutlass_stack.xmat_y_array_device, cutlass_stack.xmat_y_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
      cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
      fac, 0.0,
      data->device_backend_->queue()
    );
    cutlass_gemm(
      cutlass_stack.problem_sizes_device,
      data->problem_sizes_host.data(),
      ntasks,
      cutlass_stack.bfz_array_device, dmat_array,
      cutlass_stack.xmat_z_array_device, cutlass_stack.xmat_z_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
      cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
      fac, 0.0,
      data->device_backend_->queue()
    );
  }
}

void AoSScheme1CUTLASSBase::eval_xmat(double fac, XCDeviceData* _data, bool do_grad, density_id den_id ) {
  eval_xmat_impl<false>(fac, _data, do_grad, den_id);
}

void AoSScheme1CUTLASSBase::eval_xmat_trial(double fac, XCDeviceData* _data, bool do_grad, density_id den_id ) {
  eval_xmat_impl<true>(fac, _data, do_grad, den_id);
}


// Common implementation for inc_vxc and inc_fxc
template<bool is_fxc>
void AoSScheme1CUTLASSBase::inc_potential_impl(XCDeviceData* _data, density_id den_id, bool do_m) {
  auto* data = dynamic_cast<AoSScheme1CUTLASSBase::Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  auto cutlass_stack = data->cutlass_stack;
  const auto& work_ranges = data->task_work_ranges_;

  // V += A Z^T + Z A^T over every task, in one grouped call, or one call per
  // work range with the executor of the range's syr2k size class
  auto inc_potential_syr2k = [&]( double** ptr_A, double** ptr_B,
    int64_t* ld_A, int64_t* ld_B, double beta ) {

    if( work_ranges.empty() ) {
      cutlass_syr2k(
        cutlass_stack.syr2k_sizes_device,
        data->syr2k_sizes_host.data(),
        ntasks,
        ptr_A, ptr_B,
        cutlass_stack.vmat_array_device, cutlass_stack.vmat_array_device,
        ld_A, ld_B,
        cutlass_stack.ld64_vmat_array_device, cutlass_stack.ld64_vmat_array_device,
        1.0, beta,
        data->device_backend_->queue()
      );
      return;
    }

    for( const auto& range : work_ranges ) {
      const size_t first = range.first;
      const int    count = range.count ? range.count : ntasks - first;
      const auto   size_class = syr2k_size_class_of( range.nbe_max );
      if( size_class == syr2k_size_class::TINY ) {
        syr2k_tiny(
          count,
          cutlass_stack.syr2k_sizes_device + first,
          ptr_A + first, ptr_B + first,
          cutlass_stack.vmat_array_device + first,
          cutlass_stack.vmat_array_device + first,
          ld_A + first, ld_B + first,
          cutlass_stack.ld64_vmat_array_device + first,
          cutlass_stack.ld64_vmat_array_device + first,
          1.0, beta,
          data->device_backend_->queue()
        );
        continue;
      }
      auto cutlass_syr2k_tile = size_class == syr2k_size_class::SMALL
        ? &cutlass_syr2k_32x32 : &cutlass_syr2k;
      cutlass_syr2k_tile(
        cutlass_stack.syr2k_sizes_device + first,
        data->syr2k_sizes_host.data() + first,
        count,
        ptr_A + first, ptr_B + first,
        cutlass_stack.vmat_array_device + first,
        cutlass_stack.vmat_array_device + first,
        ld_A + first, ld_B + first,
        cutlass_stack.ld64_vmat_array_device + first,
        cutlass_stack.ld64_vmat_array_device + first,
        1.0, beta,
        data->device_backend_->queue()
      );
    }
  };

  inc_potential_syr2k( cutlass_stack.bf_array_device,
    cutlass_stack.zmat_array_device, cutlass_stack.ld64_bf_array_device,
    cutlass_stack.ld64_zmat_array_device, 0.0 );
  if(do_m) {
    inc_potential_syr2k( cutlass_stack.bfx_array_device,
      cutlass_stack.xmat_x_array_device, cutlass_stack.ld64_bf_array_device,
      cutlass_stack.ld64_zmat_array_device, 1.0 );
    inc_potential_syr2k( cutlass_stack.bfy_array_device,
      cutlass_stack.xmat_y_array_device, cutlass_stack.ld64_bf_array_device,
      cutlass_stack.ld64_zmat_array_device, 1.0 );
    inc_potential_syr2k( cutlass_stack.bfz_array_device,
      cutlass_stack.xmat_z_array_device, cutlass_stack.ld64_bf_array_device,
      cutlass_stack.ld64_zmat_array_device, 1.0 );
  }

  // Increment global VXC/FXC
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  
  double* potential_ptr;
  if constexpr (is_fxc) {
    potential_ptr = static_stack.fxc_selector(den_id);
    // cutlass_stack.vmat_array_device points to aos_stack.device_tasks[itask].nbe_scr
  } else {
    potential_ptr = static_stack.vxc_selector(den_id);
  }
  
  sym_task_inc_potential( ntasks, aos_stack.device_tasks, potential_ptr, nbf, 
    submat_block_size, data->device_backend_->queue() );
}

void AoSScheme1CUTLASSBase::inc_vxc( XCDeviceData* _data, density_id den_id, bool do_m ) {
  inc_potential_impl<false>(_data, den_id, do_m);
}

void AoSScheme1CUTLASSBase::inc_fxc( XCDeviceData* _data, density_id den_id, bool do_m ) {
  inc_potential_impl<true>(_data, den_id, do_m);
}

}
