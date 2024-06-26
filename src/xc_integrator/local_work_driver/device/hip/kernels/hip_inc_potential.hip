/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include "device_specific/hip_device_constants.hpp"
#include "device/common/inc_potential.hpp"
#include <gauxc/util/div_ceil.hpp>
#include "device_specific/hip_util.hpp"


namespace GauXC {

template <typename T>
__global__ void inc_by_submat_combined_kernel( size_t           ntasks,
                                               XCDeviceTask* device_tasks,
                                               T*               A,
                                               size_t           LDA ) {

  const int batch_id = blockIdx.z;

  if( batch_id < ntasks ) {

  auto& task = device_tasks[ batch_id ];

  const auto  ncut              = task.bfn_screening.ncut;
  const auto* submat_cut_device = task.bfn_screening.submat_cut;
  const auto  LDAS              = task.bfn_screening.nbe;
        auto* ASmall_device     = task.nbe_scr;

  //if( LDAS == LDAB ) return;


  const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

  int64_t i(0);
  for( size_t i_cut = 0; i_cut < ncut; ++i_cut ) {
    const int64_t i_cut_first  = submat_cut_device[ 3*i_cut ];
    const int64_t delta_i      = submat_cut_device[ 3*i_cut + 1 ];

    int64_t j(0);
  for( size_t j_cut = 0; j_cut < ncut; ++j_cut ) {
    const int64_t j_cut_first  = submat_cut_device[ 3*j_cut ];
    const int64_t delta_j      = submat_cut_device[ 3*j_cut + 1 ];

    auto* ASmall_begin = ASmall_device + i           + j          *LDAS;
    auto* ABig_begin   = A             + i_cut_first + j_cut_first*LDA ;

    for( size_t J = tid_y; J < delta_j; J += blockDim.y )      
    for( size_t I = tid_x; I < delta_i; I += blockDim.x )
      //ABig_begin[I + J*LDA] += ASmall_begin[I + J*LDAS];
      atomicAdd( ABig_begin + I + J*LDA, ASmall_begin[I+J*LDAS] );

    j += delta_j;
  }
    i += delta_i;
  }

  } // batch_id check
}

void sym_task_inc_potential( size_t        ntasks,
                         XCDeviceTask* device_tasks,
                         double*       V_device,
                         size_t        LDV,
                         size_t        submat_block_size,
                         device_queue queue ) {


  hipStream_t stream = queue.queue_as<util::hip_stream>();
  dim3 threads(hip::warp_size,hip::max_warps_per_thread_block,1), 
       blocks(1,1,ntasks);
  hipLaunchKernelGGL(inc_by_submat_combined_kernel, blocks, threads, 0, stream , 
    ntasks, device_tasks, V_device, LDV
  );

}


}

