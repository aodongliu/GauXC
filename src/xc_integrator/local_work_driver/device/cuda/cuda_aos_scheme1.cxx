#include "cuda_aos_scheme1.hpp"
#include "device/cuda/cuda_backend.hpp"
#include "cuda_aos_scheme1_weights.hpp"
//#include "kernels/cublas_extensions.hpp"
//#include "kernels/uvvars.hpp"
//#include "kernels/pack_submat.hpp"
//#include "kernels/cuda_inc_potential.hpp"
//#include "kernels/symmetrize_mat.hpp"

//#include "device/common/zmat_vxc.hpp"
//#include "device/common/device_blas.hpp"
//#include "device/common/xc_functional_eval_wrapper.hpp"
//#include "device/common/uvvars.hpp"
//#include "device/common/pack_submat.hpp"
//#include "device/common/inc_potential.hpp"
//#include "device/common/symmetrize_mat.hpp"

namespace GauXC {

std::unique_ptr<XCDeviceData> CudaAoSScheme1::create_device_data() {
  return std::make_unique<Data>();
}

 
void CudaAoSScheme1::partition_weights( XCDeviceData* _data ) {
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) throw std::runtime_error("BAD DATA CAST");

  auto device_backend = dynamic_cast<CUDABackend*>(data->device_backend_.get());
  if( !device_backend ) throw std::runtime_error("BAD BACKEND CAST");


  // Compute distances from grid to atomic centers
  const auto ldatoms = data->get_ldatoms();
  auto base_stack    = data->base_stack;
  auto static_stack  = data->static_stack;
  auto scheme1_stack = data->scheme1_stack;
  cuda_aos_scheme1_weights_wrapper( data->total_npts_task_batch, data->global_dims.natoms,
    base_stack.points_device, static_stack.rab_device, ldatoms, 
    static_stack.coords_device, scheme1_stack.dist_scratch_device, ldatoms, 
    scheme1_stack.iparent_device, scheme1_stack.dist_nearest_device, 
    base_stack.weights_device, *device_backend->master_stream );
}





}
