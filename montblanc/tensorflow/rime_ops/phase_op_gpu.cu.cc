#if GOOGLE_CUDA

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/device_base.h"

#define EIGEN_USE_GPU

namespace gpu = perftools::gputools;
//namespace gcudacc = platforms::gpus::gcudacc;


namespace tensorflow {

//gpu::Platform * platform = gpu::MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();

typedef Eigen::GpuDevice GPUDevice;

template <typename FT, typename CT>
__global__ void rime_phase(FT * data, int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    FT value = data[i];
    value *= 10.0;
    data[i] = value;
}


}

#endif // #if GOOGLE_CUDA