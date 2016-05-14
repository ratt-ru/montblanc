#if GOOGLE_CUDA

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

#define EIGEN_USE_GPU

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

}

#endif // #if GOOGLE_CUDA