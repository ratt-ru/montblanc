#ifndef RIME_B_SQRT_OP_GPU_H_
#define RIME_B_SQRT_OP_GPU_H_

#if GOOGLE_CUDA

#include "b_sqrt_op.h"

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;    

template <typename FT, typename CT>
class RimeBSqrt<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit RimeBSqrt(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;
    }
};

} // namespace tensorflow {

#endif

#endif // #ifndef RIME_B_SQRT_OP_GPU_H_