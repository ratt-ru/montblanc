#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "phase_op_gpu.cuh"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(
    Name("RimePhase")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    RimePhaseOp<tensorflow::GPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("RimePhase")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    RimePhaseOp<tensorflow::GPUDevice, double, tensorflow::complex128>);

} // namespace tensorflow

#endif // #if GOOGLE_CUDA