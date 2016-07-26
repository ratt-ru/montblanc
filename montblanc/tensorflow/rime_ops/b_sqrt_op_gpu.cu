#if GOOGLE_CUDA

#include "b_sqrt_op_gpu.cuh"

namespace montblanc {
namespace bsqrt {

REGISTER_KERNEL_BUILDER(
    Name("BSqrt")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    BSqrt<GPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("BSqrt")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    BSqrt<GPUDevice, double, tensorflow::complex128>);

} // namespace bsqrt {
} // namespace montblanc {

#endif