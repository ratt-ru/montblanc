#if GOOGLE_CUDA

#include "b_sqrt_op_gpu.cuh"

namespace montblanc {
namespace bsqrt {

REGISTER_KERNEL_BUILDER(
    Name("RimeBSqrt")
    .Device(tensorflow::DEVICE_GPU)
    .HostMemory("ref_freq")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    RimeBSqrt<GPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("RimeBSqrt")
    .Device(tensorflow::DEVICE_GPU)
    .HostMemory("ref_freq")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    RimeBSqrt<GPUDevice, double, tensorflow::complex128>);

} // namespace bsqrt {
} // namespace montblanc {

#endif