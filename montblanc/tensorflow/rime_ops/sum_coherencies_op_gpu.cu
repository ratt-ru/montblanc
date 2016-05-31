#if GOOGLE_CUDA

#include "sum_coherencies_op_gpu.cuh"

namespace montblanc {
namespace sumcoherencies {

REGISTER_KERNEL_BUILDER(
    Name("RimeSumCoherencies")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    RimeSumCoherencies<GPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("RimeSumCoherencies")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    RimeSumCoherencies<GPUDevice, double, tensorflow::complex128>);

} // namespace sumcoherencies {
} // namespace montblanc {

#endif