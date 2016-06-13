#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "phase_op_gpu.cuh"

namespace montblanc {
namespace phase {

REGISTER_KERNEL_BUILDER(
    Name("Phase")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    Phase<GPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("Phase")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    Phase<GPUDevice, double, tensorflow::complex128>);

} // namespace phase {
} // namespace montblanc {

#endif // #if GOOGLE_CUDA