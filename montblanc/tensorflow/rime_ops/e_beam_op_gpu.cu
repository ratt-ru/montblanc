#include "e_beam_op_gpu.cuh"

namespace montblanc {
namespace ebeam {

REGISTER_KERNEL_BUILDER(
    Name("RimeEBeam")
    .Device(tensorflow::DEVICE_GPU)
    .HostMemory("parallactic_angle")
    .HostMemory("beam_ll")
    .HostMemory("beam_ul")
    .HostMemory("beam_lm")
    .HostMemory("beam_um")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    RimeEBeam<GPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("RimeEBeam")
    .Device(tensorflow::DEVICE_GPU)
    .HostMemory("parallactic_angle")
    .HostMemory("beam_ll")
    .HostMemory("beam_ul")
    .HostMemory("beam_lm")
    .HostMemory("beam_um")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    RimeEBeam<GPUDevice, double, tensorflow::complex128>);

} // namespace ebeam {
} // namespace montblanc {
