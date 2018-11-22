#if GOOGLE_CUDA

#include "jones_multiply_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN


// Register a GPU kernel for JonesMultiply
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("JonesMultiply")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    JonesMultiply<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for JonesMultiply
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("JonesMultiply")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    JonesMultiply<GPUDevice, double, tensorflow::complex128>);



MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
