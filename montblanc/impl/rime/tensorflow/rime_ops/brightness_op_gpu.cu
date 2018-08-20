#if GOOGLE_CUDA

#include "brightness_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_BRIGHTNESS_NAMESPACE_BEGIN


// Register a GPU kernel for Brightness
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("Brightness")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    Brightness<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for Brightness
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("Brightness")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    Brightness<GPUDevice, double, tensorflow::complex128>);



MONTBLANC_BRIGHTNESS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
