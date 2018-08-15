#if GOOGLE_CUDA

#include "zernike_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_ZERNIKE_NAMESPACE_BEGIN


// Register a GPU kernel for Zernike
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("Zernike")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    Zernike<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for Zernike
// handling permutation ['float', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("Zernike")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    Zernike<GPUDevice, float, tensorflow::complex128>);

// Register a GPU kernel for Zernike
// handling permutation ['double', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("Zernike")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    Zernike<GPUDevice, double, tensorflow::complex64>);

// Register a GPU kernel for Zernike
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("Zernike")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    Zernike<GPUDevice, double, tensorflow::complex128>);



MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA