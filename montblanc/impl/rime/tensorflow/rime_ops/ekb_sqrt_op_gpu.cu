#if GOOGLE_CUDA

#include "ekb_sqrt_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_EKB_SQRT_NAMESPACE_BEGIN

// Register a GPU kernel for EKBSqrt that handles floats
REGISTER_KERNEL_BUILDER(
    Name("EKBSqrt")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    EKBSqrt<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for EKBSqrt that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("EKBSqrt")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    EKBSqrt<GPUDevice, double, tensorflow::complex128>);

MONTBLANC_EKB_SQRT_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
