#if GOOGLE_CUDA

#include "sum_coherencies_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SUM_COHERENCIES_NAMESPACE_BEGIN

// Register a GPU kernel for SumCoherencies that handles floats
REGISTER_KERNEL_BUILDER(
    Name("SumCoherencies")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    SumCoherencies<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for SumCoherencies that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("SumCoherencies")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    SumCoherencies<GPUDevice, double, tensorflow::complex128>);

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
