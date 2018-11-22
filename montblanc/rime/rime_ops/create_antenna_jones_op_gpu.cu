#if GOOGLE_CUDA

#include "create_antenna_jones_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_BEGIN

// Register a GPU kernel for CreateAntennaJones that handles floats
REGISTER_KERNEL_BUILDER(
    Name("CreateAntennaJones")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    CreateAntennaJones<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for CreateAntennaJones that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("CreateAntennaJones")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    CreateAntennaJones<GPUDevice, double, tensorflow::complex128>);

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
