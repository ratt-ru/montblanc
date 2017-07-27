#if GOOGLE_CUDA

#include "parallactic_angle_sin_cos_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_BEGIN


// Register a GPU kernel for ParallacticAngleSinCos
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("ParallacticAngleSinCos")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    ParallacticAngleSinCos<GPUDevice, float>);

// Register a GPU kernel for ParallacticAngleSinCos
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("ParallacticAngleSinCos")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    ParallacticAngleSinCos<GPUDevice, double>);



MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA