#if GOOGLE_CUDA

#include "radec_to_lm_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_RADEC_TO_LM_NAMESPACE_BEGIN


// Register a GPU kernel for RadecToLm
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("RadecToLm")
    .TypeConstraint<float>("FT")
    .HostMemory("phase_centre")
    .Device(tensorflow::DEVICE_GPU),
    RadecToLm<GPUDevice, float>);

// Register a GPU kernel for RadecToLm
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("RadecToLm")
    .TypeConstraint<double>("FT")
    .HostMemory("phase_centre")
    .Device(tensorflow::DEVICE_GPU),
    RadecToLm<GPUDevice, double>);



MONTBLANC_RADEC_TO_LM_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
