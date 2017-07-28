#if GOOGLE_CUDA

#include "circular_stokes_swap_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_BEGIN


// Register a GPU kernel for CircularStokesSwap
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("CircularStokesSwap")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    CircularStokesSwap<GPUDevice, float>);

// Register a GPU kernel for CircularStokesSwap
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("CircularStokesSwap")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    CircularStokesSwap<GPUDevice, double>);



MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA