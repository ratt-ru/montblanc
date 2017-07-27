#if GOOGLE_CUDA

#include "feed_rotation_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ROTATION_NAMESPACE_BEGIN


// Register a GPU kernel for FeedRotation
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("FeedRotation")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    FeedRotation<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for FeedRotation
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("FeedRotation")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    FeedRotation<GPUDevice, double, tensorflow::complex128>);



MONTBLANC_FEED_ROTATION_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA