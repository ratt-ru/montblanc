#if GOOGLE_CUDA

#include "feed_angle_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ANGLE_NAMESPACE_BEGIN


// Register a GPU kernel for FeedAngle
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("FeedAngle")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    FeedAngle<GPUDevice, float, tensorflow::complex64>);

// Register a GPU kernel for FeedAngle
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("FeedAngle")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    FeedAngle<GPUDevice, double, tensorflow::complex128>);



MONTBLANC_FEED_ANGLE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
