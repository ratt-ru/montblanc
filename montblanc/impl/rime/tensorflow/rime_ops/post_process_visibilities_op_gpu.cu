#if GOOGLE_CUDA

#include "post_process_visibilities_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_BEGIN


// Register a GPU kernel for PostProcessVisibilities
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("PostProcessVisibilities")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_GPU),
    PostProcessVisibilities<GPUDevice, float, tensorflow::complex64>);


// Register a GPU kernel for PostProcessVisibilities
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("PostProcessVisibilities")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_GPU),
    PostProcessVisibilities<GPUDevice, double, tensorflow::complex128>);



MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA