#if GOOGLE_CUDA

#include "gauss_shape_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_GAUSS_SHAPE_NAMESPACE_BEGIN

REGISTER_KERNEL_BUILDER(
    Name("GaussShape")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<float>("FT"),
    GaussShape<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("GaussShape")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<double>("FT"),
    GaussShape<GPUDevice, double>);

MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
