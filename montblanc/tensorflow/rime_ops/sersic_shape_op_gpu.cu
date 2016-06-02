#if GOOGLE_CUDA

#include "sersic_shape_op_gpu.cuh"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SERSIC_SHAPE_NAMESPACE_BEGIN

REGISTER_KERNEL_BUILDER(
    Name("SersicShape")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<float>("FT"),
    SersicShape<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("SersicShape")
    .Device(tensorflow::DEVICE_GPU)
    .TypeConstraint<double>("FT"),
    SersicShape<GPUDevice, double>);

MONTBLANC_SERSIC_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
