#include "gauss_shape_op_cpu.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_GAUSS_SHAPE_NAMESPACE_BEGIN

REGISTER_OP("GaussShape")
    .Input("uvw: FT")
    .Input("antenna1: int32")
    .Input("antenna2: int32")
    .Input("frequency: FT")
    .Input("params: FT")
    .Output("gauss_shape: FT")
    .Attr("FT: {float, double} = DT_FLOAT");

REGISTER_KERNEL_BUILDER(
    Name("GaussShape")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT"),
    GaussShape<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("GaussShape")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT"),
    GaussShape<CPUDevice, double>);

MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
