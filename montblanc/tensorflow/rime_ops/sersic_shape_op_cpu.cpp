#include "sersic_shape_op_cpu.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SERSIC_SHAPE_NAMESPACE_BEGIN

REGISTER_OP("SersicShape")
    .Input("uvw: FT")
    .Input("antenna1: int32")
    .Input("antenna2: int32")
    .Input("frequency: FT")
    .Input("params: FT")
    .Output("sersic_shape: FT")
    .Attr("FT: {float, double} = DT_FLOAT");

REGISTER_KERNEL_BUILDER(
    Name("SersicShape")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT"),
    SersicShape<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("SersicShape")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT"),
    SersicShape<CPUDevice, double>);

MONTBLANC_SERSIC_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
