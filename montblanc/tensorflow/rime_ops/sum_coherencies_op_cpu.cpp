#include "sum_coherencies_op_cpu.h"

namespace montblanc {
namespace sumcoherencies {

REGISTER_OP("RimeSumCoherencies")
    .Input("uvw: FT")
    .Input("gauss_shape: FT")
    .Input("sersic_shape: FT")
    .Input("frequency: FT")
    .Input("antenna1: int32")
    .Input("antenna2: int32")
    .Input("antenna_jones: CT")
    .Input("flag: uint8")
    .Input("weight: FT")
    .Input("gterm: CT")
    .Input("observed_vis: CT")
    .Output("model_vis: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64");

REGISTER_KERNEL_BUILDER(
    Name("RimeSumCoherencies")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    RimeSumCoherencies<CPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("RimeSumCoherencies")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    RimeSumCoherencies<CPUDevice, double, tensorflow::complex128>);

} // namespace sumcoherencies {
} // namespace montblanc {
