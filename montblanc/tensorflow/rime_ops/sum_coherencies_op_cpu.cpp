#include "sum_coherencies_op_cpu.h"

namespace montblanc {
namespace sumcoherencies {

REGISTER_OP("RimeSumCoherencies")
    .Input("uvw: FT")
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
