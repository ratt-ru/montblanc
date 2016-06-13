#include "b_sqrt_op_cpu.h"

namespace montblanc {
namespace bsqrt {

REGISTER_OP("BSqrt")
    .Input("stokes: FT")
    .Input("alpha: FT")
    .Input("frequency: FT")
    .Input("ref_freq: FT")
    .Output("b_sqrt: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64");

REGISTER_KERNEL_BUILDER(
    Name("BSqrt")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    BSqrt<CPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("BSqrt")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    BSqrt<CPUDevice, double, tensorflow::complex128>);

} // namespace bsqrt {
} // namespace montblanc {
