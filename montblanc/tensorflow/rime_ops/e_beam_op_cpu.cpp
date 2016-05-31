#include "e_beam_op_cpu.h"

namespace montblanc {
namespace ebeam {
REGISTER_OP("RimeEBeam")
    .Input("lm: FT")
    .Input("point_errors: FT")
    .Input("antenna_scaling: FT")
    .Input("e_beam: CT")
    .Input("parallactic_angle: FT")
    .Input("beam_ll: FT")
    .Input("beam_lm: FT")
    .Input("beam_ul: FT")
    .Input("beam_um: FT")
    .Output("jones: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64");

REGISTER_KERNEL_BUILDER(
    Name("RimeEBeam")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    RimeEBeam<CPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("RimeEBeam")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    RimeEBeam<CPUDevice, double, tensorflow::complex128>);

} // namespace ebeam {
} // namespace montblanc {
