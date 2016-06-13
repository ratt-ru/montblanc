#include "phase_op_cpu.h"

namespace montblanc {
namespace phase {

REGISTER_OP("Phase")
    .Input("lm: FT")
    .Input("uvw: FT")
    .Input("frequency: FT")
    .Output("complex_phase: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64");

REGISTER_KERNEL_BUILDER(
    Name("Phase")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    Phase<CPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("Phase")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    Phase<CPUDevice, double, tensorflow::complex128>);

} // namespace phase {
} // namespace montblanc {
