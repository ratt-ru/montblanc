#include "ekb_sqrt_op_cpu.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_EKB_SQRT_NAMESPACE_BEGIN

// Register the EKBSqrt operator.
REGISTER_OP("EKBSqrt")
    .Input("complex_phase: CT")
    .Input("bsqrt: CT")
    .Input("ejones: CT")
    .Output("ant_jones: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64");


// Register a CPU kernel for EKBSqrt that handles floats
REGISTER_KERNEL_BUILDER(
    Name("EKBSqrt")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    EKBSqrt<CPUDevice, float, tensorflow::complex64>);

// Register a CPU kernel for EKBSqrt that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("EKBSqrt")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    EKBSqrt<CPUDevice, double, tensorflow::complex128>);


MONTBLANC_EKB_SQRT_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
