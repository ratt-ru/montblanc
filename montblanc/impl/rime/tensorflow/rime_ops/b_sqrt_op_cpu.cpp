#include "b_sqrt_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

namespace montblanc {
namespace bsqrt {

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto bsqrt_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    ShapeHandle stokes = c->input(0);

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(stokes, 4, &input),
        "stokes shape must be [nsrc, ntime, nchan, 4] but is " + c->DebugString(stokes));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(stokes, 3), 4, &d),
        "stokes shape must be [nsrc, ntime, nchan, 4] but is " + c->DebugString(stokes));

    // bsqrt output is (nsrc, ntime, nchan, 4)
    ShapeHandle bsqrt = c->MakeShape({
        c->Dim(stokes, 0),
        c->Dim(stokes, 1),
        c->Dim(stokes, 2),
        4});

    // sgn_brightness output is (nsrc, ntime, nchan)
    ShapeHandle sgn_brightness = c->MakeShape({
        c->Dim(stokes, 0),
        c->Dim(stokes, 1),
        c->Dim(stokes, 2),
    });

    // Set the output shape
    c->set_output(0, bsqrt);
    c->set_output(1, sgn_brightness);

    return Status::OK();
};

REGISTER_OP("BSqrt")
    .Input("stokes: FT")
    .Output("b_sqrt: CT")
    .Output("sgn_brightness: int8")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("polarisation_type: {'linear', 'circular'} = 'linear'")
    .SetShapeFn(bsqrt_shape_function);

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
