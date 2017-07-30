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
    ShapeHandle alpha = c->input(1);
    ShapeHandle frequency = c->input(2);
    ShapeHandle ref_freq = c->input(3);

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(stokes, 3, &input),
        "stokes shape must be [nsrc, ntime, 4] but is " + c->DebugString(stokes));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(stokes, 2), 4, &d),
        "stokes shape must be [nsrc, ntime, 4] but is " + c->DebugString(stokes));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(alpha, 2, &input),
        "alpha shape must be [nsrc, ntime] but is " + c->DebugString(alpha));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(frequency, 1, &input),
        "frequency shape must be [nchan,] but is " + c->DebugString(frequency));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(ref_freq, 1, &input),
        "ref_freq shape must be [nsrc,] but is " + c->DebugString(ref_freq));

    // bsqrt output is (nsrc, ntime, nchan, 4)
    ShapeHandle bsqrt = c->MakeShape({
        c->Dim(stokes, 0),
        c->Dim(stokes, 1),
        c->Dim(frequency, 0),
        4});

    // sgn_brightness output is (nsrc, ntime)
    ShapeHandle sgn_brightness = c->MakeShape({
        c->Dim(stokes, 0),
        c->Dim(stokes, 1)});

    // Set the output shape
    c->set_output(0, bsqrt);
    c->set_output(1, sgn_brightness);

    return Status::OK();
};

REGISTER_OP("BSqrt")
    .Input("stokes: FT")
    .Input("alpha: FT")
    .Input("frequency: FT")
    .Input("ref_freq: FT")
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
