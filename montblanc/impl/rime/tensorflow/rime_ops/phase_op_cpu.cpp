#include <cstdlib>

#include "phase_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

namespace montblanc {
namespace phase {

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto phase_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // Get input shapes
    ShapeHandle lm = c->input(0);
    ShapeHandle uvw = c->input(1);
    ShapeHandle frequency = c->input(2);

    // lm should be shape (nsrc, 2)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRankAtLeast(lm, 2, &input),
        "lm shape must be [nsrc, 2] but is " + c->DebugString(lm));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(lm, 1), 2, &d),
        "lm shape must be [nsrc, 2] but is " + c->DebugString(lm));

    // uvw should be shape (ntime, na, 3)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(uvw, 3, &input),
        "uvw shape must be [ntime, na, 3] but is " + c->DebugString(uvw));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(uvw, 2), 3, &d),
        "uvw shape must be [ntime, na, 3] but is " + c->DebugString(uvw));

    // frequency should be shape (nchan,)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(frequency, 1, &input),
        "frequency shape must be [nchan,] but is " + c->DebugString(frequency));

    // Complex phase output is (nsrc, ntime, na, nchan)
    ShapeHandle output = c->MakeShape({
        c->Dim(lm, 0),
        c->Dim(uvw, 0),
        c->Dim(uvw, 1),
        c->Dim(frequency, 0)});

    // Set the output shape
    c->set_output(0, output);

    return Status::OK();
};

REGISTER_OP("Phase")
    .Input("lm: FT")
    .Input("uvw: FT")
    .Input("frequency: FT")
    .Output("complex_phase: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(phase_shape_function);

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
