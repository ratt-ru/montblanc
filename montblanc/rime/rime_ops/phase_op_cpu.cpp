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

    // Must be at least size 2
    auto lm_status = c->WithRankAtLeast(lm, 2, &input);
    // Last dimension should be exactly 2
    lm_status.Update(c->WithValue(c->Dim(lm, c->Rank(lm)-1), 2, &d));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(lm_status,
        "lm shape must be [source_0, ..., source_n, 2]");

    // Must be at least size 2
    auto uvw_status = c->WithRankAtLeast(uvw, 2, &input);
    // Last dimension should be exactly 3
    uvw_status.Update(c->WithValue(c->Dim(uvw, c->Rank(uvw)-1), 3, &d));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(uvw_status,
        "uvw shape must be [uvw_0, ..., uvw_n, 3]");

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(frequency, 1, &input),
        "frequency shape must be [chan, ]");

    ShapeHandle lm_sub;
    ShapeHandle uvw_sub;

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Subshape(lm, 0, -1, &lm_sub),
        "Couldn't extract lm subshape");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Subshape(uvw, 0, -1, &uvw_sub),
        "Couldn't extract uvw subshape");

    ShapeHandle out_shape;
    c->Concatenate(lm_sub, uvw_sub, &out_shape);
    c->Concatenate(out_shape, frequency, &out_shape);

    c->set_output(0, out_shape);

    return Status::OK();
};

REGISTER_OP("Phase")
    .Input("lm: FT")
    .Input("uvw: FT")
    .Input("frequency: FT")
    .Output("complex_phase: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("lm_schema: string = '(source, (l,m))'")
    .Attr("uvw_schema: string = '(time, ant, (u,v,w))'")
    .Attr("frequency_schema: string = '(chan,)'")
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
