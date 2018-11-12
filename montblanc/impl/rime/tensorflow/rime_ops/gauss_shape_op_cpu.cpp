#include "gauss_shape_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_GAUSS_SHAPE_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto gauss_shape_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // Get input shapes
    ShapeHandle uvw = c->input(0);
    ShapeHandle frequency = c->input(1);
    ShapeHandle params = c->input(2);

    // uvw should be shape (nvrows, 3)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(uvw, 2, &input),
        "uvw shape must be [nvrow, 3] but is " + c->DebugString(uvw));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(uvw, 1), 3, &d),
        "uvw shape must be [nvrow, 3] but is " + c->DebugString(uvw));

    // frequency should be shape (nchan,)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(frequency, 1, &input),
        "frequency shape must be [nchan,] but is " + c->DebugString(frequency));

    // params should be shape (3,ngsrc)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(params, 2, &input),
        "params shape must be [3, ngsrc] but is " + c->DebugString(params));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(params, 0), 3, &d),
        "params shape must be [3, ngsrc] but is " + c->DebugString(params));

    // Gauss shape output is (ngsrc, nvrow, nchan)
    ShapeHandle output = c->MakeShape({
        c->Dim(params, 1),
        c->Dim(uvw, 0),
        c->Dim(frequency, 0)});

    // Set the output shape
    c->set_output(0, output);

    return Status::OK();
};


REGISTER_OP("GaussShape")
    .Input("uvw: FT")
    .Input("frequency: FT")
    .Input("params: FT")
    .Output("gauss_shape: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("uvw_schema: string = '(row,(u,v,w))'")
    .Attr("frequency_schema: string = '(chan,)'")
    .Attr("params_schema: string = '(3,source)'")
    .SetShapeFn(gauss_shape_shape_function);

REGISTER_KERNEL_BUILDER(
    Name("GaussShape")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT"),
    GaussShape<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("GaussShape")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT"),
    GaussShape<CPUDevice, double>);

MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
