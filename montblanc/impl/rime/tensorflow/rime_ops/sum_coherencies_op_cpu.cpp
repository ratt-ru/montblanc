#include "sum_coherencies_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SUM_COHERENCIES_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto sum_coherencies_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // Get input shapes
    ShapeHandle time_index = c->input(0);
    ShapeHandle antenna1 = c->input(1);
    ShapeHandle antenna2 = c->input(2);
    ShapeHandle shape = c->input(3);
    ShapeHandle ant_jones = c->input(4);
    ShapeHandle sgn_brightness = c->input(5);
    ShapeHandle base_coherencies = c->input(6);

    // time_index
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(time_index, 1, &input),
        "time_index shape must be [nvrows] but is " + c->DebugString(time_index));

    // antenna1
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(antenna1, 1, &input),
        "antenna1 shape must be [nvrows] but is " + c->DebugString(antenna1));

    // antenna2
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(antenna2, 1, &input),
        "antenna2 shape must be [nvrows] but is " + c->DebugString(antenna2));

    // shape
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(shape, 3, &input),
        "shape shape must be [nsrc, nvrows, nchan] but is " +
        c->DebugString(shape));

    // ant_jones
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(ant_jones, 5, &input),
        "ant_jones shape must be [nsrc, ntime, na, nchan, 4] but is " +
        c->DebugString(ant_jones));

    // sgn_brightness
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(sgn_brightness, 2, &input),
        "sgn_brightness shape must be [nsrc, ntime] but is " +
        c->DebugString(sgn_brightness));

    // base_coherencies
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(base_coherencies, 3, &input),
        "base_coherencies shape must be [nvrows, nchan, 4] but is " +
        c->DebugString(base_coherencies));

    // Coherency output is (nvrows, nchan, 4)
    ShapeHandle coherencies = c->MakeShape({
        c->Dim(base_coherencies, 0),
        c->Dim(base_coherencies, 1),
        c->Dim(base_coherencies, 2)});

    // Set the output shape
    c->set_output(0, coherencies);

    return Status::OK();
};


// Register the SumCoherencies operator.
REGISTER_OP("SumCoherencies")
    .Input("time_index: int32")
    .Input("antenna1: int32")
    .Input("antenna2: int32")
    .Input("shape: FT")
    .Input("ant_jones: CT")
    .Input("sgn_brightness: int8")
    .Input("base_coherencies: CT")
    .Output("coherencies: CT")
    .Attr("FT: {double, float} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(sum_coherencies_shape_function);

// Register a CPU kernel for SumCoherencies that handles floats
REGISTER_KERNEL_BUILDER(
    Name("SumCoherencies")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    SumCoherencies<CPUDevice, float, tensorflow::complex64>);

// Register a CPU kernel for SumCoherencies that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("SumCoherencies")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    SumCoherencies<CPUDevice, double, tensorflow::complex128>);

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
