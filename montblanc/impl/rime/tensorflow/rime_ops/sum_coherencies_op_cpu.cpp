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
    ShapeHandle antenna1 = c->input(0);
    ShapeHandle antenna2 = c->input(1);
    ShapeHandle shape = c->input(2);
    ShapeHandle ant_jones = c->input(3);
    ShapeHandle sgn_brightness = c->input(4);
    ShapeHandle flag = c->input(5);
    ShapeHandle gterm = c->input(6);
    ShapeHandle model_vis_in = c->input(7);
    ShapeHandle apply_dies = c->input(8);

    // antenna1
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(antenna1, 2, &input),
        "antenna1 shape must be [ntime, nbl] but is " + c->DebugString(antenna1));

    // antenna2
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(antenna2, 2, &input),
        "antenna2 shape must be [ntime, nbl] but is " + c->DebugString(antenna2));

    // shape
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(shape, 4, &input),
        "shape shape must be [nsrc, ntime, nbl, nchan] but is " +
        c->DebugString(shape));

    // ant_jones
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(ant_jones, 5, &input),
        "ant_jones shape must be [nsrc, ntime, nbl, nchan, 4] but is " +
        c->DebugString(ant_jones));

    // sgn_brightness
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(sgn_brightness, 2, &input),
        "sgn_brightness shape must be [nsrc, ntime] but is " +
        c->DebugString(sgn_brightness));


    // flag
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(flag, 4, &input),
        "flag shape must be [ntime, nbl, nchan, 4] but is " +
        c->DebugString(flag));

    // gterm
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(gterm, 4, &input),
        "gterm shape must be [ntime, na, nchan, 4] but is " +
        c->DebugString(gterm));

    // model_vis_in
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(model_vis_in, 4, &input),
        "model_vis_in shape must be [ntime, nbl, nchan, 4] but is " +
        c->DebugString(model_vis_in));

    // apply_dies
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(apply_dies, 0, &input),
        "apply_dies must be scalar " + c->DebugString(apply_dies));

    // Model visibility output is (ntime, nbl, nchan, 4)
    ShapeHandle model_vis_out = c->MakeShape({
        c->Dim(model_vis_in, 0),
        c->Dim(model_vis_in, 1),
        c->Dim(model_vis_in, 2),
        c->Dim(model_vis_in, 3)});

    // Set the output shape
    c->set_output(0, model_vis_out);

    return Status::OK();
};


// Register the SumCoherencies operator.
REGISTER_OP("SumCoherencies")
    .Input("antenna1: int32")
    .Input("antenna2: int32")
    .Input("shape: FT")
    .Input("ant_jones: CT")
    .Input("sgn_brightness: int8")
    .Input("flag: uint8")
    .Input("gterm: CT")
    .Input("model_vis_in: CT")
    .Input("apply_dies: bool")
    .Output("model_vis_out: CT")
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
