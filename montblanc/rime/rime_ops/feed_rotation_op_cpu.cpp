#include "feed_rotation_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ROTATION_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'parallactic_angle_sin'
    ShapeHandle in_parallactic_angle_sin = c->input(0);
    // Assert 'parallactic_angle_sin' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_parallactic_angle_sin, 2, &input),
        "parallactic_angle_sin must have shape [None, None] but is " +
        c->DebugString(in_parallactic_angle_sin));

    // TODO. Check shape and dimension sizes for 'parallactic_angle_cos'
    ShapeHandle in_parallactic_angle_cos = c->input(1);
    // Assert 'parallactic_angle_cos' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_parallactic_angle_cos, 2, &input),
        "parallactic_angle_cos must have shape [None, None] but is " +
        c->DebugString(in_parallactic_angle_cos));



    // TODO: Supply a proper shapes for output variables here,
    // usually derived from input shapes
    // ShapeHandle output_1 = c->MakeShape({
    //      c->Dim(input_1, 0),  // input_1 dimension 0
    //      c->Dim(input_2, 1)}); // input_2 dimension 1""")

    ShapeHandle out_feed_rotation = c->MakeShape({
        c->Dim(in_parallactic_angle_sin, 0),
        c->Dim(in_parallactic_angle_sin, 1),
        4
    });

    c->set_output(0, out_feed_rotation);


    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the FeedRotation operator.
REGISTER_OP("FeedRotation")
    .Input("parallactic_angle_sin: FT")
    .Input("parallactic_angle_cos: FT")
    .Output("feed_rotation: CT")
    .Attr("feed_type: {'linear', 'circular'}")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Doc(R"doc(Given the sin and cosine of the parallactic angle, compute the feed rotation matrix.)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for FeedRotation
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("FeedRotation")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    FeedRotation<CPUDevice, float, tensorflow::complex64>);

// Register a CPU kernel for FeedRotation
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("FeedRotation")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    FeedRotation<CPUDevice, double, tensorflow::complex128>);



MONTBLANC_FEED_ROTATION_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP