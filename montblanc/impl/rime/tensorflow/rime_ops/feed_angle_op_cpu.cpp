#include "feed_angle_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ANGLE_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'feed_angle'
    ShapeHandle in_feed_angle = c->input(0);
    // Assert 'feed_angle' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_feed_angle, 1, &input),
        "feed_angle must have shape [na] but is " +
        c->DebugString(in_feed_angle));

    // TODO. Check shape and dimension sizes for 'parallactic_angle'
    ShapeHandle in_parallactic_angle = c->input(1);
    // Assert 'parallactic_angle' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_parallactic_angle, 2, &input),
        "parallactic_angle must have shape [ntime, na] but is " +
        c->DebugString(in_parallactic_angle));

    ShapeHandle out_feed_angle_rotation = c->MakeShape({
        c->Dim(in_parallactic_angle, 0),
        c->Dim(in_parallactic_angle, 1),
        4 });

    c->set_output(0, out_feed_angle_rotation);

    return Status::OK();
};

// Register the FeedAngle operator.
REGISTER_OP("FeedAngle")
    .Input("feed_angle: FT")
    .Input("parallactic_angle: FT")
    .Output("feed_angle_rotation: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Doc(R"doc(Given tensors
  (1) of feed_angle with shape (na,)
  (2) of parallactic_angle with shape (ntime, na)
compute the feed angle rotation with shape (ntime, na, 4)
)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for FeedAngle
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("FeedAngle")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    FeedAngle<CPUDevice, float, tensorflow::complex64>);


// Register a CPU kernel for FeedAngle
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("FeedAngle")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    FeedAngle<CPUDevice, double, tensorflow::complex128>);



MONTBLANC_FEED_ANGLE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
