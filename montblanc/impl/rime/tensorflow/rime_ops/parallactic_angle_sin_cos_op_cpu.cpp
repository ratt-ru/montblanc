#include "parallactic_angle_sin_cos_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'parallactic_angle'
    ShapeHandle in_parallactic_angle = c->input(0);
    // Assert 'parallactic_angle' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_parallactic_angle, 1, &input),
        "parallactic_angle must have shape [arow] but is " +
        c->DebugString(in_parallactic_angle));

    ShapeHandle out = c->MakeShape({c->Dim(in_parallactic_angle, 0)});

    c->set_output(0, out);
    c->set_output(1, out);


    return Status::OK();
};

// Register the ParallacticAngleSinCos operator.
REGISTER_OP("ParallacticAngleSinCos")
    .Input("parallactic_angle: FT")
    .Output("pa_sin: FT")
    .Output("pa_cos: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(Given the parallactic angle, returns the sine and cosine of the angle.)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for ParallacticAngleSinCos
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("ParallacticAngleSinCos")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    ParallacticAngleSinCos<CPUDevice, float>);

// Register a CPU kernel for ParallacticAngleSinCos
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("ParallacticAngleSinCos")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    ParallacticAngleSinCos<CPUDevice, double>);



MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
