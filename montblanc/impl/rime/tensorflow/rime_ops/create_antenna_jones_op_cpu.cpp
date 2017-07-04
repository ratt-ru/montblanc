#include "create_antenna_jones_op_cpu.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto ekb_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // Get input shapes
    ShapeHandle bsqrt = c->input(0);
    ShapeHandle complex_phase = c->input(1);
    ShapeHandle feed_rotation = c->input(2);
    ShapeHandle ejones = c->input(3);

    // complex_phase
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(complex_phase, 4, &input),
        "complex_phase shape must be [nsrc, ntime, na, nchan] but is " +
        c->DebugString(complex_phase));

    // bsqrt
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(bsqrt, 4, &input),
        "bsqrt shape must be [nsrc, na, nchan, 4] but is " +
        c->DebugString(bsqrt));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(bsqrt, 3), 4, &d),
        "bsqrt shape must be [nsrc, na, nchan, 4] but is " +
        c->DebugString(bsqrt));

    // feed_rotation
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(feed_rotation, 3, &input),
        "bsqrt shape must be [ntime, na, 4] but is " +
        c->DebugString(feed_rotation));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(feed_rotation, 2), 4, &d),
        "bsqrt shape must be [ntime, na, 4] but is " +
        c->DebugString(feed_rotation));

    // ejones
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(ejones, 5, &input),
        "ejones shape must be [nsrc, ntime, na, nchan, 4] but is " +
        c->DebugString(ejones));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(ejones, 4), 4, &d),
        "ejones shape must be [nsrc, ntime, na, nchan, 4] but is " +
        c->DebugString(ejones));

    // ant_jones output is (nsrc, ntime, na, nchan, 4)
    ShapeHandle ant_jones = c->MakeShape({
        c->Dim(complex_phase, 0),
        c->Dim(complex_phase, 1),
        c->Dim(complex_phase, 2),
        c->Dim(complex_phase, 3),
        4});

    // Set the output shape
    c->set_output(0, ant_jones);

    return Status::OK();
};



// Register the CreateAntennaJones operator.
REGISTER_OP("CreateAntennaJones")
    .Input("bsqrt: CT")
    .Input("complex_phase: CT")
    .Input("feed_rotation: CT")
    .Input("ejones: CT")
    .Output("ant_jones: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(ekb_shape_function);


// Register a CPU kernel for CreateAntennaJones that handles floats
REGISTER_KERNEL_BUILDER(
    Name("CreateAntennaJones")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    CreateAntennaJones<CPUDevice, float, tensorflow::complex64>);

// Register a CPU kernel for CreateAntennaJones that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("CreateAntennaJones")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    CreateAntennaJones<CPUDevice, double, tensorflow::complex128>);


MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
