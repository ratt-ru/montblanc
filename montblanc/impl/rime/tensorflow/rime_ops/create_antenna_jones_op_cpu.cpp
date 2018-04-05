#include "create_antenna_jones_op_cpu.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_BEGIN

using tensorflow::errors::InvalidArgument;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto create_antenna_jones_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    bool have_bsqrt = false;
    bool have_complex_phase = false;
    bool have_feed_rotation = false;
    bool have_ddes = false;

    c->GetAttr("have_bsqrt", &have_bsqrt);
    c->GetAttr("have_complex_phase", &have_complex_phase);
    c->GetAttr("have_feed_rotation", &have_feed_rotation);
    c->GetAttr("have_ddes", &have_ddes);

    // Get input shapes
    ShapeHandle bsqrt = c->input(0);
    ShapeHandle complex_phase = c->input(1);
    ShapeHandle feed_rotation = c->input(2);
    ShapeHandle ddes = c->input(3);

    auto nsrc = c->UnknownDim();
    auto ntime = c->UnknownDim();
    auto na = c->UnknownDim();
    auto nchan = c->UnknownDim();
    auto npol = c->UnknownDim();

    auto update_dim = [&c](const std::string & name,
                        DimensionHandle & old_size,
                        DimensionHandle new_size) -> Status
    {
        if(old_size.SameHandle(c->UnknownDim()))
        {
            old_size = new_size;
        }
        else if(!old_size.SameHandle(new_size))
        {
            return Status(InvalidArgument(
                    "Previously set size '",  c->Value(old_size),
                    "' for dimension '", name,
                    "' does not equal new size '", c->Value(new_size), "'"));
        }

        return Status::OK();
    };

    // bsqrt
    if(have_bsqrt)
    {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(bsqrt, 4, &input),
            "bsqrt shape must be [nsrc, ntime, nchan, 4] but is " +
            c->DebugString(bsqrt));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(bsqrt, 3), 4, &d),
            "bsqrt shape must be [nsrc, ntime, nchan, 4] but is " +
            c->DebugString(bsqrt));

        update_dim("nsrc", nsrc, c->Dim(bsqrt, 0));
        update_dim("ntime", ntime, c->Dim(bsqrt, 1));
        update_dim("nchan", nchan, c->Dim(bsqrt, 2));
        update_dim("npol", npol, c->Dim(bsqrt, 3));
    }

    // complex_phase
    if(have_complex_phase)
    {
        // complex_phase
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(complex_phase, 4, &input),
            "complex_phase shape must be [nsrc, ntime, na, nchan] but is " +
            c->DebugString(complex_phase));

        update_dim("nsrc", nsrc, c->Dim(complex_phase, 0));
        update_dim("ntime", ntime, c->Dim(complex_phase, 1));
        update_dim("na", na, c->Dim(complex_phase, 2));
        update_dim("nchan", nchan, c->Dim(complex_phase, 3));
    }

    // feed_rotation
    if(have_feed_rotation)
    {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(feed_rotation, 3, &input),
            "bsqrt shape must be [ntime, na, 4] but is " +
            c->DebugString(feed_rotation));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(feed_rotation, 2), 4, &d),
            "bsqrt shape must be [ntime, na, 4] but is " +
            c->DebugString(feed_rotation));

        update_dim("ntime", ntime, c->Dim(feed_rotation, 0));
        update_dim("na", na, c->Dim(feed_rotation, 1));
    }

    // DDES
    if(have_ddes)
    {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(ddes, 5, &input),
            "ddes shape must be [nsrc, ntime, na, nchan, 4] but is " +
            c->DebugString(ddes));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(ddes, 4), 4, &d),
            "ddes shape must be [nsrc, ntime, na, nchan, 4] but is " +
            c->DebugString(ddes));

        update_dim("nsrc", nsrc, c->Dim(ddes, 0));
        update_dim("ntime", ntime, c->Dim(ddes, 1));
        update_dim("na", na, c->Dim(ddes, 2));
        update_dim("nchan", nchan, c->Dim(ddes, 3));
        update_dim("npol", npol, c->Dim(ddes, 4));
    }

    ShapeHandle ant_jones = c->MakeShape({nsrc, ntime, na, nchan, npol});
    // Set the output shape
    c->set_output(0, ant_jones);

    return Status::OK();
};

// Register the CreateAntennaJones operator.
REGISTER_OP("CreateAntennaJones")
    .Input("bsqrt: CT")
    .Input("complex_phase: CT")
    .Input("feed_rotation: CT")
    .Input("ddes: CT")
    .Output("ant_jones: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("have_bsqrt: bool = true")
    .Attr("have_complex_phase: bool = true")
    .Attr("have_feed_rotation: bool = true")
    .Attr("have_ddes: bool = true")
    .SetShapeFn(create_antenna_jones_shape_function);


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
