#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "create_antenna_jones_op_cpu.h"
#include "shapes.h"


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

    TensorflowInputFacade<TFShapeInference> in_facade(c);

    TF_RETURN_IF_ERROR(in_facade.inspect({"bsqrt", "complex_phase",
                                        "feed_rotation", "ddes"}));

    DimensionHandle nsrc, ntime, na, nchan, ncorr;
    TF_RETURN_IF_ERROR(in_facade.get_dim("source", &nsrc));
    TF_RETURN_IF_ERROR(in_facade.get_dim("time", &ntime));
    TF_RETURN_IF_ERROR(in_facade.get_dim("ant", &na));
    TF_RETURN_IF_ERROR(in_facade.get_dim("chan", &nchan));
    TF_RETURN_IF_ERROR(in_facade.get_dim("corr", &ncorr));

    ShapeHandle ant_jones = c->MakeShape({
        nsrc, ntime, na, nchan, ncorr});
    // Set the output shape
    c->set_output(0, ant_jones);

    return tensorflow::Status::OK();
};


// Register the CreateAntennaJones operator.
REGISTER_OP("CreateAntennaJones")
    .Input("bsqrt: bsqrt_type")
    .Input("complex_phase: complex_phase_type")
    .Input("feed_rotation: feed_rotation_type")
    .Input("ddes: ddes_type")
    .Output("ant_jones: CT")
    .Attr("bsqrt_type: list({complex64, complex128}) >= 0")
    .Attr("complex_phase_type: list({complex64, complex128}) >= 0")
    .Attr("feed_rotation_type: list({complex64, complex128}) >= 0")
    .Attr("ddes_type: list({complex64, complex128}) >= 0")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("bsqrt_schema: string = '(source,time,chan,corr)'")
    .Attr("complex_phase_schema: string = '(source,time,ant,chan)'")
    .Attr("feed_rotation_schema: string = '(time,ant,corr)'")
    .Attr("ddes_schema: string = '(source,time,ant,chan,corr)'")
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
