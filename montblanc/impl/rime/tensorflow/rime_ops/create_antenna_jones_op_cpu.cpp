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
    InferenceInputDimSizes input_dim_sizes;
    InferenceDimSizes dim_sizes;

    // Get input shapes
    TF_RETURN_IF_ERROR(get_input_and_schema_for_inference(c, "bsqrt", input_dim_sizes));
    TF_RETURN_IF_ERROR(get_input_and_schema_for_inference(c, "complex_phase", input_dim_sizes));
    TF_RETURN_IF_ERROR(get_input_and_schema_for_inference(c, "feed_rotation", input_dim_sizes));
    TF_RETURN_IF_ERROR(get_input_and_schema_for_inference(c, "ddes", input_dim_sizes));

    TF_RETURN_IF_ERROR(merge_input_dims(c, input_dim_sizes, dim_sizes));

    ShapeHandle ant_jones = c->MakeShape({
        dim_sizes["source"],
        dim_sizes["time"],
        dim_sizes["ant"],
        dim_sizes["chan"],
        dim_sizes["corr"]});
    // Set the output shape
    c->set_output(0, ant_jones);

    return Status::OK();
};


// Register the CreateAntennaJones operator.
REGISTER_OP("CreateAntennaJones")
    .Input("bsqrt: have_bsqrt*CT")
    .Input("complex_phase: have_complex_phase*CT")
    .Input("feed_rotation: have_feed_rotation*CT")
    .Input("ddes: have_ddes*CT")
    .Output("ant_jones: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("have_bsqrt: int >= 0")
    .Attr("have_complex_phase: int >= 0")
    .Attr("have_feed_rotation: int >= 0")
    .Attr("have_ddes: int >= 0")
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
