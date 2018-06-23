#include "sum_coherencies_op_cpu.h"
#include "shapes.h"

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

    TensorflowInputFacade<TFShapeInference> in_facade(c);

    TF_RETURN_IF_ERROR(in_facade.inspect({"time_index",
                                        "antenna1",
                                        "antenna2",
                                        "shape",
                                        "ant_jones",
                                        "sgn_brightness",
                                        "complex_phase",
                                        "base_coherencies"}));

    DimensionHandle nrow, nchan, ncorr;
    TF_RETURN_IF_ERROR(in_facade.get_dim("row", &nrow));
    TF_RETURN_IF_ERROR(in_facade.get_dim("chan", &nchan));
    TF_RETURN_IF_ERROR(in_facade.get_dim("corr", &ncorr));

    // Coherency output is (row, chan, corr)
    ShapeHandle coherencies = c->MakeShape({nrow, nchan, ncorr});

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
    .Input("sgn_brightness: have_sgn_brightness*int8")
    .Input("complex_phase: have_complex_phase*CT")
    .Input("base_coherencies: have_base_coherencies*CT")
    .Output("coherencies: CT")
    .Attr("FT: {double, float} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("have_sgn_brightness: int >= 0")
    .Attr("have_complex_phase: int >= 0")
    .Attr("have_base_coherencies: int >= 0")
    .Attr("time_index_schema: string = '(row,)'")
    .Attr("antenna1_schema: string = '(row,)'")
    .Attr("antenna2_schema: string = '(row,)'")
    .Attr("ant_jones_schema: string = '(source,time,ant,chan,corr)'")
    .Attr("base_coherencies_schema: string = '(row, chan, corr)'")
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
