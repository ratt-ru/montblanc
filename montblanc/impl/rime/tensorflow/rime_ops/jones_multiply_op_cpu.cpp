#include "jones_multiply_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    return Status::OK();
};

// Register the JonesMultiply operator.
REGISTER_OP("JonesMultiply")
    .Input("in: N * CT")
    .Output("out: CT")
    .Attr("N: int >= 1")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("schemas: list(string)")
    .Attr("output_schema: string = '(source,time,ant,chan,corr)'")
    .Doc(R"doc(Jones Matrix Multiplication)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for JonesMultiply
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("JonesMultiply")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    JonesMultiply<CPUDevice, float, tensorflow::complex64>);

// Register a CPU kernel for JonesMultiply
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("JonesMultiply")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    JonesMultiply<CPUDevice, double, tensorflow::complex128>);



MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
