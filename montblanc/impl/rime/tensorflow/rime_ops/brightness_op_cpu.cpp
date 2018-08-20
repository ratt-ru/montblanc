#include "brightness_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_BRIGHTNESS_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    c->set_output(0, c->input(0));

    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the Brightness operator.
REGISTER_OP("Brightness")
    .Input("stokes: FT")
    .Output("brightness: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Doc(R"doc(Stokes parameters
)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for Brightness
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("Brightness")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    Brightness<CPUDevice, float, tensorflow::complex64>);


// Register a CPU kernel for Brightness
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("Brightness")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    Brightness<CPUDevice, double, tensorflow::complex128>);



MONTBLANC_BRIGHTNESS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
