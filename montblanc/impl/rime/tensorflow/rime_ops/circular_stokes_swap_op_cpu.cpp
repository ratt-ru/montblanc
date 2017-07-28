#include "circular_stokes_swap_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'stokes_in'
    ShapeHandle in_stokes_in = c->input(0);
    // Assert 'stokes_in' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_stokes_in, 3, &input),
        "stokes_in must have shape [nsrc, ntime, 4] but is " +
        c->DebugString(in_stokes_in));
    // Assert 'stokes_in' dimension '2' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_stokes_in, 2), 4, &d),
        "stokes_in must have shape [nsrc, ntime, 4] but is " +
        c->DebugString(in_stokes_in));

    c->set_output(0, in_stokes_in);
    return Status::OK();
};

// Register the CircularStokesSwap operator.
REGISTER_OP("CircularStokesSwap")
    .Input("stokes_in: FT")
    .Output("stokes_out: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(Swaps Stokes parameters around so that a circular brightness matrix is created, rather than a)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for CircularStokesSwap
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("CircularStokesSwap")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    CircularStokesSwap<CPUDevice, float>);

// Register a CPU kernel for CircularStokesSwap
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("CircularStokesSwap")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    CircularStokesSwap<CPUDevice, double>);



MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP