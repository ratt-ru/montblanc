#include "zernike_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_ZERNIKE_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'coords'
    ShapeHandle in_coords = c->input(0);
    // Assert 'coords' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_coords, 3, &input),
        "coords must have shape [None, None, 2] but is " +
        c->DebugString(in_coords));
    // Assert 'coords' dimension '2' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_coords, 2), 2, &d),
        "coords must have shape [None, None, 2] but is " +
        c->DebugString(in_coords));
    
    // TODO. Check shape and dimension sizes for 'coeffs'
    ShapeHandle in_coeffs = c->input(1);
    // Assert 'coeffs' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_coeffs, 4, &input),
        "coeffs must have shape [None, None, None, None] but is " +
        c->DebugString(in_coeffs));
    
    // TODO. Check shape and dimension sizes for 'noll_index'
    ShapeHandle in_noll_index = c->input(2);
    // Assert 'noll_index' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_noll_index, 4, &input),
        "noll_index must have shape [None, None, None, None] but is " +
        c->DebugString(in_noll_index));
    
    // TODO. Check shape and dimension sizes for 'pointing_error'
    ShapeHandle in_pointing_error = c->input(3);
    // Assert 'pointing_error' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_pointing_error, 5, &input),
        "pointing_error must have shape [None, None, None, None, 2] but is " +
        c->DebugString(in_pointing_error));
    // Assert 'pointing_error' dimension '4' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_pointing_error, 4), 2, &d),
        "pointing_error must have shape [None, None, None, None, 2] but is " +
        c->DebugString(in_pointing_error));
    
    // TODO. Check shape and dimension sizes for 'antenna_scaling'
    ShapeHandle in_antenna_scaling = c->input(4);
    // Assert 'antenna_scaling' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_antenna_scaling, 4, &input),
        "antenna_scaling must have shape [None, None, None, 2] but is " +
        c->DebugString(in_antenna_scaling));
    // Assert 'antenna_scaling' dimension '3' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_antenna_scaling, 3), 2, &d),
        "antenna_scaling must have shape [None, None, None, 2] but is " +
        c->DebugString(in_antenna_scaling));
    
    

    // TODO: Supply a proper shapes for output variables here,
    // usually derived from input shapes
    // ShapeHandle output_1 = c->MakeShape({
    //      c->Dim(input_1, 0),  // input_1 dimension 0
    //      c->Dim(input_2, 1)}); // input_2 dimension 1""")

    ShapeHandle out_zernike_value = c->MakeShape({ 
        c->Dim(in_coords, 0), 
        c->Dim(in_coords, 1), 
        c->Dim(in_pointing_error, 1), 
        c->Dim(in_coeffs, 1), 
        c->Dim(in_coeffs, 2) });
    
    c->set_output(0, out_zernike_value);
    

    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the Zernike operator.
REGISTER_OP("Zernike")
    .Input("coords: FT")
    .Input("coeffs: CT")
    .Input("noll_index: int32")
    .Input("pointing_error: FT")
    .Input("antenna_scaling: FT")
    .Output("zernike_value: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Doc(R"doc(Given tensors
  (1) of (l, m) coordinates
  (2) of Zernike coefficients
  (3) of noll Zernike index
  (4) of pointing error
  (5) of antenna scaling
Compute the Zernike value with output tensor shape (ncorr, source, time, ant, chan)
)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for Zernike
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("Zernike")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    Zernike<CPUDevice, float, tensorflow::complex64>);


// Register a CPU kernel for Zernike
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("Zernike")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    Zernike<CPUDevice, double, tensorflow::complex128>);



MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP