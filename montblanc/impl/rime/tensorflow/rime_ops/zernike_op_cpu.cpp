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
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_coords, 5, &input),
        "coords must have shape [3, None, None, None, None] but is " +
        c->DebugString(in_coords));
    // Assert 'coords' dimension '0' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_coords, 0), 3, &d),
        "coords must have shape [3, None, None, None, None] but is " +
        c->DebugString(in_coords));
    
    // TODO. Check shape and dimension sizes for 'coeffs'
    ShapeHandle in_coeffs = c->input(1);
    // Assert 'coeffs' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_coeffs, 3, &input),
        "coeffs must have shape [None, None, None] but is " +
        c->DebugString(in_coeffs));
    
    // TODO. Check shape and dimension sizes for 'noll_index'
    ShapeHandle in_noll_index = c->input(2);
    // Assert 'noll_index' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_noll_index, 3, &input),
        "noll_index must have shape [None, None, None] but is " +
        c->DebugString(in_noll_index));
    
    

    // TODO: Supply a proper shapes for output variables here,
    // usually derived from input shapes
    // ShapeHandle output_1 = c->MakeShape({
    //      c->Dim(input_1, 0),  // input_1 dimension 0
    //      c->Dim(input_2, 1)}); // input_2 dimension 1""")

    ShapeHandle out_zernike_value = c->MakeShape({ 
        c->Dim(in_coords, 1),  
        c->Dim(in_coords, 2),  
        c->Dim(in_coords, 3),  
        c->Dim(in_coords, 4)});
    
    c->set_output(0, out_zernike_value);
    

    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the Zernike operator.
REGISTER_OP("Zernike")
    .Input("coords: FT")
    .Input("coeffs: CT")
    .Input("noll_index: FT")
    .Output("zernike_value: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Doc(R"doc(Given tensors
  (1) of l, m, and frequency coordinates
  (2) of Zernike coefficients
  (3) of noll Zernike index
Compute the Zernike value with output tensor shape (source, time, ant, chan)
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