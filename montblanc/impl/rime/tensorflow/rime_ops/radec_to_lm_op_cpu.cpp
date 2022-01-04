#include "radec_to_lm_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_RADEC_TO_LM_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function(InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'radec'
    ShapeHandle in_radec = c->input(0);
    // Assert 'radec' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_radec, 2, &input),
        "radec must have shape [None, 2] but is " +
        c->DebugString(in_radec));
    // Assert 'radec' dimension '1' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_radec, 1), 2, &d),
        "radec must have shape [None, 2] but is " +
        c->DebugString(in_radec));
    
    // TODO. Check shape and dimension sizes for 'phase_centre'
    ShapeHandle in_phase_centre = c->input(1);
    // Assert 'phase_centre' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_phase_centre, 1, &input),
        "phase_centre must have shape [2] but is " +
        c->DebugString(in_phase_centre));
    // Assert 'phase_centre' dimension '0' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_phase_centre, 0), 2, &d),
        "phase_centre must have shape [2] but is " +
        c->DebugString(in_phase_centre));
    


    // TODO: Supply a proper shapes for output variables here,
    // usually derived from input shapes
    // ShapeHandle output_1 = c->MakeShape({
    //      c->Dim(input_1, 0),  // input_1 dimension 0
    //      c->Dim(input_2, 1)}); // input_2 dimension 1""")

    ShapeHandle out_lm = c->MakeShape({ c->Dim(in_radec, 0), 2 });

    c->set_output(0, out_lm);


    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the RadecToLm operator.
REGISTER_OP("RadecToLm")
    .Input("radec: FT")
    .Input("phase_centre: FT")
    .Output("lm: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(Given tensors
  (1) of (ra, dec) sky coordinates with shape (nsrc, 2)
  (2) phase_centre with shape (2,)
compute the LM coordinates
)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for RadecToLm
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("RadecToLm")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    RadecToLm<CPUDevice, float>);

// Register a CPU kernel for RadecToLm
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("RadecToLm")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    RadecToLm<CPUDevice, double>);



MONTBLANC_RADEC_TO_LM_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
