#include "post_process_visibilities_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'time_index'
    ShapeHandle in_time_index = c->input(0);
    // Assert 'time_index' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_time_index, 1, &input),
        "antenna1 must have shape [nvrow] but is " +
        c->DebugString(in_time_index));


    // TODO. Check shape and dimension sizes for 'antenna1'
    ShapeHandle in_antenna1 = c->input(1);
    // Assert 'antenna1' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_antenna1, 1, &input),
        "antenna1 must have shape [nvrow] but is " +
        c->DebugString(in_antenna1));

    // TODO. Check shape and dimension sizes for 'antenna2'
    ShapeHandle in_antenna2 = c->input(2);
    // Assert 'antenna2' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_antenna2, 1, &input),
        "antenna2 must have shape [nvrow] but is " +
        c->DebugString(in_antenna2));

    // TODO. Check shape and dimension sizes for 'direction_independent_effects'
    ShapeHandle in_die = c->input(3);
    // Assert 'direction_independent_effects' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_die, 4, &input),
        "direction_independent_effects must have shape [ntime, na, nchan, 4] but is " +
        c->DebugString(in_die));
    // Assert 'direction_independent_effects' dimension '3' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_die, 3), 4, &d),
        "direction_independent_effects must have shape [ntime, na, nchan, 4] but is " +
        c->DebugString(in_die));

    // TODO. Check shape and dimension sizes for 'flag'
    ShapeHandle in_flag = c->input(4);
    // Assert 'flag' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_flag, 3, &input),
        "flag must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_flag));
    // Assert 'flag' dimension '3' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_flag, 2), 4, &d),
        "flag must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_flag));

    // TODO. Check shape and dimension sizes for 'weight'
    ShapeHandle in_weight = c->input(5);
    // Assert 'weight' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_weight, 3, &input),
        "weight must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_weight));
    // Assert 'weight' dimension '3' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_weight, 2), 4, &d),
        "weight must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_weight));

    // TODO. Check shape and dimension sizes for 'base_vis'
    ShapeHandle in_base_vis = c->input(6);
    // Assert 'base_vis' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_base_vis, 3, &input),
        "base_vis must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_base_vis));
    // Assert 'base_vis' dimension '3' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_base_vis, 2), 4, &d),
        "base_vis must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_base_vis));


    // TODO. Check shape and dimension sizes for 'model_vis'
    ShapeHandle in_model_vis = c->input(7);
    // Assert 'model_vis' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_model_vis, 3, &input),
        "model_vis must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_model_vis));
    // Assert 'model_vis' dimension '3' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_model_vis, 2), 4, &d),
        "model_vis must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_model_vis));

    // TODO. Check shape and dimension sizes for 'observed_vis'
    ShapeHandle in_observed_vis = c->input(8);
    // Assert 'observed_vis' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_observed_vis, 3, &input),
        "observed_vis must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_observed_vis));
    // Assert 'observed_vis' dimension '3' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_observed_vis, 2), 4, &d),
        "observed_vis must have shape [nvrow, nchan, 4] but is " +
        c->DebugString(in_observed_vis));

    // Final visibilities have same shape as input visibilities
    ShapeHandle out_final_vis = c->MakeShape({
        c->Dim(in_model_vis, 0),
        c->Dim(in_model_vis, 1),
        c->Dim(in_model_vis, 2) });

    ShapeHandle out_chi_squared = c->MakeShape({  });

    c->set_output(0, out_final_vis);
    c->set_output(1, out_chi_squared);


    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the PostProcessVisibilities operator.
REGISTER_OP("PostProcessVisibilities")
    .Input("time_index: int32")
    .Input("antenna1: int32")
    .Input("antenna2: int32")
    .Input("direction_independent_effects: CT")
    .Input("flag: uint8")
    .Input("weight: FT")
    .Input("base_vis: CT")
    .Input("model_vis: CT")
    .Input("observed_vis: CT")
    .Output("final_vis: CT")
    .Output("chi_squared: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")

    // Input array schemas
    .Attr("time_index_schema: string = '(row,)'")
    .Attr("antenna1_schema: string = '(row,)'")
    .Attr("antenna2_schema: string = '(row,)'")
    .Attr("direction_independent_effects_schema: "
          "string = '(time, ant, chan, corr)'")
    .Attr("flag_schema: string = '(row, chan, corr)'")
    .Attr("weight_schema: string = '(row, chan, corr)'")
    .Attr("base_vis_schema: string = '(row, chan, corr)'")
    .Attr("observed_vis_schema: string = '(row, chan, corr)'")

    // Output array schemas
    .Attr("final_vis_schema: string = '(row, chan, corr)'")
    .Attr("chi_squared_schema: string = '()'")
    .Doc(R"doc(Post Processes Visibilities)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for PostProcessVisibilities
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("PostProcessVisibilities")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    PostProcessVisibilities<CPUDevice, float, tensorflow::complex64>);


// Register a CPU kernel for PostProcessVisibilities
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("PostProcessVisibilities")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    PostProcessVisibilities<CPUDevice, double, tensorflow::complex128>);



MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
