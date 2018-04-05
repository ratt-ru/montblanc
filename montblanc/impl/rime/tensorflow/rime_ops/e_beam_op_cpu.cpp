#include "e_beam_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

namespace montblanc {
namespace ebeam {

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto ebeam_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // Get input shapes
    ShapeHandle lm = c->input(0);
    ShapeHandle frequency = c->input(1);
    ShapeHandle point_errors = c->input(2);
    ShapeHandle antenna_scaling = c->input(3);
    ShapeHandle parallactic_angle_sin = c->input(4);
    ShapeHandle parallactic_angle_cos = c->input(5);
    ShapeHandle beam_extents = c->input(6);
    ShapeHandle beam_freq_map = c->input(7);
    ShapeHandle ebeam = c->input(8);

    // lm should be shape (nsrc, 2)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(lm, 2, &input),
        "lm shape must be [nsrc, 2] but is " + c->DebugString(lm));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(lm, 1), 2, &d),
        "lm shape must be [nsrc, 2] but is " + c->DebugString(lm));

    // frequency should be shape (nchan,)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(frequency, 1, &input),
        "frequency shape must be [nchan,] but is " + c->DebugString(frequency));

    // point errors should be shape (ntime, na, nchan, 2)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(point_errors, 4, &input),
        "point_errors shape must be [ntime, na, nchan, 2] but is " +
        c->DebugString(point_errors));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(point_errors, 3), 2, &d),
        "point_errors shape must be [ntime, na, nchan, 2] but is " +
        c->DebugString(point_errors));

    // antenna scaling should be shape (na, nchan, 2)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(antenna_scaling, 3, &input),
        "point_errors shape must be [na, nchan, 2] but is " +
        c->DebugString(antenna_scaling));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(antenna_scaling, 2), 2, &d),
        "point_errors shape must be [na, nchan, 2] but is " +
        c->DebugString(antenna_scaling));

    // parallactic angle_sin should be shape (ntime, na)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(parallactic_angle_sin, 2, &input),
        "parallactic_angle shape_sin must be [ntime, na] but is " +
        c->DebugString(parallactic_angle_sin));

    // parallactic angle_cos should be shape (ntime, na)
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(parallactic_angle_cos, 2, &input),
        "parallactic_angle_cos shape must be [ntime, na] but is " +
        c->DebugString(parallactic_angle_cos));

    // beam_extents
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(beam_extents, 1, &input),
        "beam_extents shape must be [6,] but is " +
        c->DebugString(beam_extents));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(beam_extents, 0), 6, &d),
        "beam_extents shape must be [6,] but is " +
        c->DebugString(beam_extents));

    // beam_freq_map
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(beam_freq_map, 1, &input),
        "beam_freq_map shape must be [beam_nud,] but is " +
        c->DebugString(beam_freq_map));

    // ebeam
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(ebeam, 4, &input),
        "ebeam should shape must be [beam_lw, beam_mh, beam_nud, 4] but is " +
        c->DebugString(ebeam));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(ebeam, 3), 4, &d),
        "ebeam shape must be [beam_lw, beam_mh, beam_nud, 4] but is " +
        c->DebugString(ebeam));

    // E Jones output is (nsrc, ntime, na, nchan, 4)
    ShapeHandle ejones = c->MakeShape({
        c->Dim(lm, 0),
        c->Dim(parallactic_angle_sin, 0),
        c->Dim(parallactic_angle_sin, 1),
        c->Dim(frequency, 0),
        4});

    // Set the output shape
    c->set_output(0, ejones);

    return Status::OK();
};


REGISTER_OP("EBeam")
    .Input("lm: FT")
    .Input("frequency: FT")
    .Input("point_errors: FT")
    .Input("antenna_scaling: FT")
    .Input("parallactic_angle_sin: FT")
    .Input("parallactic_angle_cos: FT")
    .Input("beam_extents: FT")
    .Input("beam_freq_map: FT")
    .Input("e_beam: CT")
    .Output("jones: CT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(ebeam_shape_function);

REGISTER_KERNEL_BUILDER(
    Name("EBeam")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT"),
    EBeam<CPUDevice, float, tensorflow::complex64>);

REGISTER_KERNEL_BUILDER(
    Name("EBeam")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT"),
    EBeam<CPUDevice, double, tensorflow::complex128>);

} // namespace ebeam {
} // namespace montblanc {
