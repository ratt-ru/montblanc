#ifndef RIME_E_BEAM_OP_CPU_H_
#define RIME_E_BEAM_OP_CPU_H_

#include "e_beam_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;    

template <typename FT, typename CT>
class RimeEBeam<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit RimeEBeam(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_point_errors = context->input(1);
        const tf::Tensor & in_antenna_scaling = context->input(2);
        const tf::Tensor & in_E_beam = context->input(3);
        const tf::Tensor & in_parallactic_angle = context->input(4);
        const tf::Tensor & in_beam_ll = context->input(5);
        const tf::Tensor & in_beam_lm = context->input(6);
        const tf::Tensor & in_beam_ul = context->input(7);
        const tf::Tensor & in_beam_um = context->input(8);

        OP_REQUIRES(context, in_lm.dims() == 2 && in_lm.dim_size(1) == 2,
            tf::errors::InvalidArgument("lm should be of shape (nsrc, 2)"))

        OP_REQUIRES(context, in_point_errors.dims() == 4
            && in_point_errors.dim_size(3) == 2,
            tf::errors::InvalidArgument("point_errors should be of shape "
                                        "(ntime, na, nchan, 2)"))

        OP_REQUIRES(context, in_antenna_scaling.dims() == 3
            && in_antenna_scaling.dim_size(2) == 2,
            tf::errors::InvalidArgument("antenna_scaling should be of shape "
                                        "(na, nchan, 2)"))

        OP_REQUIRES(context, in_E_beam.dims() == 4
            && in_E_beam.dim_size(3) == 4,
            tf::errors::InvalidArgument("E_Beam should be of shape "
                                        "(beam_lw, beam_mh, beam_nud, 4)"))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_parallactic_angle.shape()),
            tf::errors::InvalidArgument("parallactic_angle is not scalar: ",
                in_parallactic_angle.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_ll.shape()),
            tf::errors::InvalidArgument("in_beam_ll is not scalar: ",
                in_beam_ll.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_lm.shape()),
            tf::errors::InvalidArgument("in_beam_lm is not scalar: ",
                in_beam_lm.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_ul.shape()),
            tf::errors::InvalidArgument("in_beam_ul is not scalar: ",
                in_beam_ul.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_um.shape()),
            tf::errors::InvalidArgument("in_beam_um is not scalar: ",
                in_beam_um.shape().DebugString()))

        // Extract problem dimensions
        int nsrc = in_lm.dim_size(0);
        int ntime = in_point_errors.dim_size(0);
        int na = in_point_errors.dim_size(1);
        int nchan = in_point_errors.dim_size(2);

        // Reason about our output shape
        tf::TensorShape jones_shape({nsrc, ntime, na, nchan, 4});

        // Create a pointer for the jones result
        tf::Tensor * jones_ptr = nullptr;

        // Allocate memory for the jones
        OP_REQUIRES_OK(context, context->allocate_output(
            0, jones_shape, &jones_ptr));

        if (jones_ptr->NumElements() == 0)
            { return; }
    }
};

} // namespace tensorflow {

#endif // #ifndef RIME_E_BEAM_OP_CPU_H_