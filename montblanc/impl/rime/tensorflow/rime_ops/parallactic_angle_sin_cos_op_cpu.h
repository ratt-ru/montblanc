#ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_CPU_H
#define RIME_PARALLACTIC_ANGLE_SIN_COS_OP_CPU_H

#include "parallactic_angle_sin_cos_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the ParallacticAngleSinCos op for CPUs
template <typename FT>
class ParallacticAngleSinCos<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit ParallacticAngleSinCos(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_parallactic_angle = context->input(0);

        int ntime = in_parallactic_angle.dim_size(0);
        int na = in_parallactic_angle.dim_size(1);

        // Allocate output tensors
        // Allocate space for output tensor 'pa_sin'
        tf::Tensor * pa_sin_ptr = nullptr;
        tf::TensorShape pa_sin_shape = tf::TensorShape({ ntime, na });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, pa_sin_shape, &pa_sin_ptr));
        // Allocate space for output tensor 'pa_cos'
        tf::Tensor * pa_cos_ptr = nullptr;
        tf::TensorShape pa_cos_shape = tf::TensorShape({ ntime, na });
        OP_REQUIRES_OK(context, context->allocate_output(
            1, pa_cos_shape, &pa_cos_ptr));

        // Extract Eigen tensors
        auto parallactic_angle = in_parallactic_angle.flat<FT>();
        auto pa_sin = pa_sin_ptr->flat<FT>();
        auto pa_cos = pa_cos_ptr->flat<FT>();

        #pragma omp parallel
        for(int pa=0; pa < parallactic_angle.size(); ++pa)
        {
            pa_sin(pa) = std::sin(parallactic_angle(pa));
            pa_cos(pa) = std::cos(parallactic_angle(pa));
        }
    }
};

MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_CPU_H