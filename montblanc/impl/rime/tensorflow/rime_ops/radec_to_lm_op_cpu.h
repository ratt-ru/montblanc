#ifndef RIME_RADEC_TO_LM_OP_CPU_H
#define RIME_RADEC_TO_LM_OP_CPU_H

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "radec_to_lm_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_RADEC_TO_LM_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the RadecToLm op for CPUs
template <typename FT>
class RadecToLm<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit RadecToLm(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_radec = context->input(0);
        const auto & in_phase_centre = context->input(1);

        int nsrc = in_radec.dim_size(0);

        // Allocate output tensors
        // Allocate space for output tensor 'lm'
        tf::Tensor * lm_ptr = nullptr;
        tf::TensorShape lm_shape = tf::TensorShape({ nsrc, 2 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, lm_shape, &lm_ptr));

        // Extract Eigen tensors
        auto radec = in_radec.tensor<FT, 2>();
        auto phase_centre = in_phase_centre.tensor<FT, 1>();
        auto lm = lm_ptr->tensor<FT, 2>();

        // Sin and cosine of phase centre DEC
        auto sin_d0 = sin(phase_centre(1));
        auto cos_d0 = cos(phase_centre(1));

        for(int src=0; src < nsrc; ++src)
        {
            // Sin and cosine of (source RA - phase centre RA)
            auto da = radec(src, 0) - phase_centre(0);
            auto sin_da = sin(da);
            auto cos_da = cos(da);

            // Sine and cosine of source DEC
            auto sin_d =  sin(radec(src, 1));
            auto cos_d =  cos(radec(src, 1));

            lm(src, 0) = cos_d*sin_da;
            lm(src, 1) = sin_d*cos_d0 - cos_d*sin_d0*cos_da;
        }
    }
};

MONTBLANC_RADEC_TO_LM_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_RADEC_TO_LM_OP_CPU_H
