#ifndef RIME_POST_PROCESS_VISIBILITIES_OP_CPU_H
#define RIME_POST_PROCESS_VISIBILITIES_OP_CPU_H

#include "post_process_visibilities_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Compute chi-squared term from correlation and associated weight
template <typename FT, typename CT>
FT chi_squared_term(const CT & model_corr,
    const CT & observed_corr,
    const FT & weight)
{
    CT d = model_corr - observed_corr;
    return (d.real()*d.real() + d.imag()*d.imag())*weight;
}

// Specialise the PostProcessVisibilities op for CPUs
template <typename FT, typename CT>
class PostProcessVisibilities<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit PostProcessVisibilities(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_antenna1 = context->input(0);
        const auto & in_antenna2 = context->input(1);
        const auto & in_direction_independent_effects = context->input(2);
        const auto & in_flag = context->input(3);
        const auto & in_weight = context->input(4);
        const auto & in_base_vis = context->input(5);
        const auto & in_model_vis = context->input(6);
        const auto & in_observed_vis = context->input(7);

        int ntime = in_model_vis.dim_size(0);
        int nbl = in_model_vis.dim_size(1);
        int nchan = in_model_vis.dim_size(2);
        int npol = in_model_vis.dim_size(3);

        // Allocate output tensors
        // Allocate space for output tensor 'final_vis'
        tf::Tensor * final_vis_ptr = nullptr;
        tf::TensorShape final_vis_shape = tf::TensorShape({ ntime, nbl, nchan, npol });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, final_vis_shape, &final_vis_ptr));
        // Allocate space for output tensor 'chi_squared'
        tf::Tensor * chi_squared_ptr = nullptr;
        tf::TensorShape chi_squared_shape = tf::TensorShape({  });
        OP_REQUIRES_OK(context, context->allocate_output(
            1, chi_squared_shape, &chi_squared_ptr));

        // Extract Eigen tensors
        auto antenna1 = in_antenna1.tensor<tensorflow::int32, 2>();
        auto antenna2 = in_antenna2.tensor<tensorflow::int32, 2>();
        auto direction_independent_effects = in_direction_independent_effects.tensor<CT, 4>();
        auto flag = in_flag.tensor<tensorflow::uint8, 4>();
        auto weight = in_weight.tensor<FT, 4>();
        auto base_vis = in_base_vis.tensor<CT, 4>();
        auto model_vis = in_model_vis.tensor<CT, 4>();
        auto observed_vis = in_observed_vis.tensor<CT, 4>();

        auto final_vis = final_vis_ptr->tensor<CT, 4>();
        auto chi_squared = chi_squared_ptr->tensor<FT, 0>();

        // Initialise a float to store the chi squared result,
        // needed for the OpenMP reduction below
        FT chi_squared_ = FT(0);

        #pragma omp parallel for collapse(2) reduction(+:chi_squared_)
        for(int time=0; time < ntime; ++time)
        {
            for(int bl=0; bl < nbl; ++bl)
            {
                int ant1 = antenna1(time, bl);
                int ant2 = antenna2(time, bl);

                for(int chan=0; chan < nchan; ++chan)
                {
                    // Load in current model visibilities
                    CT mv0 = model_vis(time, bl, chan, 0);
                    CT mv1 = model_vis(time, bl, chan, 1);
                    CT mv2 = model_vis(time, bl, chan, 2);
                    CT mv3 = model_vis(time, bl, chan, 3);

                    // Reference direction_independent_effects for antenna 1
                    const CT & a0 = direction_independent_effects(time, ant1, chan, 0);
                    const CT & a1 = direction_independent_effects(time, ant1, chan, 1);
                    const CT & a2 = direction_independent_effects(time, ant1, chan, 2);
                    const CT & a3 = direction_independent_effects(time, ant1, chan, 3);

                    // Multiply model visibilities by antenna 1 g
                    CT r0 = a0*mv0 + a1*mv2;
                    CT r1 = a0*mv1 + a1*mv3;
                    CT r2 = a2*mv0 + a3*mv2;
                    CT r3 = a2*mv1 + a3*mv3;

                    // Conjugate transpose of antenna 2 g term
                    CT b0 = std::conj(direction_independent_effects(time, ant2, chan, 0));
                    CT b1 = std::conj(direction_independent_effects(time, ant2, chan, 2));
                    CT b2 = std::conj(direction_independent_effects(time, ant2, chan, 1));
                    CT b3 = std::conj(direction_independent_effects(time, ant2, chan, 3));

                    // Multiply to produce model visibilities
                    mv0 = r0*b0 + r1*b2;
                    mv1 = r0*b1 + r1*b3;
                    mv2 = r2*b0 + r3*b2;
                    mv3 = r2*b1 + r3*b3;

                    // Add base visibilities
                    mv0 += base_vis(time, bl, chan, 0);
                    mv1 += base_vis(time, bl, chan, 1);
                    mv2 += base_vis(time, bl, chan, 2);
                    mv3 += base_vis(time, bl, chan, 3);

                    // Flags
                    bool f0 = flag(time, bl, chan, 0) > 0;
                    bool f1 = flag(time, bl, chan, 1) > 0;
                    bool f2 = flag(time, bl, chan, 2) > 0;
                    bool f3 = flag(time, bl, chan, 3) > 0;

                    // Write out model visibilities, zeroed if flagged
                    final_vis(time, bl, chan, 0) = f0 ? CT(0) : mv0;
                    final_vis(time, bl, chan, 1) = f1 ? CT(0) : mv1;
                    final_vis(time, bl, chan, 2) = f2 ? CT(0) : mv2;
                    final_vis(time, bl, chan, 3) = f3 ? CT(0) : mv3;

                    const CT & ov0 = observed_vis(time, bl, chan, 0);
                    const CT & ov1 = observed_vis(time, bl, chan, 1);
                    const CT & ov2 = observed_vis(time, bl, chan, 2);
                    const CT & ov3 = observed_vis(time, bl, chan, 3);

                    // Weights
                    const FT & w0 = weight(time, bl, chan, 0);
                    const FT & w1 = weight(time, bl, chan, 1);
                    const FT & w2 = weight(time, bl, chan, 2);
                    const FT & w3 = weight(time, bl, chan, 3);

                    // Compute chi squared
                    FT d0 = f0 ? FT(0) : chi_squared_term(mv0, ov0, w0);
                    FT d1 = f1 ? FT(0) : chi_squared_term(mv1, ov1, w1);
                    FT d2 = f2 ? FT(0) : chi_squared_term(mv2, ov2, w2);
                    FT d3 = f3 ? FT(0) : chi_squared_term(mv3, ov3, w3);

                    // Accumulate chi squared values
                    chi_squared_ = chi_squared_ + d0 + d1 + d2 + d3;
                }
            }
        }

        // Store reduction result
        chi_squared(0) = chi_squared_;
    }
};

MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_POST_PROCESS_VISIBILITIES_OP_CPU_H