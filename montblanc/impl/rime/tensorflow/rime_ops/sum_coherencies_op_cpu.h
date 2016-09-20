#ifndef RIME_SUM_COHERENCIES_OP_CPU_H
#define RIME_SUM_COHERENCIES_OP_CPU_H

#include "sum_coherencies_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SUM_COHERENCIES_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the SumCoherencies op for CPUs
template <typename FT, typename CT>
class SumCoherencies<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit SumCoherencies(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_antenna1 = context->input(0);
        const tf::Tensor & in_antenna2 = context->input(1);
        const tf::Tensor & in_shape = context->input(2);
        const tf::Tensor & in_ant_jones = context->input(3);
        const tf::Tensor & in_neg_ant_jones = context->input(4);
        const tf::Tensor & in_flag = context->input(5);
        const tf::Tensor & in_gterm = context->input(6);
        const tf::Tensor & in_model_vis_in = context->input(7);
        const tf::Tensor & in_apply_dies = context->input(8);

        int nsrc = in_shape.dim_size(0);
        int ntime = in_shape.dim_size(1);
        int nbl = in_shape.dim_size(2);
        int nchan = in_shape.dim_size(3);
        int na = in_ant_jones.dim_size(2);
        int npol = in_ant_jones.dim_size(4);
        int npolchan = nchan*npol;

        // Allocate an output tensor
        tf::Tensor * model_vis_out_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, in_model_vis_in.shape(), &model_vis_out_ptr));

        auto antenna1 = in_antenna1.tensor<int,2>();
        auto antenna2 = in_antenna2.tensor<int,2>();
        auto shape = in_shape.tensor<FT, 4>();
        auto ant_jones = in_ant_jones.tensor<CT, 5>();
        auto neg_ant_jones = in_neg_ant_jones.tensor<tf::int8, 2>();
        auto flag = in_flag.tensor<tf::uint8, 4>();
        auto gterm = in_gterm.tensor<CT,4>();
        auto model_vis_in = in_model_vis_in.tensor<CT, 4>();
        auto model_vis_out = model_vis_out_ptr->tensor<CT, 4>();
        auto apply_dies = in_apply_dies.tensor<bool, 0>()(0);

        for(int time=0; time<ntime; ++time)
        {
            for(int bl=0; bl<nbl; ++bl)
            {
                // Antenna pairs for this baseline
                int ant1 = antenna1(time, bl);
                int ant2 = antenna2(time, bl);

                for(int chan=0; chan<nchan; ++chan)
                {
                    // Load in the input model visibilities
                    CT s0 = model_vis_in(time, bl, chan, 0);
                    CT s1 = model_vis_in(time, bl, chan, 1);
                    CT s2 = model_vis_in(time, bl, chan, 2);
                    CT s3 = model_vis_in(time, bl, chan, 3);

                    for(int src=0; src<nsrc; ++src)
                    {
                        // Reference antenna 1 jones
                        const CT & a0 = ant_jones(src, time, ant1, chan, 0);
                        const CT & a1 = ant_jones(src, time, ant1, chan, 1);
                        const CT & a2 = ant_jones(src, time, ant1, chan, 2);
                        const CT & a3 = ant_jones(src, time, ant1, chan, 3);

                        // Multiply shape value into antenna1 jones
                        const FT & s = shape(src, time, bl, chan);

                        // Conjugate transpose of antenna 2 jones with shape factor
                        CT b0 = std::conj(ant_jones(src, time, ant2, chan, 0)*s);
                        CT b1 = std::conj(ant_jones(src, time, ant2, chan, 2)*s);
                        CT b2 = std::conj(ant_jones(src, time, ant2, chan, 1)*s);
                        CT b3 = std::conj(ant_jones(src, time, ant2, chan, 3)*s);

                        FT sign = neg_ant_jones(src, time);

                        // Multiply jones matrices and accumulate them
                        // in the sum terms
                        s0 += sign*(a0*b0 + a1*b2);
                        s1 += sign*(a0*b1 + a1*b3);
                        s2 += sign*(a2*b0 + a3*b2);
                        s3 += sign*(a2*b1 + a3*b3);
                    }

                    // Apply Direction Independent Effects if required
                    if(apply_dies)
                    {
                        // Reference antenna 1 g terms
                        const CT & a0 = gterm(time,ant1,chan,0);
                        const CT & a1 = gterm(time,ant1,chan,1);
                        const CT & a2 = gterm(time,ant1,chan,2);
                        const CT & a3 = gterm(time,ant1,chan,3);

                        // Multiply model visibilities by antenna 1 g
                        CT r0 = a0*s0 + a1*s2;
                        CT r1 = a0*s1 + a1*s3;
                        CT r2 = a2*s0 + a3*s2;
                        CT r3 = a2*s1 + a3*s3;

                        // Conjugate transpose of antenna 2 g term
                        CT b0 = std::conj(gterm(time,ant2,chan,0));
                        CT b1 = std::conj(gterm(time,ant2,chan,2));
                        CT b2 = std::conj(gterm(time,ant2,chan,1));
                        CT b3 = std::conj(gterm(time,ant2,chan,3));

                        // Multiply
                        s0 = r0*b0 + r1*b2;
                        s1 = r0*b1 + r1*b3;
                        s2 = r2*b0 + r3*b2;
                        s3 = r2*b1 + r3*b3;
                    }

                    // If flags apply, zero out the polarisation
                    if(flag(time, bl, chan, 0)) { s0 = {0.0, 0.0}; }
                    if(flag(time, bl, chan, 1)) { s1 = {0.0, 0.0}; }
                    if(flag(time, bl, chan, 2)) { s2 = {0.0, 0.0}; }
                    if(flag(time, bl, chan, 3)) { s3 = {0.0, 0.0}; }

                    // Output accumulated model visibilities
                    model_vis_out(time, bl, chan, 0) = s0;
                    model_vis_out(time, bl, chan, 1) = s1;
                    model_vis_out(time, bl, chan, 2) = s2;
                    model_vis_out(time, bl, chan, 3) = s3;
                }
            }
        }
    }
};

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SUM_COHERENCIES_OP_CPU_H
