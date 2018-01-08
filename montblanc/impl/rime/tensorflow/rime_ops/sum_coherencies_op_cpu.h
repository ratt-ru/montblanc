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

        const tf::Tensor & in_time_index = context->input(0);
        const tf::Tensor & in_antenna1 = context->input(1);
        const tf::Tensor & in_antenna2 = context->input(2);
        const tf::Tensor & in_shape = context->input(3);
        const tf::Tensor & in_ant_jones = context->input(4);
        const tf::Tensor & in_sgn_brightness = context->input(5);
        const tf::Tensor & in_base_coherencies = context->input(6);

        int nvrow = in_time_index.dim_size(0);
        int nsrc = in_shape.dim_size(0);
        int nchan = in_shape.dim_size(2);
        int na = in_ant_jones.dim_size(2);
        int npol = in_ant_jones.dim_size(4);
        int npolchan = nchan*npol;

        // Allocate an output tensor
        tf::Tensor * coherencies_ptr = nullptr;
        tf::TensorShape coherencies_shape = tf::TensorShape({
            nvrow, nchan, npol });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, coherencies_shape, &coherencies_ptr));

        auto time_index = in_time_index.tensor<int,1>();
        auto antenna1 = in_antenna1.tensor<int,1>();
        auto antenna2 = in_antenna2.tensor<int,1>();
        auto shape = in_shape.tensor<FT, 3>();
        auto ant_jones = in_ant_jones.tensor<CT, 5>();
        auto sgn_brightness = in_sgn_brightness.tensor<tf::int8, 2>();
        auto base_coherencies = in_base_coherencies.tensor<CT, 3>();
        auto coherencies = coherencies_ptr->tensor<CT, 3>();

        #pragma omp parallel for
        for(int vrow=0; vrow<nvrow; ++vrow)
        {
            // Antenna pairs for this baseline
            int ant1 = antenna1(vrow);
            int ant2 = antenna2(vrow);
            int time = time_index(vrow);

            for(int chan=0; chan<nchan; ++chan)
            {
                // Load in the input model visibilities
                CT s0 = base_coherencies(vrow, chan, 0);
                CT s1 = base_coherencies(vrow, chan, 1);
                CT s2 = base_coherencies(vrow, chan, 2);
                CT s3 = base_coherencies(vrow, chan, 3);

                for(int src=0; src<nsrc; ++src)
                {
                    // Reference antenna 1 jones
                    const CT & a0 = ant_jones(src, time, ant1, chan, 0);
                    const CT & a1 = ant_jones(src, time, ant1, chan, 1);
                    const CT & a2 = ant_jones(src, time, ant1, chan, 2);
                    const CT & a3 = ant_jones(src, time, ant1, chan, 3);

                    // Multiply shape value into antenna1 jones
                    const FT & s = shape(src, vrow, chan);

                    // Conjugate transpose of antenna 2 jones with shape factor
                    CT b0 = std::conj(ant_jones(src, time, ant2, chan, 0)*s);
                    CT b1 = std::conj(ant_jones(src, time, ant2, chan, 2)*s);
                    CT b2 = std::conj(ant_jones(src, time, ant2, chan, 1)*s);
                    CT b3 = std::conj(ant_jones(src, time, ant2, chan, 3)*s);

                    FT sign = sgn_brightness(src, time);

                    // Multiply jones matrices and accumulate them
                    // in the sum terms
                    s0 += sign*(a0*b0 + a1*b2);
                    s1 += sign*(a0*b1 + a1*b3);
                    s2 += sign*(a2*b0 + a3*b2);
                    s3 += sign*(a2*b1 + a3*b3);
                }

                // Output accumulated model visibilities
                coherencies(vrow, chan, 0) = s0;
                coherencies(vrow, chan, 1) = s1;
                coherencies(vrow, chan, 2) = s2;
                coherencies(vrow, chan, 3) = s3;
            }
        }
    }
};

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SUM_COHERENCIES_OP_CPU_H
