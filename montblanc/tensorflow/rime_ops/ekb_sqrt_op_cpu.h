#ifndef RIME_EKB_SQRT_OP_CPU_H
#define RIME_EKB_SQRT_OP_CPU_H

#include "ekb_sqrt_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_EKB_SQRT_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice; 

// Specialise the EKBSqrt op for CPUs
template <typename FT, typename CT>
class EKBSqrt<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit EKBSqrt(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_complex_phase = context->input(0);
        const tf::Tensor & in_bsqrt = context->input(1);
        const tf::Tensor & in_ejones = context->input(2);

        // Extract problem dimensions
        int nsrc = in_complex_phase.dim_size(0);
        int ntime = in_complex_phase.dim_size(1);
        int na = in_complex_phase.dim_size(2);
        int nchan = in_complex_phase.dim_size(3);
        int npol = in_bsqrt.dim_size(3);

        OP_REQUIRES(context, in_bsqrt.dims() == 4 &&
            in_bsqrt.dim_size(0) == nsrc &&
            in_bsqrt.dim_size(1) == ntime &&
            in_bsqrt.dim_size(2) == nchan &&
            in_bsqrt.dim_size(3) == npol,
            tf::errors::InvalidArgument(
                "bsqrt should be of shape (nsrc,ntime,nchan,npol)"))

        OP_REQUIRES(context, in_ejones.dims() == 5 &&
            in_ejones.dim_size(0) == nsrc &&
            in_ejones.dim_size(1) == ntime &&
            in_ejones.dim_size(2) == na &&
            in_ejones.dim_size(3) == nchan &&
            in_ejones.dim_size(4) == npol,
            tf::errors::InvalidArgument(
                "ejones should be of shape (nsrc,ntime,na,nchan,npol)"))

        tf::TensorShape ant_jones_shape({nsrc, ntime, na, nchan, npol});

        // Allocate an output tensor
        tf::Tensor * ant_jones_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, ant_jones_shape, &ant_jones_ptr));

        // Get pointers to flattened tensor data buffers
        auto complex_phase = in_complex_phase.tensor<CT, 4>();
        auto bsqrt = in_bsqrt.tensor<CT, 4>();
        auto ejones = in_ejones.tensor<CT, 5>();
        auto ant_jones = ant_jones_ptr->tensor<CT, 5>();

        for(int src=0; src < nsrc; ++src)
        {
            for(int time=0; time < ntime; ++time)
            {
                for(int ant=0; ant < na; ++ant)
                {
                    for(int chan=0; chan < nchan; ++chan)
                    {
                        // Reference the complex phase
                        const CT & cp = complex_phase(src, time, ant, chan);

                        // Multiply brightness square root by complex phase
                        const CT b0 = cp*bsqrt(src, time, chan, 0);
                        const CT b1 = cp*bsqrt(src, time, chan, 1);
                        const CT b2 = cp*bsqrt(src, time, chan, 2);
                        const CT b3 = cp*bsqrt(src, time, chan, 3);

                        // Reference ejones matrix
                        const CT & a0 = ejones(src, time, ant, chan, 0);
                        const CT & a1 = ejones(src, time, ant, chan, 1);
                        const CT & a2 = ejones(src, time, ant, chan, 2);
                        const CT & a3 = ejones(src, time, ant, chan, 3);

                        // Perform 2x2 jones multiply
                        ant_jones(src, time, ant, chan, 0) = a0*b0 + a1*b2;
                        ant_jones(src, time, ant, chan, 1) = a0*b1 + a1*b3;
                        ant_jones(src, time, ant, chan, 2) = a2*b0 + a3*b2;
                        ant_jones(src, time, ant, chan, 3) = a2*b1 + a3*b3;
                    }
                }
            }
        }
    }
};

MONTBLANC_EKB_SQRT_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_EKB_SQRT_OP_CPU_H
