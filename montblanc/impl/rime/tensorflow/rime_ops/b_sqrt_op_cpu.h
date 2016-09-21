#ifndef RIME_B_SQRT_OP_CPU_H_
#define RIME_B_SQRT_OP_CPU_H_

#include "b_sqrt_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace montblanc {
namespace bsqrt {

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename FT, typename CT>
class BSqrt<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit BSqrt(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_stokes = context->input(0);
        const tf::Tensor & in_alpha = context->input(1);
        const tf::Tensor & in_frequency = context->input(2);
        const tf::Tensor & in_ref_freq = context->input(3);

        OP_REQUIRES(context, in_stokes.dims() == 3 && in_stokes.dim_size(2) == 4,
            tf::errors::InvalidArgument(
                "stokes should be of shape (nsrc, ntime, 4)"))

        OP_REQUIRES(context, in_alpha.dims() == 2,
            tf::errors::InvalidArgument(
                "alpha should be of shape (nsrc, ntime)"))

        OP_REQUIRES(context, in_frequency.dims() == 1,
            tf::errors::InvalidArgument(
                "frequency should be of shape (nchan)"))

        OP_REQUIRES(context, in_ref_freq.dims() == 1,
            tf::errors::InvalidArgument(
                "ref_frequency should be of shape (nchan)"))

        // Extract problem dimensions
        int nsrc = in_stokes.dim_size(0);
        int ntime = in_stokes.dim_size(1);
        int nchan = in_frequency.dim_size(0);

        // Reason about the shape of the b_sqrt tensor and
        // create a pointer to it
        tf::TensorShape b_sqrt_shape({nsrc, ntime, nchan, 4});
        tf::Tensor * b_sqrt_ptr = nullptr;

        // Allocate memory for the b_sqrt
        OP_REQUIRES_OK(context, context->allocate_output(
            0, b_sqrt_shape, &b_sqrt_ptr));

        if (b_sqrt_ptr->NumElements() == 0)
            { return; }

        // Reason about shape of the invert tensor
        // and create a pointer to it
        tf::TensorShape invert_shape({nsrc, ntime});
        tf::Tensor * invert_ptr = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(
            1, invert_shape, &invert_ptr));

        auto stokes = in_stokes.tensor<FT, 3>();
        auto alpha = in_alpha.tensor<FT, 2>();
        auto frequency = in_frequency.tensor<FT, 1>();
        auto ref_freq = in_ref_freq.tensor<FT, 1>();
        auto b_sqrt = b_sqrt_ptr->tensor<CT, 4>();
        auto neg_ant_jones = invert_ptr->tensor<tf::int8, 2>();

        enum { iI, iQ, iU, iV };
        enum { XX, XY, YX, YY };

        constexpr FT two = 2.0;
        constexpr FT half = 0.5;

        for(int src=0; src < nsrc; ++src)
        {
            for(int time=0; time < ntime; ++time)
            {
                // Reference stokes parameters
                const FT & I = stokes(src, time, iI);
                const FT & Q = stokes(src, time, iQ);
                const FT & U = stokes(src, time, iU);
                const FT & V = stokes(src, time, iV);

                // Sign variable, used to attempt to ensure
                // positive definiteness of the brightness matrix
                // and a valid Cholesky decomposition
                FT sign = 1.0;

                if(I + Q < 0)
                    { sign = -1.0; }

                // Compute cholesky decomposition
                CT L00 = std::sqrt(sign*CT(I+Q));
                CT L10 = sign*CT(U, -V) / L00;
                CT L11 = std::sqrt(CT(sign*(I*I - Q*Q - U*U - V*V)/(I+Q), 0.0));

                // Indicate that we inverted the sign of the brightness
                // matrix to obtain the cholesky decomposition
                neg_ant_jones(src, time) = (sign == 1.0 ? 1 : -1);

                for(int chan=0; chan < nchan; ++chan)
                {
                    // Compute square root of spectral index
                    FT psqrt = std::pow(
                        frequency(chan)/ref_freq(chan),
                        alpha(src, time)*0.5);

                    // Assign square root of the brightness matrix
                    b_sqrt(src, time, chan, XX) = L00*psqrt;
                    b_sqrt(src, time, chan, XY) = 0.0;
                    b_sqrt(src, time, chan, YX) = L10*psqrt;
                    b_sqrt(src, time, chan, YY) = L11*psqrt;
                }
            }
        }
    }
};

} // namespace bsqrt {
} // namespace montblanc {

#endif // #ifndef RIME_B_SQRT_OP_CPU_H_