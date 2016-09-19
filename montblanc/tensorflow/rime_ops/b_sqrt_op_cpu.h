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

        // Reason about our output shape
        tf::TensorShape b_sqrt_shape({nsrc, ntime, nchan, 4});

        // Create a pointer for the b_sqrt result
        tf::Tensor * b_sqrt_ptr = nullptr;

        // Allocate memory for the b_sqrt
        OP_REQUIRES_OK(context, context->allocate_output(
            0, b_sqrt_shape, &b_sqrt_ptr));

        if (b_sqrt_ptr->NumElements() == 0)
            { return; }

        auto stokes = in_stokes.tensor<FT, 3>();
        auto alpha = in_alpha.tensor<FT, 2>();
        auto frequency = in_frequency.tensor<FT, 1>();
        auto ref_freq = in_ref_freq.tensor<FT, 1>();
        auto b_sqrt = b_sqrt_ptr->tensor<CT, 4>();

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

                // Compute the trace and determinant of the brightness matrix
                // trace = I+Q + I-Q = 2I
                // det = (I+Q)*(I-Q) - (U+iV)*(U-iV) = I**2-Q**2-U**2-V**2
                // so we have real values in all cases
                CT trace = CT(two*I, 0.0);
                CT det = CT(I*I - Q*Q - U*U - V*V, 0.0);

                // Precompute matrix terms
                CT B0 = CT(I + Q, 0.0);
                CT B1 = CT(U    ,  V );
                CT B2 = CT(U    ,  -V);
                CT B3 = CT(I - Q, 0.0);

                // scalar matrix case
                if(det.real() == I*I)
                {
                    B0 = std::sqrt(B0);
                    B3 = std::sqrt(B3);
                }
                else
                {
                    // Complex square root of the determinant
                    CT s = std::sqrt(det);
                    CT t = std::sqrt(trace + two*s);

                    B0 += s;
                    B3 += s;

                    // Complex division
                    if(std::norm(t) > 0.0)
                    {
                        B0 /= t;
                        B1 /= t;
                        B2 /= t;
                        B3 /= t;
                    }
                }

                for(int chan=0; chan < nchan; ++chan)
                {
                    // Compute spectral index
                    FT power = std::pow(
                        frequency(chan)/ref_freq(chan),
                        alpha(src, time));

                    // Square root of spectral index
                    FT psqrt = std::sqrt(power);

                    // Assign square root of the brightness matrix
                    b_sqrt(src, time, chan, XX) = B0*psqrt;
                    b_sqrt(src, time, chan, XY) = B1*psqrt;
                    b_sqrt(src, time, chan, YX) = B2*psqrt;
                    b_sqrt(src, time, chan, YY) = B3*psqrt;
                }
            }
        }
    }
};

} // namespace bsqrt {
} // namespace montblanc {

#endif // #ifndef RIME_B_SQRT_OP_CPU_H_