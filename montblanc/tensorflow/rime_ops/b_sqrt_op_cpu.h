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

        OP_REQUIRES(context, in_ref_freq.dims() == 1 && in_ref_freq.dim_size(0) == 1,
            tf::errors::InvalidArgument(
                "ref_freq should be a scalar"))


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
        FT ref_freq = in_ref_freq.tensor<FT, 1>()(0);
        auto b_sqrt = b_sqrt_ptr->tensor<CT, 4>();

        enum { iI, iQ, iU, iV };
        enum { XX, XY, YX, YY };

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
                FT trace = 2.0*I;
                FT det = I*I - Q*Q - U*U - V*V;

                for(int chan=0; chan < nchan; ++chan)
                {
                    // Compute spectral index
                    FT power = std::pow(frequency(chan)/ref_freq,
                        alpha(src, time));

                    // Compute s and t, used to find matrix roots
                    FT s = std::sqrt(det);
                    FT t = std::sqrt(trace + 2.0*s);

                    // Set t to 1.0 to avoid nans/infs in the output
                    // t == 0.0 (and s == 0.0) imply a zero matrix
                    // in any case
                    if(t == 0.0)
                        { t = 1.0; }

                    // Create some common sub-expressions here
                    FT Is = I + s;
                    FT pst = std::sqrt(power)/t;
                    FT Utmp = U*pst;
                    FT Vtmp = V*pst;

                    // Assign square root of the brightness matrix
                    b_sqrt(src, time, chan, XX) = CT(pst*(Is + Q), 0    );
                    b_sqrt(src, time, chan, XY) = CT(Utmp        , Vtmp );
                    b_sqrt(src, time, chan, YX) = CT(Utmp        , -Vtmp);
                    b_sqrt(src, time, chan, YY) = CT(pst*(Is - Q), 0    );
                }
            }
        }
    }
};

} // namespace bsqrt {
} // namespace montblanc {

#endif // #ifndef RIME_B_SQRT_OP_CPU_H_