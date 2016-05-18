#ifndef RIME_B_SQRT_OP_CPU_H_
#define RIME_B_SQRT_OP_CPU_H_

#include "b_sqrt_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;    

template <typename FT, typename CT>
class RimeBSqrt<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit RimeBSqrt(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

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

        for(int src=0; src < nsrc; ++src)
        {
            for(int time=0; time < ntime; ++time)
            {
                // Reference stokes parameters
                const FT & I = stokes(src, time, iI);
                const FT & Q = stokes(src, time, iQ);
                const FT & U = stokes(src, time, iU);
                const FT & V = stokes(src, time, iV);

                // Compute the brightness matrix
                CT _XX = CT(I + Q, 0.0);
                CT _XY = CT(U    ,   V);
                CT _YX = CT(U    ,  -V);
                CT _YY = CT(I - Q, 0.0);

                // Compute the trace and determinant of the brightness matrix
                // trace = I+Q + I-Q = 2I
                // det = (I+Q)*(I-Q) - (U+iV)*(U-iV) = I**2-Q**2-U**2-V**2
                // so we have real values in all cases
                FT _trace = 2.0*I;
                FT _det = I*I - Q*Q - U*U - V*V;

                for(int chan=0; chan < nchan; ++chan)
                {
                    // Compute spectral index
                    FT power = std::pow(frequency(chan)/ref_freq,
                        alpha(src, time));

                    // Multiply spi into the brightness matrix 
                    CT XX = _XX*power;
                    CT XY = _XY*power;
                    CT YX = _YX*power;
                    CT YY = _YY*power;

                    // Multiply spi into the trace and det. Need
                    // power*power for det because its composed of squares
                    FT trace = _trace*power;
                    FT det = _det*power*power;

                    // Compute s and t, used to find matrix roots
                    FT s = std::sqrt(det);
                    FT t = std::sqrt(trace + 2.0*s);

                    // We only have roots for matrices when
                    // both s and t are non-zero. If this is
                    // not the case, this implies the matrix
                    // entries are zero.
                    XX += s; YY += s;

                    if(t != 0.0)
                    {
                        // Scale the matrix
                        XX /= t; XY /= t;
                        YX /= t; YY /= t;
                    }

                    // Assign values
                    b_sqrt(src, time, chan, 0) = XX;
                    b_sqrt(src, time, chan, 1) = XY;
                    b_sqrt(src, time, chan, 2) = YX;
                    b_sqrt(src, time, chan, 3) = YY;
                }
            }
        }
    }
};

} // namespace tensorflow {

#endif // #ifndef RIME_B_SQRT_OP_CPU_H_