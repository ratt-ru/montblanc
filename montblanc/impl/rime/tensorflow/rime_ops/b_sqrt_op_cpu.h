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
private:
    std::string polarisation_type;

public:
    explicit BSqrt(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("polarisation_type",
                                                 &polarisation_type));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_stokes = context->input(0);

        // Extract problem dimensions
        int nsrc = in_stokes.dim_size(0);
        int ntime = in_stokes.dim_size(1);
        int nchan = in_stokes.dim_size(2);

        // Reason about the shape of the b_sqrt tensor and
        // create a pointer to it
        tf::TensorShape b_sqrt_shape({nsrc, ntime, nchan, 4});
        tf::Tensor * b_sqrt_ptr = nullptr;

        // Allocate memory for the b_sqrt
        OP_REQUIRES_OK(context, context->allocate_output(
            0, b_sqrt_shape, &b_sqrt_ptr));

        if (b_sqrt_ptr->NumElements() == 0)
            { return; }

        // Reason about shape of the sgn_brightness tensor
        // and create a pointer to it
        tf::TensorShape sgn_brightness_shape({nsrc, ntime, nchan});
        tf::Tensor * sgn_brightness_ptr = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(
            1, sgn_brightness_shape, &sgn_brightness_ptr));

        auto stokes = in_stokes.tensor<FT, 4>();
        auto b_sqrt = b_sqrt_ptr->tensor<CT, 4>();
        auto sgn_brightness = sgn_brightness_ptr->tensor<tf::int8, 3>();

        // Linear polarisation or circular polarisation
        bool linear = (polarisation_type == "linear");
        unsigned int iI = 0;
        unsigned int iQ = linear ? 1 : 3;
        unsigned int iU = linear ? 2 : 1;
        unsigned int iV = linear ? 3 : 2;

        // Correlation indices
        enum { XX, XY, YX, YY };

        constexpr FT zero = 0.0;
        constexpr FT one = 1.0;

        #pragma omp parallel for collapse(2)
        for(int src=0; src < nsrc; ++src)
        {
            for(int time=0; time < ntime; ++time)
            {
                for(int chan=0; chan < nchan; ++chan)
                {
                    // Reference stokes parameters.
                    // Input order of stokes parameters differs
                    // depending on whether linear or circular polarisation
                    // is used, but the rest of the calculation is the same...
                    FT I = stokes(src, time, chan, iI);
                    FT Q = stokes(src, time, chan, iQ);
                    FT U = stokes(src, time, chan, iU);
                    FT V = stokes(src, time, chan, iV);

                    // sgn variable, used to indicate whether
                    // brightness matrix is negative, zero or positive
                    // and a valid Cholesky decomposition
                    FT IQ = I + Q;
                    FT sgn = (zero < IQ) - (IQ < zero);
                    // I *= sign;
                    // Q *= sign;
                    U *= sgn;
                    V *= sgn;
                    IQ *= sgn;

                    // Indicate negative, zero or positive brightness matrix
                    sgn_brightness(src, time, chan) = sgn;

                    // Compute cholesky decomposition
                    CT L00 = std::sqrt(CT(IQ, zero));
                    // Store L00 as a divisor of L10
                    CT div = L00;

                    // Gracefully handle zero matrices
                    if(IQ == zero)
                    {
                        div = CT(one, zero);
                        IQ = one;
                    }

                    CT L10 = CT(U, -V) / div;
                    FT L11_real = (I*I - Q*Q - U*U - V*V)/IQ;
                    CT L11 = std::sqrt(CT(L11_real, zero));

                    // Assign square root of the brightness matrix,
                    // computed via cholesky decomposition
                    b_sqrt(src, time, chan, XX) = L00;
                    b_sqrt(src, time, chan, XY) = 0.0;
                    b_sqrt(src, time, chan, YX) = L10;
                    b_sqrt(src, time, chan, YY) = L11;
                }
            }
        }
    }
};

} // namespace bsqrt {
} // namespace montblanc {

#endif // #ifndef RIME_B_SQRT_OP_CPU_H_