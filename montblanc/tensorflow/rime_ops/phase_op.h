#ifndef RIME_PHASE_OP_H_
#define RIME_PHASE_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

// For M_PI
#define _USE_MATH_DEFINES
#include <cmath>

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Declare the fully templated RimePhaseOp class type up front
template <typename Device, typename FT, typename CT> class RimePhaseOp;

// Partially specialise RimePhaseOp for CPUDevice
template <typename FT, typename CT>
class RimePhaseOp<CPUDevice, FT, CT> : public tensorflow::OpKernel {
public:
    explicit RimePhaseOp(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_uvw = context->input(1);
        const tf::Tensor & in_frequency = context->input(2);

        OP_REQUIRES(context, in_lm.dims() == 2 && in_lm.dim_size(1) == 2,
            tf::errors::InvalidArgument(
                "lm should be of shape (nsrc, 2)"))

        OP_REQUIRES(context, in_uvw.dims() == 3 && in_uvw.dim_size(2) == 3,
            tf::errors::InvalidArgument(
                "uvw should be of shape (ntime, na, 3)"))

        OP_REQUIRES(context, in_frequency.dims() == 1,
            tf::errors::InvalidArgument(
                "frequency should be of shape (nchan)"))

        // Extract problem dimensions
        int nsrc = in_lm.dim_size(0);
        int ntime = in_uvw.dim_size(0);
        int na = in_uvw.dim_size(1);
        int nchan = in_frequency.dim_size(0);

        // Reason about our output shape
        tf::TensorShape complex_phase_shape({nsrc, ntime, na, nchan});

        // Create a pointer for the complex_phase result
        tf::Tensor * complex_phase_ptr = nullptr;
        tf::Tensor out_complex_phase;

        // Allocate memory for the complex_phase
        OP_REQUIRES_OK(context, context->allocate_output(
            0, complex_phase_shape, &complex_phase_ptr));

        if (complex_phase_ptr->NumElements() == 0)
            { return; }

        CHECK(out_complex_phase.CopyFrom(
            *complex_phase_ptr, complex_phase_shape));        

        // Access the underlying tensors, proper
        auto lm = in_lm.tensor<FT, 2>();
        auto uvw = in_uvw.tensor<FT, 3>();
        auto frequency = in_frequency.tensor<FT, 1>();
        auto complex_phase = out_complex_phase.tensor<CT, 4>();

        // Constant
        constexpr FT lightspeed = 299792458;

        // Compute the complex phase
        for(int src=0; src<nsrc; ++src)
        {
            FT l = lm(src,0);
            FT m = lm(src,1);
            FT n = std::sqrt(1.0 - l*l - m*m) - 1.0;

            for(int time=0; time<ntime; ++time)
            {
                for(int antenna=0; antenna<na; ++antenna)
                {

                    FT u = uvw(time,antenna,0);
                    FT v = uvw(time,antenna,1);
                    FT w = uvw(time,antenna,2);

                    FT real_phase_base = FT(-2*M_PI)*(l*u + m*v + n*w)/lightspeed;

                    for(int chan=0; chan<nchan; ++chan)
                    {
                        // Our real phase input to the exponential function is purely imaginary so we can
                        // can elide a call to std::exp<complex<FT>> and just compute the cos and sin
                        FT real_phase = real_phase_base*frequency(chan);
                        complex_phase(src,time,antenna,chan) = { std::cos(real_phase), std::sin(real_phase) };
                    }
                }
            }
        }

        /*
        // Doing it this way might give us SIMD's and threading automatically...
        // but compared to the above, it creates an expression tree and I don't
        // know how to evaluate the final result yet...
        auto l = lm.tensor<FT, 2>().slice(
            Eigen::DSizes<int,2>(0, 0),
            Eigen::DSizes<int,2>(nsrc, 1));

        auto m = lm.tensor<FT, 2>().slice(
            Eigen::DSizes<int,2>(0, 1),
            Eigen::DSizes<int,2>(nsrc, 2));


        // Create a tensor to hold a constant
        Eigen::Tensor<FT, 1> one(1);
        one[0] = 1.0;

        auto n = (one - l*l - m*m).sqrt() - one;


        auto u = uvw.tensor<FT, 3>().slice(
            Eigen::DSizes<int,3>(0,0,0),
            Eigen::DSizes<int,3>(ntime,na,1));

        auto v = uvw.tensor<FT, 3>().slice(
            Eigen::DSizes<int,3>(0,0,1),
            Eigen::DSizes<int,3>(ntime,na,2));

        auto w = uvw.tensor<FT, 3>().slice(
            Eigen::DSizes<int,3>(0,0,2),
            Eigen::DSizes<int,3>(ntime,na,3));

        auto phase = l*u + m*v + n*w;
        */
    }

};

} // namespace tensorflow {

#endif // #define RIME_PHASE_OP_H