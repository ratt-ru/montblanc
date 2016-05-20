#ifndef RIME_PHASE_OP_CPU_H_
#define RIME_PHASE_OP_CPU_H_

#include "phase_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#define RIME_PHASE_LOOP_STRATEGY 0
#define RIME_PHASE_EIGEN_STRATEGY 1
#define RIME_PHASE_CPU_STRATEGY RIME_PHASE_LOOP_STRATEGY

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

// For M_PI
#define _USE_MATH_DEFINES
#include <cmath>

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct make_complex_functor
{
    typedef std::complex<T> result_type;
    
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type
    operator()(T real, T imag) const
        { return std::complex<T>(real, imag); }
};

// Partially specialise RimePhaseOp for CPUDevice
template <typename FT, typename CT>
class RimePhaseOp<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
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

        // Allocate memory for the complex_phase
        OP_REQUIRES_OK(context, context->allocate_output(
            0, complex_phase_shape, &complex_phase_ptr));

        if (complex_phase_ptr->NumElements() == 0)
            { return; }

        // Access the underlying tensors, proper
        auto lm = in_lm.tensor<FT, 2>();
        auto uvw = in_uvw.tensor<FT, 3>();
        auto frequency = in_frequency.tensor<FT, 1>();
        auto complex_phase = complex_phase_ptr->tensor<CT, 4>();

        // Constant
        constexpr FT lightspeed = 299792458.0;

#if RIME_PHASE_CPU_STRATEGY == RIME_PHASE_LOOP_STRATEGY
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

#elif RIME_PHASE_CPU_STRATEGY == RIME_PHASE_EIGEN_STRATEGY
        // Doing it this way might give us SIMD's and threading automatically...
        // but compared to the above, it creates an expression tree and I don't
        // know how to evaluate the final result yet...

        const CPUDevice & device = context->eigen_device<CPUDevice>();

#if !defined(EIGEN_HAS_INDEX_LIST)
        Eigen::DSizes<int, 4> lm_shape( nsrc, 1,     1,  1);
        Eigen::DSizes<int, 4> uvw_shape( 1,    ntime, na, 1);
        Eigen::DSizes<int, 4> freq_shape(1,    1,     1,  nsrc);
#else
        Eigen::IndexList<int,
            Eigen::type2index<1>,
            Eigen::type2index<1>,
            Eigen::type2index<1> > lm_shape;
        lm_shape.set(0, nsrc);

        Eigen::IndexList<Eigen::type2index<1>,
            int,
            int,
            Eigen::type2index<1> > uvw_shape;
        uvw_shape.set(1, ntime);
        uvw_shape.set(2, na);

        Eigen::IndexList<Eigen::type2index<1>,
            Eigen::type2index<1>,
            Eigen::type2index<1>,
            int > freq_shape;
        freq_shape.set(3, nchan);
#endif

        auto l = lm.slice(
                Eigen::DSizes<int, 2>(0,    0),
                Eigen::DSizes<int, 2>(nsrc, 1))
            .reshape(lm_shape);

        auto m = lm.slice(
                Eigen::DSizes<int, 2>(0,    1),
                Eigen::DSizes<int, 2>(nsrc, 2))
            .reshape(lm_shape);


        // Create a tensor to hold one as a constant
        Eigen::Tensor<FT, 1> one(1);
        one[0] = 1.0;

        // Create a tensor to hold -2*pi/C
        Eigen::Tensor<FT, 1> minus_two_pi_over_c(1);
        minus_two_pi_over_c[0] = FT(-2*M_PI/lightspeed);

        auto n = (one - l*l - m*m).sqrt() - one;

        auto u = uvw.slice(
                Eigen::DSizes<int, 3>(0,     0,  0),
                Eigen::DSizes<int, 3>(ntime, na, 1))
            .reshape(uvw_shape);

        auto v = uvw.slice(
                Eigen::DSizes<int, 3>(0,     0,  1),
                Eigen::DSizes<int, 3>(ntime, na, 2))
            .reshape(uvw_shape);

        auto w = uvw.slice(
                Eigen::DSizes<int, 3>(0,     0,  2),
                Eigen::DSizes<int, 3>(ntime, na, 3))
            .reshape(uvw_shape);

        auto f = frequency.reshape(freq_shape);

        auto phase = minus_two_pi_over_c*(l*u + m*v + n*w)*f/lightspeed;
        // eigen is missing sin and cos members on TensorBase
        // call them with unaryExpr
        auto sinp = phase.unaryExpr(Eigen::internal::scalar_sin_op<FT>());
        auto cosp = phase.unaryExpr(Eigen::internal::scalar_cos_op<FT>());

        complex_phase.device(device) = sinp.binaryExpr(
            cosp, make_complex_functor<FT>());
#endif
    }
};

} // namespace tensorflow {

#endif // #define RIME_PHASE_OP_H