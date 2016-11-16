#ifndef RIME_SERSIC_SHAPE_OP_CPU_H
#define RIME_SERSIC_SHAPE_OP_CPU_H

#include "sersic_shape_op.h"
#include "constants.h"


// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SERSIC_SHAPE_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the SersicShape op for CPUs
template <typename FT>
class SersicShape<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit SersicShape(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context){}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_uvw = context->input(0);
        const tf::Tensor & in_antenna1 = context->input(1);
        const tf::Tensor & in_antenna2 = context->input(2);
        const tf::Tensor & in_frequency = context->input(3);
        const tf::Tensor & in_sersic_params = context->input(4);

        int ntime = in_uvw.dim_size(0);
        int na = in_uvw.dim_size(1);
        int nbl = in_antenna1.dim_size(1);
        int nchan = in_frequency.dim_size(0);
        int nssrc = in_sersic_params.dim_size(1);

        tf::TensorShape sersic_shape_shape{nssrc,ntime,nbl,nchan};

        // Allocate an output tensor
        tf::Tensor * sersic_shape_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, sersic_shape_shape, &sersic_shape_ptr));

        auto uvw = in_uvw.tensor<FT, 3>();
        auto antenna1 = in_antenna1.tensor<int, 2>();
        auto antenna2 = in_antenna2.tensor<int, 2>();
        auto frequency = in_frequency.tensor<FT, 1>();
        auto sersic_params = in_sersic_params.tensor<FT, 2>();
        auto sersic_shape = sersic_shape_ptr->tensor<FT, 4>();

        constexpr FT one = FT(1.0);

        #pragma omp parallel
        for(int ssrc=0; ssrc < nssrc; ++ssrc)
        {
            auto e1 = sersic_params(0,ssrc);
            auto e2 = sersic_params(1,ssrc);
            auto ss = sersic_params(2,ssrc);

            #pragma omp for collapse(2)
            for(int time=0; time < ntime; ++time)
            {
                for(int bl=0; bl < nbl; ++bl)
                {
                    // Antenna pairs for this baseline
                    int ant1 = antenna1(time,bl);
                    int ant2 = antenna2(time,bl);

                    // UVW coordinates for this baseline
                    FT u = uvw(time,ant2,0) - uvw(time,ant1,0);
                    FT v = uvw(time,ant2,1) - uvw(time,ant1,1);

                    for(int chan=0; chan < nchan; ++chan)
                    {
                        FT scaled_freq = montblanc::constants<FT>::two_pi_over_c*frequency(chan);

                        // sersic source in  the Fourier domain
                        FT u1 = u*(one + e1) + v*e2;
                        u1 *= scaled_freq;
                        u1 *= ss/(one - e1*e1 - e2*e2);

                        FT v1 = u*e2 + v*(one - e1);
                        v1 *= scaled_freq;
                        v1 *= ss/(one - e1*e1 - e2*e2);

                        FT sersic_factor = one + u1*u1+v1*v1;

                        sersic_shape(ssrc,time,bl,chan) = one / (ss*std::sqrt(sersic_factor));
                    }
                }
            }
        }
    }
};

MONTBLANC_SERSIC_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SERSIC_SHAPE_OP_CPU_H
