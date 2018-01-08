#ifndef RIME_GAUSS_SHAPE_OP_CPU_H
#define RIME_GAUSS_SHAPE_OP_CPU_H

#include "gauss_shape_op.h"
#include "constants.h"


// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_GAUSS_SHAPE_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the GaussShape op for CPUs
template <typename FT>
class GaussShape<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit GaussShape(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context){}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_time_index = context->input(0);
        const tf::Tensor & in_uvw = context->input(1);
        const tf::Tensor & in_antenna1 = context->input(2);
        const tf::Tensor & in_antenna2 = context->input(3);
        const tf::Tensor & in_frequency = context->input(4);
        const tf::Tensor & in_gauss_params = context->input(5);

        int nvrow = in_antenna1.dim_size(0);
        int nchan = in_frequency.dim_size(0);
        int ngsrc = in_gauss_params.dim_size(1);

        tf::TensorShape gauss_shape_shape{ngsrc,nvrow,nchan};

        // Allocate an output tensor
        tf::Tensor * gauss_shape_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, gauss_shape_shape, &gauss_shape_ptr));

        auto time_index = in_time_index.tensor<int, 1>();
        auto uvw = in_uvw.tensor<FT, 3>();
        auto antenna1 = in_antenna1.tensor<int, 1>();
        auto antenna2 = in_antenna2.tensor<int, 1>();
        auto frequency = in_frequency.tensor<FT, 1>();
        auto gauss_params = in_gauss_params.tensor<FT, 2>();
        auto gauss_shape = gauss_shape_ptr->tensor<FT, 3>();

        #pragma omp parallel
        for(int gsrc=0; gsrc < ngsrc; ++gsrc)
        {
            auto el = gauss_params(0,gsrc);
            auto em = gauss_params(1,gsrc);
            auto eR = gauss_params(2,gsrc);

            #pragma omp parallel for
            for(int vrow=0; vrow < nvrow; ++vrow)
            {
                // Antenna pairs for this baseline
                int ant1 = antenna1(vrow);
                int ant2 = antenna2(vrow);
                int time = time_index(vrow);

                // UVW coordinates for this baseline
                FT u = uvw(time,ant2,0) - uvw(time,ant1,0);
                FT v = uvw(time,ant2,1) - uvw(time,ant1,1);

                for(int chan=0; chan < nchan; ++chan)
                {
                    FT scaled_freq = montblanc::constants<FT>::gauss_scale*frequency(chan);

                    FT u1 = u*em - v*el;
                    u1 *= scaled_freq*eR;

                    FT v1 = u*el + v*em;
                    v1 *= scaled_freq;

                    gauss_shape(gsrc,vrow,chan) = std::exp(-(u1*u1 + v1*v1));
                }
            }
        }
    }
};

MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_GAUSS_SHAPE_OP_CPU_H
