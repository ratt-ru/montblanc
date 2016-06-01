#ifndef RIME_SUM_COHERENCIES_OP_CPU_H
#define RIME_SUM_COHERENCIES_OP_CPU_H

#include "sum_coherencies_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace montblanc {
namespace sumcoherencies {

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;    

template <typename FT, typename CT>
class RimeSumCoherencies<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit RimeSumCoherencies(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_uvw = context->input(0);
        const tf::Tensor & in_gauss_shape = context->input(1);
        const tf::Tensor & in_sersic_shape = context->input(2);
        const tf::Tensor & in_frequency = context->input(3);
        const tf::Tensor & in_antenna1 = context->input(4);
        const tf::Tensor & in_antenna2 = context->input(5);
        const tf::Tensor & in_ant_jones = context->input(6);
        const tf::Tensor & in_flag = context->input(7);
        const tf::Tensor & in_weight = context->input(8);
        const tf::Tensor & in_gterm = context->input(9);
        const tf::Tensor & in_obs_vis = context->input(10);

        OP_REQUIRES(context, in_uvw.dims() == 3 && in_uvw.dim_size(2) == 3,
            tf::errors::InvalidArgument(
                "uvw should be of shape (ntime, na, 3)"))

        OP_REQUIRES(context, in_obs_vis.dims() == 4 && in_obs_vis.dim_size(3) == 4,
            tf::errors::InvalidArgument(
                "obs_vis should be of shape (ntime, nbl, nchan, 4"))

        int ntime = in_obs_vis.dim_size(0);
        int nbl = in_obs_vis.dim_size(1);
        int nchan = in_obs_vis.dim_size(2);
        int npol = in_obs_vis.dim_size(3);
        int npolchan = nchan*npol;
        
        int ngsrc = in_gauss_shape.dim_size(0);
        int nssrc = in_sersic_shape.dim_size(0);
        int nsrc = in_ant_jones.dim_size(0);
        int na = in_ant_jones.dim_size(2);
        int npsrc = nsrc - ngsrc - nssrc;

        // Allocate an output tensor
        tf::TensorShape model_vis_shape({ntime, nbl, nchan, npol});
        tf::Tensor * model_vis_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, model_vis_shape, &model_vis_ptr));
        
        if(model_vis_ptr->NumElements() == 0)
            { return; }

        /*
        for(int time=0; time<ntime; ++time)
        {
            for(int bl=0; bl<nbl; ++bl)
            {
                int ant1 = antenna1(time, bl);
                int ant2 = antenna2(time, bl);
                FT u = uvw(time, ant2, 0) - uvw(time, ant1, 0);
                FT v = uvw(time, ant2, 1) - uvw(time, ant1, 1);
                FT w = uvw(time, ant2, 2) - uvw(time, ant1, 2);

                for(int ch=0; ch<nchan; ++ch)
                {
                    for(int pol=0; pol < npol; ++pol)
                    {
                        CT polsum = {0, 0};

                        if(cdata.nsrc.lower_extent > 0)
                        {
                            polsum = *model_vis_ptr(time, bl, ch);
                        }
                    }
                }
            }
        }
        */
    }
};

} // namespace sumcoherencies {
} // namespace montblanc {

#endif // #define RIME_SUM_COHERENCIES_OP_CPU_H