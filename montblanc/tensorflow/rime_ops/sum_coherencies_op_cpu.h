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

        tf::TensorShape model_vis_shape({ntime, nbl, nchan, npol});
        tf::Tensor * model_vis_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, model_vis_shape, &model_vis_ptr));
    }
};

} // namespace sumcoherencies {
} // namespace montblanc {

#endif // #define RIME_SUM_COHERENCIES_OP_CPU_H