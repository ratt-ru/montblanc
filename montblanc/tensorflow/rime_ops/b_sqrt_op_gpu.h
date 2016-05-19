#ifndef RIME_B_SQRT_OP_GPU_H_
#define RIME_B_SQRT_OP_GPU_H_

#if GOOGLE_CUDA

#include "b_sqrt_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;    

// Traits class defined by float and complex types
template <typename FT>
class LaunchTraits;

// Specialise for float
template <>
class LaunchTraits<float>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 2;

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, na, ntime);
    }        
};

// Specialise for double
template <>
class LaunchTraits<double>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, na, ntime);
    }        
};

template <typename FT, typename CT>
class RimeBSqrt<GPUDevice, FT, CT> : public tensorflow::OpKernel
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
    }
};

} // namespace tensorflow {

#endif

#endif // #ifndef RIME_B_SQRT_OP_GPU_H_