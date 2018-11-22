#if GOOGLE_CUDA

#ifndef RIME_GAUSS_SHAPE_OP_GPU_CUH
#define RIME_GAUSS_SHAPE_OP_GPU_CUH

#include "gauss_shape_op.h"
#include <montblanc/abstraction.cuh>
#include "constants.h"

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_GAUSS_SHAPE_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// Traits class defined by float types
template <typename FT> class LaunchTraits;

// Specialise for float
template <> struct LaunchTraits<float>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};

// Specialise for double
template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};

// CUDA kernel outline
template <typename Traits>
__global__ void rime_gauss_shape(
    const typename Traits::uvw_type * uvw,
    const typename Traits::frequency_type * frequency,
    const typename Traits::gauss_param_type * gauss_params,
    typename Traits::gauss_shape_type * gauss_shape,
    const typename Traits::FT gauss_scale,
    int ngsrc, int nvrow, int nchan)
{
    int chan = blockIdx.x*blockDim.x + threadIdx.x;
    int vrow = blockIdx.y*blockDim.y + threadIdx.y;

    using FT = typename Traits::FT;
    using LTr = LaunchTraits<FT>;
    using Po = montblanc::kernel_policies<FT>;


    __shared__ struct {
        typename Traits::uvw_type uvw[LTr::BLOCKDIMY];
        typename Traits::frequency_type scaled_freq[LTr::BLOCKDIMX];
    } shared;

    int i;

    if(vrow >= nvrow || chan >= nchan)
        { return; }

    // Wavelength varies by channel, but not baseline
    if(threadIdx.y == 0)
        { shared.scaled_freq[threadIdx.x] = gauss_scale*frequency[chan]; }

    // UVW coordinates vary by baseline, but not channel
    if(threadIdx.x == 0)
        { shared.uvw[threadIdx.y] = uvw[vrow]; }

    // Reference u, v and w in shared memory for this thread
    FT & u = shared.uvw[threadIdx.y].x;
    FT & v = shared.uvw[threadIdx.y].y;

    __syncthreads();

    for(int gsrc=0; gsrc < ngsrc; ++gsrc)
    {
        i = (gsrc*nvrow + vrow)*nchan + chan;

        i = gsrc;   FT el = cub::ThreadLoad<cub::LOAD_LDG>(gauss_params+i);
        i += ngsrc; FT em = cub::ThreadLoad<cub::LOAD_LDG>(gauss_params+i);
        i += ngsrc; FT eR = cub::ThreadLoad<cub::LOAD_LDG>(gauss_params+i);

        FT u1 = u*em - v*el;
        u1 *= shared.scaled_freq[threadIdx.x]*eR;

        FT v1 = u*el + v*em;
        v1 *= shared.scaled_freq[threadIdx.x];

        i = (gsrc*nvrow + vrow)*nchan + chan;
        gauss_shape[i] = Po::exp(-(u1*u1 + v1*v1));
    }
}

// Specialise the GaussShape op for GPUs
template <typename FT>
class GaussShape<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit GaussShape(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_uvw = context->input(0);
        const tf::Tensor & in_frequency = context->input(1);
        const tf::Tensor & in_gauss_params = context->input(2);

        int nvrow = in_uvw.dim_size(0);
        int nchan = in_frequency.dim_size(0);
        int ngsrc = in_gauss_params.dim_size(1);

        tf::TensorShape gauss_shape_shape{ngsrc, nvrow, nchan};

        // Allocate an output tensor
        tf::Tensor * gauss_shape_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, gauss_shape_shape, &gauss_shape_ptr));

        using LTr = LaunchTraits<FT>;
        using Tr = montblanc::kernel_traits<FT>;

        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            nchan, nvrow, 1);
        dim3 grid(montblanc::grid_from_thread_block(block,
            nchan, nvrow, 1));

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        auto uvw = reinterpret_cast<const typename Tr::uvw_type *>(
            in_uvw.flat<FT>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto gauss_params = reinterpret_cast<const typename Tr::gauss_param_type *>(
            in_gauss_params.flat<FT>().data());
        auto gauss_shape = reinterpret_cast<typename Tr::gauss_shape_type *>(
            gauss_shape_ptr->flat<FT>().data());

        rime_gauss_shape<Tr><<<grid, block, 0, stream>>>(
            uvw, frequency, gauss_params, gauss_shape,
            montblanc::constants<FT>::gauss_scale,
            ngsrc, nvrow, nchan);

        cudaError_t e = cudaPeekAtLastError();
        if(e != cudaSuccess) {
            OP_REQUIRES_OK(context,
                tf::errors::Internal("Cuda Failure ", __FILE__, __LINE__, " ",
                                             cudaGetErrorString(e)));
        }

    }
};

MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_GAUSS_SHAPE_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
