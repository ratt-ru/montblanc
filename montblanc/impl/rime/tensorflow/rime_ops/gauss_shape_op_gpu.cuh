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
    const typename Traits::antenna_type * antenna1,
    const typename Traits::antenna_type * antenna2,
    const typename Traits::frequency_type * frequency,
    const typename Traits::gauss_param_type * gauss_params,
    typename Traits::gauss_shape_type * gauss_shape,
    const typename Traits::FT gauss_scale,
    int ngsrc, int ntime, int nbl, int na, int nchan)
{
    int chan = blockIdx.x*blockDim.x + threadIdx.x;
    int bl = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;

    using FT = typename Traits::FT;
    using LTr = LaunchTraits<FT>;
    using Po = montblanc::kernel_policies<FT>;

    if(time >= ntime || bl >= nbl || chan >= nchan)
        { return; }

    __shared__ struct {
        typename Traits::uvw_type uvw[LTr::BLOCKDIMZ][LTr::BLOCKDIMY];
        typename Traits::frequency_type scaled_freq[LTr::BLOCKDIMX];
    } shared;

    // Reference u, v and w in shared memory for this thread
    FT & u = shared.uvw[threadIdx.z][threadIdx.y].x;
    FT & v = shared.uvw[threadIdx.z][threadIdx.y].y;
    FT & w = shared.uvw[threadIdx.z][threadIdx.y].z;

    // Retrieve antenna pairs for the current baseline
    int i = time*nbl + bl;
    int ant1 = antenna1[i];
    int ant2 = antenna2[i];

    // UVW coordinates vary by baseline and time, but not channel
    if(threadIdx.x == 0)
    {
        // UVW, calculated from u_pq = u_p - u_q
        i = time*na + ant2;
        shared.uvw[threadIdx.z][threadIdx.y] = uvw[i];

        i = time*na + ant1;
        typename Traits::uvw_type ant1_uvw = uvw[i];
        u -= ant1_uvw.x;
        v -= ant1_uvw.y;
        w -= ant1_uvw.z;
    }

    // Wavelength varies by channel, but not baseline and time
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { shared.scaled_freq[threadIdx.x] = gauss_scale*frequency[chan]; }

    __syncthreads();

    for(int gsrc=0; gsrc < ngsrc; ++gsrc)
    {
        i = gsrc;   FT el = cub::ThreadLoad<cub::LOAD_LDG>(gauss_params+i);
        i += ngsrc; FT em = cub::ThreadLoad<cub::LOAD_LDG>(gauss_params+i);
        i += ngsrc; FT eR = cub::ThreadLoad<cub::LOAD_LDG>(gauss_params+i);

        FT u1 = u*em - v*el;
        u1 *= shared.scaled_freq[threadIdx.x]*eR;

        FT v1 = u*el + v*em;
        v1 *= shared.scaled_freq[threadIdx.x];

        i = ((gsrc*ntime + time)*nbl + bl)*nchan + chan;
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
        const tf::Tensor & in_antenna1 = context->input(1);
        const tf::Tensor & in_antenna2 = context->input(2);
        const tf::Tensor & in_frequency = context->input(3);
        const tf::Tensor & in_gauss_params = context->input(4);

        int ntime = in_uvw.dim_size(0);
        int na = in_uvw.dim_size(1);
        int nbl = in_antenna1.dim_size(1);
        int nchan = in_frequency.dim_size(0);
        int ngsrc = in_gauss_params.dim_size(1);

        tf::TensorShape gauss_shape_shape{ngsrc, ntime, nbl, nchan};

        // Allocate an output tensor
        tf::Tensor * gauss_shape_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, gauss_shape_shape, &gauss_shape_ptr));

        using LTr = LaunchTraits<FT>;
        using Tr = montblanc::kernel_traits<FT>;

        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            nchan, nbl, ntime);
        dim3 grid(montblanc::grid_from_thread_block(
            block, nchan, nbl, ntime));

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        auto uvw = reinterpret_cast<const typename Tr::uvw_type *>(
            in_uvw.flat<FT>().data());
        auto antenna1 = reinterpret_cast<const typename Tr::antenna_type *>(
            in_antenna1.flat<int>().data());
        auto antenna2 = reinterpret_cast<const typename Tr::antenna_type *>(
            in_antenna2.flat<int>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto gauss_params = reinterpret_cast<const typename Tr::gauss_param_type *>(
            in_gauss_params.flat<FT>().data());
        auto gauss_shape = reinterpret_cast<typename Tr::gauss_shape_type *>(
            gauss_shape_ptr->flat<FT>().data());

        rime_gauss_shape<Tr><<<grid, block, 0, stream>>>(
            uvw, antenna1, antenna2,
            frequency, gauss_params, gauss_shape,
            montblanc::constants<FT>::gauss_scale,
            ngsrc, ntime, nbl, na, nchan);
    }
};

MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_GAUSS_SHAPE_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
