#if GOOGLE_CUDA

#ifndef RIME_SERSIC_SHAPE_OP_GPU_CUH
#define RIME_SERSIC_SHAPE_OP_GPU_CUH

#include "constants.h"
#include "op_kernel_utils.h"
#include "sersic_shape_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SERSIC_SHAPE_NAMESPACE_BEGIN

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
__global__ void rime_sersic_shape(
    const typename Traits::uvw_type * uvw,
    const typename Traits::frequency_type * frequency,
    const typename Traits::sersic_param_type * sersic_params,
    typename Traits::sersic_shape_type * sersic_shape,
    const typename Traits::FT two_pi_over_c,
    int nssrc, int nvrows, int nchan)
{
    int chan = blockIdx.x*blockDim.x + threadIdx.x;
    int vrow = blockIdx.y*blockDim.y + threadIdx.y;

    using FT = typename Traits::FT;
    using LTr = LaunchTraits<FT>;
    using Po = montblanc::kernel_policies<FT>;

    constexpr FT one = FT(1.0);

    if(vrow >= nvrows || chan >= nchan)
        { return; }

    __shared__ struct {
        typename Traits::uvw_type uvw[LTr::BLOCKDIMY];
        typename Traits::frequency_type scaled_freq[LTr::BLOCKDIMX];
    } shared;

    int i;

    // UVW coordinates vary by baseline but not channel
    if(threadIdx.x == 0)
        { shared.uvw[threadIdx.y] = uvw[vrow]; }

    // Wavelength varies by channel, but not baseline
    if(threadIdx.y == 0)
        { shared.scaled_freq[threadIdx.x] = two_pi_over_c*frequency[chan]; }

    __syncthreads();

    // Reference u, v and w in shared memory for this thread
    FT & u = shared.uvw[threadIdx.y].x;
    FT & v = shared.uvw[threadIdx.y].y;


    for(int ssrc=0; ssrc < nssrc; ++ssrc)
    {
        i = ssrc;   FT e1 = cub::ThreadLoad<cub::LOAD_LDG>(sersic_params+i);
        i += nssrc; FT e2 = cub::ThreadLoad<cub::LOAD_LDG>(sersic_params+i);
        i += nssrc; FT ss = cub::ThreadLoad<cub::LOAD_LDG>(sersic_params+i);

        // sersic source in  the Fourier domain
        FT u1 = u*(one + e1) + v*e2;
        u1 *= shared.scaled_freq[threadIdx.x];
        u1 *= ss/(one - e1*e1 - e2*e2);

        FT v1 = u*e2 + v*(one - e1);
        v1 *= shared.scaled_freq[threadIdx.x];
        v1 *= ss/(one - e1*e1 - e2*e2);

        FT sersic_factor = one + u1*u1+v1*v1;

        i = (ssrc*nvrows + vrow)*nchan + chan;
        sersic_shape[i] = one / (ss*Po::sqrt(sersic_factor));
    }
}

// Specialise the SersicShape op for GPUs
template <typename FT>
class SersicShape<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit SersicShape(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;
        const tf::Tensor & in_uvw = context->input(0);
        const tf::Tensor & in_frequency = context->input(1);
        const tf::Tensor & in_sersic_params = context->input(2);

        int nvrows = in_uvw.dim_size(0);
        int nchan = in_frequency.dim_size(0);
        int nssrc = in_sersic_params.dim_size(1);

        tf::TensorShape sersic_shape_shape{nssrc, nvrows, nchan};

        // Allocate an output tensor
        tf::Tensor * sersic_shape_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, sersic_shape_shape, &sersic_shape_ptr));

        using LTr = LaunchTraits<FT>;
        using Tr = montblanc::kernel_traits<FT>;

        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            nchan, nvrows, 1);
        dim3 grid(montblanc::grid_from_thread_block(
            block, nchan, nvrows, 1));

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        auto uvw = reinterpret_cast<const typename Tr::uvw_type *>(
            in_uvw.flat<FT>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto sersic_params = reinterpret_cast<const typename Tr::sersic_param_type *>(
            in_sersic_params.flat<FT>().data());
        auto sersic_shape = reinterpret_cast<typename Tr::sersic_shape_type *>(
            sersic_shape_ptr->flat<FT>().data());

        rime_sersic_shape<Tr><<<grid, block, 0, stream>>>(
            uvw, frequency, sersic_params, sersic_shape,
            montblanc::constants<FT>::two_pi_over_c,
            nssrc, nvrows, nchan);

        cudaError_t e = cudaPeekAtLastError();
        if(e != cudaSuccess) {
            OP_REQUIRES_OK(context,
                tf::errors::Internal("Cuda Failure ", __FILE__, __LINE__, " ",
                                             cudaGetErrorString(e)));
        }

    }
};

MONTBLANC_SERSIC_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SERSIC_SHAPE_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
