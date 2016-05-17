#ifndef RIME_PHASE_OP_GPU_H_
#define RIME_PHASE_OP_GPU_H_

#if GOOGLE_CUDA

#include "phase_op.h"

#define EIGEN_USE_GPU

namespace tensorflow {

// Ensure that the dimensions of the supplied block
// are not greater than (X,Y,Z)
dim3 modify_small_dims(dim3 && block, int X, int Y, int Z)
{
    if(X < block.x)
        { block.x = X; }

    if(Y < block.y)
        { block.y = Y; }

    if(Z < block.z)
        { block.z = Z; }

    return std::move(block);
}

dim3 grid_from_thread_block(const dim3 & block, int X, int Y, int Z)
{
    int GX = X / block.x;
    int GY = Y / block.y;
    int GZ = Z / block.z;

    return dim3(GX, GY, GZ);
}

typedef Eigen::GpuDevice GPUDevice;

// Traits class defined by float and complex types
template <typename FT, typename CT>
class RimePhaseTraits;

// Specialise for float and complex64
template <>
class RimePhaseTraits<float, tensorflow::complex64>
{
public:
    typedef float FT;
    typedef float2 CT;
    typedef float2 lm_type;
    typedef float3 uvw_type;
    typedef float frequency_type;
    typedef float2 complex_phase_type;

    __device__ __forceinline__ static
    CT make_complex(const FT & real, const FT & imag)
        { return ::make_float2(real, imag); }

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return modify_small_dims(dim3(32, 8, 2),
            nchan, na, ntime);
    }        
};

// Specialise for double and complex128
template <>
class RimePhaseTraits<double, tensorflow::complex128>
{
public:
    typedef double FT;
    typedef double2 CT;
    typedef double2 lm_type;
    typedef double3 uvw_type;
    typedef double frequency_type;
    typedef double2 complex_phase_type;

    __device__ __forceinline__ static
    CT make_complex(const FT & real, const FT & imag)
        { return ::make_double2(real, imag); }

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return modify_small_dims(dim3(32, 4, 1),
            nchan, na, ntime);
    }        
};

// CUDA kernel computing the phase term
template <typename Traits>
__global__ void rime_phase(
    const typename Traits::lm_type * lm,
    const typename Traits::uvw_type * uvw,
    const typename Traits::frequency_type * frequency,
    typename Traits::complex_phase_type * complex_phase,
    int32 nsrc, int32 ntime, int32 na, int32 nchan)
{
    typedef typename Traits::FT FT;

    int chan = blockIdx.x*blockDim.x + threadIdx.x;
    int ant = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;

    if(chan >= nchan || ant >= na || time >= ntime)
        { return; }

    for(int src=0; src < nsrc; ++src)
    {
        typename Traits::lm_type r_lm = lm[src];
        // TODO: This code doesn't do anything sensible yet
        FT n = FT(1.0) - r_lm.x*r_lm.x - r_lm.y*r_lm.y;
        complex_phase[nsrc] = Traits::make_complex(n, 0.0);
    }
}

// Partially specialise RimePhaseOp for GPUDevice
template <typename FT, typename CT>
class RimePhaseOp<GPUDevice, FT, CT> : public tensorflow::OpKernel {
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

        // Cast input into CUDA types defined within the Traits class
        typedef RimePhaseTraits<FT, CT> Tr;

        // Set up our kernel dimensions
        dim3 blocks(Tr::block_size(nchan, na, ntime));
        dim3 grid(grid_from_thread_block(blocks, nchan, na, ntime));

        auto lm = reinterpret_cast<const typename Tr::lm_type *>(
            in_lm.flat<FT>().data());
        auto uvw = reinterpret_cast<const typename Tr::uvw_type *>(
            in_uvw.flat<FT>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto complex_phase = reinterpret_cast<
            typename Tr::complex_phase_type *>(
                complex_phase_ptr->flat<CT>().data());

        // Get the stream
        const auto & stream = context->eigen_device<GPUDevice>().stream();

        // Invoke the kernel, casting to the expected types
        rime_phase<Tr> <<<grid, blocks, 0, stream>>>(
            lm, uvw, frequency, complex_phase,
            nsrc, ntime, na, nchan);
    }
};

} // namespace tensorflow {

#endif // #if GOOGLE_CUDA

#endif // #define RIME_PHASE_OP_GPU_H