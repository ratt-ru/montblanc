#ifndef RIME_B_SQRT_OP_GPU_H_
#define RIME_B_SQRT_OP_GPU_H_

#if GOOGLE_CUDA

#include "b_sqrt_op.h"
#include <montblanc/abstraction.cuh>
#include <montblanc/brightness.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace montblanc {
namespace bsqrt {

// Traits class defined by float and complex types
template <typename FT> class LaunchTraits;

// Specialise for float
template <> class LaunchTraits<float>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }        
};

// Specialise for double
template <> class LaunchTraits<double>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }        
};

} // namespace montblanc {
} // namespace bsqrt {

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;    

constexpr int BSQRT_NPOL = 4;

template <typename Traits>
__global__ void rime_b_sqrt(
    const typename Traits::stokes_type * stokes,
    const typename Traits::alpha_type * alpha,
    const typename Traits::frequency_type * frequency,
    typename Traits::B_sqrt_type * B_sqrt,
    typename Traits::FT ref_freq,
    int nsrc, int ntime, int npolchan)
{
    // Simpler float and complex types
    typedef typename Traits::FT FT;
    typedef typename Traits::CT CT;

    typedef typename montblanc::kernel_policies<FT> Po;
    typedef typename montblanc::bsqrt::LaunchTraits<FT> LTr;

    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;
    #define POL (threadIdx.x & 0x3)

    if(SRC >= nsrc || TIME >= ntime || POLCHAN >= npolchan)
        return;

    __shared__  FT freq[LTr::BLOCKDIMX];

    // TODO. Using 3 times more shared memory than we
    // really require here, since there's only
    // one frequency per channel.
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        freq[threadIdx.x] = frequency[POLCHAN >> 2];
    }

    __syncthreads();

    // Calculate the power term
    int i = SRC*ntime + TIME;
    FT freq_ratio = freq[threadIdx.x]/ref_freq;
    FT power = Po::pow(freq_ratio, alpha[i]);

    // Read in the stokes parameter,
    // multiplying it by the power term
    i = i*BSQRT_NPOL + POL;
    FT pol = stokes[i]*power;
    CT B_square_root;

    // Create the square root of the brightness matrix
    montblanc::create_brightness_sqrt<FT>(B_square_root, pol);

    // Write out the square root of the brightness
    i = (SRC*ntime + TIME)*npolchan + POLCHAN;
    B_sqrt[i] = B_square_root;
    #undef POL
}

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
        int npolchan = nchan*BSQRT_NPOL;

        // Reason about our output shape
        tf::TensorShape b_sqrt_shape({nsrc, ntime, nchan, 4});

        // Create a pointer for the b_sqrt result
        tf::Tensor * b_sqrt_ptr = nullptr;

        // Allocate memory for the b_sqrt
        OP_REQUIRES_OK(context, context->allocate_output(
            0, b_sqrt_shape, &b_sqrt_ptr));

        if (b_sqrt_ptr->NumElements() == 0)
            { return; }

        // Cast input into CUDA types defined within the Traits class
        typedef montblanc::kernel_traits<FT> Tr;
        typedef montblanc::bsqrt::LaunchTraits<FT> LTr;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(npolchan, ntime, nsrc));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, npolchan, ntime, nsrc));

        printf("Threads per block: X %d Y %d Z %d\n",
            blocks.x, blocks.y, blocks.z);

        printf("Grid: X %d Y %d Z %d\n",
            grid.x, grid.y, grid.z);

        // Get the device pointers of our GPU memory arrays
        auto stokes = reinterpret_cast<const typename Tr::stokes_type *>(
            in_stokes.flat<FT>().data());
        auto alpha = reinterpret_cast<const typename Tr::alpha_type *>(
            in_alpha.flat<FT>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto b_sqrt = reinterpret_cast<typename Tr::B_sqrt_type *>(
            b_sqrt_ptr->flat<CT>().data());

        // By contrast we can access ref_freq on the host 
        // because of the .HostMemory("ref_freq") directive
        // used in REGISTER_KERNEL_BUILDER
        FT ref_freq = in_ref_freq.tensor<FT, 1>()(0);

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        rime_b_sqrt<Tr> <<<grid, blocks, 0, stream>>>(
            stokes, alpha, frequency, b_sqrt, ref_freq,
            nsrc, ntime, npolchan);
    }
};

} // namespace tensorflow {

#endif

#endif // #ifndef RIME_B_SQRT_OP_GPU_H_