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

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

constexpr int BSQRT_NPOL = 4;

template <typename Traits> __device__ __forceinline__
int bsqrt_pol()
    { return threadIdx.x & 0x3; }

template <typename Traits>
__global__ void rime_b_sqrt(
    const typename Traits::stokes_type * stokes,
    const typename Traits::alpha_type * alpha,
    const typename Traits::frequency_type * frequency,
    const typename Traits::frequency_type * ref_freq,
    typename Traits::visibility_type * B_sqrt,
    typename Traits::neg_ant_jones_type * neg_ant_jones,
    int nsrc, int ntime, int nchan)
{
    // Simpler float and complex types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using ST = typename Traits::stokes_type;
    using VT = typename Traits::visibility_type;

    typedef typename montblanc::kernel_policies<FT> Po;
    typedef typename montblanc::bsqrt::LaunchTraits<FT> LTr;

    int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;

    if(SRC >= nsrc || TIME >= ntime || CHAN >= nchan)
        { return; }

    __shared__  FT freq_ratio[LTr::BLOCKDIMX];

    // TODO. Using 3 times more shared memory than we
    // really require here, since there's only
    // one frequency per channel.
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        freq_ratio[threadIdx.x] = (frequency[CHAN] / ref_freq[CHAN]);
    }

    __syncthreads();

    // Calculate power term
    int i = SRC*ntime + TIME;
    FT power = Po::pow(freq_ratio[threadIdx.x], alpha[i]);

    // Read in stokes parameters (IQUV)
    ST _stokes = stokes[i];
    FT & I = _stokes.x;
    FT & Q = _stokes.y;
    FT & U = _stokes.z;
    FT & V = _stokes.w;

    // Sign variable, used to attempt to ensure
    // positive definiteness of the brightness matrix
    // and a valid Cholesky decomposition. Exists to
    // handle cases where we have negative flux
    FT sign = 1.0;

    if(I + Q < 0.0)
        { sign = -1.0; }

    I *= power;
    Q *= power;
    U *= power;
    V *= power;

    // Create the cholesky decomposition of the brightness matrix
    // L00 = sqrt(I+Q)
    // L01 = 0
    // L10 = (U+iV)/sqrt(I+Q)
    // L11 = sqrt(I - Q - L10*conj(L10))

    // Use the YY and XY correlations as scratch space
    VT B;
    // sqrt(I+Q)
    B.XX = Po::sqrt(Po::make_ct((I+Q)*sign, 0.0));
    // (U-iV)/sqrt(I+Q)
    B.YY = Po::make_ct(U*sign, -V*sign);
    FT r2 = Po::abs_squared(B.XX);
    B.YX = Po::make_ct(
        (B.YY.x*B.XX.x + B.YY.y*B.XX.y)/r2,
        (B.YY.y*B.XX.x - B.YY.x*B.XX.y)/r2);

    montblanc::complex_conjugate_multiply<FT>(B.XY, B.YX, B.YX);
    B.XY.x = -B.XY.x;
    B.XY.y = -B.XY.y;
    B.XY.x += sign*(I - Q);

    B.YY = Po::sqrt(B.XY);
    B.XY = Po::make_ct(0.0, 0.0);

    i = SRC*ntime + TIME;

    // Indicate that we inverted the sign of the brightness
    // matrix to obtain the cholesky decomposition
    if(CHAN == 0)
        { neg_ant_jones[i] = (sign == 1.0 ? 1 : -1); }

    B_sqrt[i*nchan + CHAN] = B;
}

template <typename FT, typename CT>
class BSqrt<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit BSqrt(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

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

        OP_REQUIRES(context, in_ref_freq.dims() == 1,
            tf::errors::InvalidArgument(
                "ref_freq should be of shape (nchan)"))

        // Extract problem dimensions
        int nsrc = in_stokes.dim_size(0);
        int ntime = in_stokes.dim_size(1);
        int nchan = in_frequency.dim_size(0);

        // Reason about the shape of the b_sqrt tensor and
        // create a pointer to it
        tf::TensorShape b_sqrt_shape({nsrc, ntime, nchan, 4});
        tf::Tensor * b_sqrt_ptr = nullptr;

        // Allocate memory for the b_sqrt
        OP_REQUIRES_OK(context, context->allocate_output(
            0, b_sqrt_shape, &b_sqrt_ptr));

        if (b_sqrt_ptr->NumElements() == 0)
            { return; }

        // Reason about shape of the invert tensor
        // and create a pointer to it
        tf::TensorShape invert_shape({nsrc, ntime});
        tf::Tensor * invert_ptr = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(
            1, invert_shape, &invert_ptr));

        // Cast input into CUDA types defined within the Traits class
        typedef montblanc::kernel_traits<FT> Tr;
        typedef montblanc::bsqrt::LaunchTraits<FT> LTr;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(nchan, ntime, nsrc));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, nchan, ntime, nsrc));

        //printf("Threads per block: X %d Y %d Z %d\n",
        //    blocks.x, blocks.y, blocks.z);

        //printf("Grid: X %d Y %d Z %d\n",
        //    grid.x, grid.y, grid.z);

        // Get the device pointers of our GPU memory arrays
        auto stokes = reinterpret_cast<const typename Tr::stokes_type *>(
            in_stokes.flat<FT>().data());
        auto alpha = reinterpret_cast<const typename Tr::alpha_type *>(
            in_alpha.flat<FT>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto ref_freq = reinterpret_cast<const typename Tr::frequency_type *>(
            in_ref_freq.flat<FT>().data());
        auto b_sqrt = reinterpret_cast<typename Tr::visibility_type *>(
            b_sqrt_ptr->flat<CT>().data());
        auto neg_ant_jones = reinterpret_cast<typename Tr::neg_ant_jones_type *>(
            invert_ptr->flat<tf::int8>().data());

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        rime_b_sqrt<Tr> <<<grid, blocks, 0, stream>>>(stokes,
            alpha, frequency, ref_freq,
            b_sqrt, neg_ant_jones,
            nsrc, ntime, nchan);
    }
};

} // namespace sqrt {
} // namespace montblanc {

#endif

#endif // #ifndef RIME_B_SQRT_OP_GPU_H_