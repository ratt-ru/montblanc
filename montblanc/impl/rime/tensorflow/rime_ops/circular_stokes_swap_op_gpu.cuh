#if GOOGLE_CUDA

#ifndef RIME_CIRCULAR_STOKES_SWAP_OP_GPU_CUH
#define RIME_CIRCULAR_STOKES_SWAP_OP_GPU_CUH

#include "circular_stokes_swap_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for type permutations
template <typename FT> struct LaunchTraits {};

// Specialise for float
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<float>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }
};
// Specialise for double
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }
};


// The base stokes index. 0 0 0 0 4 4 4 4 8 8 8 8
#define _MONTBLANC_STOKES_BASE_IDX int(cub::LaneId() & 28)

// CUDA kernel outline
template <typename FT>
__global__ void rime_circular_stokes_swap(
    const FT * stokes_in,
    FT * stokes_out,
    int nstokes, int npol)

{
    using LTr = LaunchTraits<FT>;

    // Index into flattened stokes*pol array
    int stokes_pol = blockIdx.x*blockDim.x + threadIdx.x;

    // Thread guard
    if(stokes_pol >= nstokes*npol)
        { return; }

    // Load in the polarisation
    FT pol = stokes_in[stokes_pol];

    // Get the polarisation index for this stokes parameter
    int POL = stokes_pol & (npol-1);

    // The following index shuffles [I Q U V] to [I V Q U]
    int swap_index[4] = { 0, 3, 1, 2 };

    // 0 3 1 2 4 7 5 6 8 11 9 10 ...
    int shfl_idx = _MONTBLANC_STOKES_BASE_IDX + swap_index[POL];

    // Write out shuffled polarisation
    stokes_out[stokes_pol] = cub::ShuffleIndex(pol, shfl_idx);
}

// Specialise the CircularStokesSwap op for GPUs
template <typename FT>
class CircularStokesSwap<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit CircularStokesSwap(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & stokes_in = context->input(0);

        int nsrc = stokes_in.dim_size(0);
        int ntime = stokes_in.dim_size(1);
        int npol = stokes_in.dim_size(2);
        int nstokes = nsrc*ntime;

        OP_REQUIRES(context, npol == 4,
            tf::errors::InvalidArgument("Circular Stokes Swap given '", npol,
                                        "' polarisations but can only handle 4 "
                                        "at this point in time."));


        // Allocate output tensors
        // Allocate space for output tensor 'stokes_out'
        tf::Tensor * stokes_out_ptr = nullptr;
        tf::TensorShape stokes_out_shape = tf::TensorShape({ nsrc, ntime, npol });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, stokes_out_shape, &stokes_out_ptr));


        using LTr = LaunchTraits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block(LTr::block_size(nstokes*npol, 1, 1));
        dim3 grid(montblanc::grid_from_thread_block(
            block, nstokes*npol, 1, 1));

        // Get pointers to flattened tensor data buffers
        const auto fin_stokes_in = stokes_in.flat<FT>().data();
        auto fout_stokes_out = stokes_out_ptr->flat<FT>().data();


        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the rime_circular_stokes_swap CUDA kernel
        rime_circular_stokes_swap<FT>
            <<<grid, block, 0, device.stream()>>>(
                fin_stokes_in, fout_stokes_out,
                nstokes, npol);
    }
};

MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CIRCULAR_STOKES_SWAP_OP_GPU_CUH

#endif // #if GOOGLE_CUDA