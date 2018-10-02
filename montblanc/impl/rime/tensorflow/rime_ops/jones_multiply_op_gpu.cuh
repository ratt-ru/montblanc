#if GOOGLE_CUDA

#ifndef RIME_JONES_MULTIPLY_OP_GPU_CUH
#define RIME_JONES_MULTIPLY_OP_GPU_CUH

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU


#include "jones_multiply_op.h"
#include "shapes.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <montblanc/abstraction.cuh>
#include <montblanc/jones.cuh>


MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for type permutations
template <typename FT> struct LaunchTraits {};

// Specialise for float, tensorflow::complex64
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<float>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;
};

// Specialise for double, tensorflow::complex128
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;
};

constexpr int MAX_TENSORS = 10;

// Get the current correlation from the thread ID
__device__ __forceinline__ int _jones_corr()
    { return threadIdx.x & 0x3; }

// CUDA kernel outline
template <typename Traits>
__global__ void rime_jones_multiply(
    const typename Traits::CT ** in_in,
    const uint32_t * in_shapes,
    typename Traits::CT * out_out,
    int ntensors, int ntensor_elements,
    int nsrc, int ntime, int na,
    int ncorrchan)

{
    // Shared memory usage unnecessary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;

    __shared__ const CT * tensor_ptrs[MAX_TENSORS];
    __shared__ uint32_t tensor_sizes[MAX_TENSORS*MAX_TENSOR_NDIM];

    uint32_t i;

    uint32_t corrchan = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t ant = blockIdx.y*blockDim.y + threadIdx.y;
    uint32_t time = blockIdx.z*blockDim.z + threadIdx.z;

    // 3D thread ID
    i = threadIdx.z*blockDim.x*blockDim.y
        + threadIdx.y*blockDim.x
        + threadIdx.x;

    // Fill shared memory
    if(i < ntensors)
        { tensor_ptrs[i] = in_in[i]; }

    if(i < ntensors*ntensor_elements)
        { tensor_sizes[i] = in_shapes[i]; }

    __syncthreads();

    if(time >= ntime || ant >= na || corrchan >= ncorrchan)
        { return; }

    // Iterate over sources and then tensors
    // Necessary to do it this way as
    for(uint32_t osrc=0; osrc < nsrc; ++osrc)
    {
        // Initialise result to identity
        CT result = montblanc::jones_identity<FT>();

        for(uint32_t j=0; j<ntensors; ++j)
        {
            // Dimensions of this tensor
            const uint32_t & nisrc = tensor_sizes[j*ntensor_elements + 0];
            const uint32_t & nitime = tensor_sizes[j*ntensor_elements + 1];
            const uint32_t & niant = tensor_sizes[j*ntensor_elements + 2];
            const uint32_t & nichan = tensor_sizes[j*ntensor_elements + 3];
            const uint32_t & nicorr = tensor_sizes[j*ntensor_elements + 4];
            const uint32_t nicorrchan = nichan*nicorr;

            // Input indices are either 0 or equal to the
            // output indices of the greater solution space
            const uint32_t isrc = nisrc == 1 ? 0 : osrc;
            const uint32_t itime = nitime == 1 ? 0 : time;
            const uint32_t iant = niant == 1 ? 0 : ant;
            const uint32_t icorrchan =
                    // No correlations or channels case
                    (nicorrchan == 1 ? 0 :
                    // Correlations only case
                    (nicorrchan == nicorr ? _jones_corr() :
                    // Channels only case
                    (nicorrchan == nichan ? corrchan / 4 :
                    // Should never happen!
                     corrchan)));

            // Load in the value for this tensor,
            // attempting to take advantage of any values
            // stored in the readonly L1 cache
            i = ((isrc*nitime + itime)*niant + iant)*nicorrchan + icorrchan;
            CT in = cub::ThreadLoad<cub::LOAD_LDG>(tensor_ptrs[j] + i);

            // Handle the no-correlation case
            if(nicorr == 1)
                { montblanc::complex_multiply_in_place<FT>(result, in); }
            else
                { montblanc::jones_multiply_4x4_in_place<FT>(result, in); }
        }

        // Set shared buffer to thread index
        i = ((osrc*ntime + time)*na + ant)*ncorrchan + corrchan;
        out_out[i] = result;
    }
}

// Specialise the JonesMultiply op for GPUs
template <typename FT, typename CT>
class JonesMultiply<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    std::string str_output_schema;
    std::vector<std::string> schemas;
    std::vector<std::string> output_schema;
    std::unordered_map<std::string, int> output_index;
    bool squeeze;
    int N;

public:
    explicit JonesMultiply(tensorflow::OpKernelConstruction * context)
        : tensorflow::OpKernel(context)
    {
        namespace tf = tensorflow;

        OP_REQUIRES_OK(context, context->GetAttr("schemas", &schemas));
        OP_REQUIRES_OK(context, context->GetAttr("N", &N));
        OP_REQUIRES_OK(context, context->GetAttr("squeeze", &squeeze));
        OP_REQUIRES_OK(context, context->GetAttr("output_schema", &str_output_schema));

        OP_REQUIRES_OK(context,
            parse_shape_schema(str_output_schema, output_schema));

        OP_REQUIRES(context, MAX_TENSOR_NDIM - output_schema.size() >= 0,
                tf::errors::InvalidArgument("Output schema size ",
                    output_schema.size(), " exceeds ", MAX_TENSOR_NDIM));

        int diff = MAX_TENSOR_NDIM - output_schema.size();

        for(int i=0; i < output_schema.size(); ++i)
            { output_index.insert({output_schema[i], diff + i}); }
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        tensorflow::OpInputList in_list;
        context->input_list("jones_in", & in_list);

        OP_REQUIRES(context, in_list.size() == schemas.size(),
            tf::errors::InvalidArgument("Number of schemas ", schemas.size(),
                                        " does not match number of inputs ",
                                        in_list.size()));

        OP_REQUIRES(context, in_list.size() <= MAX_TENSORS,
            tf::errors::InvalidArgument("Only ", MAX_TENSORS,
                                        " Jones matrices supported"));

        OP_REQUIRES(context, output_schema.size() <= MAX_TENSOR_NDIM,
            tf::errors::InvalidArgument("Only ", MAX_TENSOR_NDIM,
                                        " output_schema elements supported"));


        std::vector<std::vector<tf::int64>> reshapes;
        reshapes.reserve(in_list.size());
        std::unordered_map<std::string, int> output_sizes;

        OP_REQUIRES_OK(context, infer_dimensionality(in_list,
                                 schemas, str_output_schema,
                                 output_schema, output_index, reshapes,
                                 output_sizes));


        // Get pointers to flattened tensor data buffers
        using Tr = montblanc::kernel_traits<FT>;
        using LTr = LaunchTraits<FT>;

        // Determine output tensor shape, this may be < MAX_TENSOR_NDIM
        tf::TensorShape output_shape;
        // Reshape output tensor to MAX_TENSOR_NDIM
        std::vector<tf::int64> out_reshape;

        for(int i=0; i < MAX_TENSOR_NDIM - output_schema.size(); ++i)
        {
            out_reshape.push_back(1);
        }

        for(int i=0; i<output_schema.size(); ++i)
        {
            // Was this output dimension in the inputs?
            auto it = output_sizes.find(output_schema[i]);

            // No
            if(it == output_sizes.end())
            {
                out_reshape.push_back(1);

                // Ignore if we're squeezing else set to 1
                if(squeeze)
                    { continue; }

                output_shape.AddDim(1);
            }
            else
            {
                out_reshape.push_back(it->second);
                output_shape.AddDim(it->second);
            }
        }

        // Allocate an output tensor
        tf::Tensor * output_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, output_shape, &output_ptr));

        // Create a Pinned Memory Allocator
        tf::AllocatorAttributes pinned_allocator;
        pinned_allocator.set_gpu_compatible(true);
        pinned_allocator.set_on_host(true);

        // Create a GPU Allocator
        tf::AllocatorAttributes gpu_allocator;
        gpu_allocator.set_gpu_compatible(true);

        // Tensors in pinned host and gpu memory
        // which contain pointers to the input arrays
        // of Jones matrices
        std::size_t input_arrays_bytes = in_list.size() * sizeof(CT *);

        tf::Tensor h_input_arrays;
        tf::Tensor d_input_arrays;

        tf::TensorShape input_arrays_shape = tf::TensorShape({
            (long long)input_arrays_bytes });

        // GPU Array
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DT_UINT8, input_arrays_shape,
            &d_input_arrays, gpu_allocator));

        // Pinned Memory
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DT_UINT8, input_arrays_shape,
            &h_input_arrays, pinned_allocator));

        // Tensors in pinned host and gpu memory
        // which contain pointers to the sizes of the input
        // arrays of Jones matrices
        tf::TensorShape array_size_shape({(long long) in_list.size(),
                                          (long long) out_reshape.size()});

        tf::Tensor h_array_sizes;
        tf::Tensor d_array_sizes;

        // GPU Array
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DT_UINT32, array_size_shape,
            &d_array_sizes, gpu_allocator));

        // Pinned Memory
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DT_UINT32, array_size_shape,
            &h_array_sizes, pinned_allocator));

        auto host_input_array_ptrs = reinterpret_cast<const typename Tr::CT **>(
                            h_input_arrays.flat<tf::uint8>().data());

        auto dev_input_array_ptrs = reinterpret_cast<const typename Tr::CT **>(
                            d_input_arrays.flat<tf::uint8>().data());

        auto host_array_sizes = h_array_sizes.tensor<uint32_t, 2>();

        auto dev_array_size_ptrs = reinterpret_cast<const uint32_t *>(
                            d_array_sizes.flat<uint32_t>().data());

        auto output = reinterpret_cast<typename Tr::CT *>(
                            output_ptr->flat<CT>().data());

        // Set the input array pointers and sizes
        for(int i=0; i < in_list.size(); ++i)
        {
            const tf::Tensor & tensor = in_list[i];
            auto & shape = reshapes[i];
            host_input_array_ptrs[i] = reinterpret_cast<const typename Tr::CT *>(
                            tensor.flat<CT>().data());

            for(int s=0; s < out_reshape.size(); ++s)
                { host_array_sizes(i, s) = shape[s]; }
        }

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Copy array of tensor pointers to the device
        cudaMemcpyAsync((void *) dev_input_array_ptrs,
            (const void *) host_input_array_ptrs,
            input_arrays_bytes,
            cudaMemcpyHostToDevice,
            device.stream());

        // Copy array of tensor sizes to the device
        cudaMemcpyAsync((void *) dev_array_size_ptrs,
            (const void *) host_array_sizes.data(),
            h_array_sizes.TotalBytes(),
            cudaMemcpyHostToDevice,
            device.stream());

        int nsrc = out_reshape[0];
        int ntime = out_reshape[1];
        int na = out_reshape[2];
        int nchan = out_reshape[3];
        int ncorr = out_reshape[4];
        int npolchan = nchan*ncorr;

        int ntensors = in_list.size();
        int ntensor_elements = out_reshape.size();

        OP_REQUIRES(context, ntensors < MAX_TENSORS,
            tf::errors::InvalidArgument("ntensors ",
                ntensors, " >= ", MAX_TENSORS));

        OP_REQUIRES(context, ntensors < MAX_TENSORS,
            tf::errors::InvalidArgument("ntensor_elements ",
                ntensor_elements, " != ", MAX_TENSOR_NDIM));

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            npolchan, na, ntime);
        dim3 grid(montblanc::grid_from_thread_block(
            block, npolchan, na, ntime));

        // Call the rime_jones_multiply CUDA kernel
        rime_jones_multiply<Tr>
            <<<grid, block, 0, device.stream()>>>(
                dev_input_array_ptrs,
                dev_array_size_ptrs,
                output,
                ntensors, ntensor_elements,
                nsrc, ntime, na, npolchan);

    }
};

MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_JONES_MULTIPLY_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
