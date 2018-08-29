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
constexpr int MAX_TENSOR_ELEMENTS = 5;

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
    __shared__ uint32_t tensor_sizes[MAX_TENSORS*MAX_TENSOR_ELEMENTS];

    uint32_t i;

    uint32_t corrchan = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t ant = blockIdx.y*blockDim.y + threadIdx.y;
    uint32_t time = blockIdx.z*blockDim.z + threadIdx.z;

    if(time >= ntime || ant >= na || corrchan >= ncorrchan)
        { return; }

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

    // Iterate over sources and then tensors
    // Necessary to do it this way as
    for(uint32_t osrc=0; osrc < nsrc; ++osrc)
    {
        // Initialise result to identity
        CT result = montblanc::jones_identity<FT>();

        for(uint32_t j=0; j<ntensors; ++j)
        {
            // Dimensions of this tensors
            const uint32_t & nisrc = tensor_sizes[j*ntensor_elements + 0];
            const uint32_t & nitime = tensor_sizes[j*ntensor_elements + 1];
            const uint32_t & niant = tensor_sizes[j*ntensor_elements + 2];
            const uint32_t & nichan = tensor_sizes[j*ntensor_elements + 3];
            const uint32_t & nicorr = tensor_sizes[j*ntensor_elements + 4];
            const uint32_t nicorrchan = nichan*nicorr;

            const uint32_t isrc = nisrc == 1 ? 0 : osrc;
            const uint32_t itime = nitime == 1 ? 0 : time;
            const uint32_t iant = niant == 1 ? 0 : ant;
            const uint32_t icorrchan = nichan == 1 ? _jones_corr() : corrchan;

            // Load in the value for this tensor,
            // attempting to take advantage of any values stored
            // in the readonly L1 cache
            i = ((isrc*nitime + itime)*niant + iant)*nicorrchan + icorrchan;
            CT in = cub::ThreadLoad<cub::LOAD_LDG>(tensor_ptrs[j] + i);

            montblanc::jones_multiply_4x4_in_place<FT>(result, in);
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
    int N;

public:
    explicit JonesMultiply(tensorflow::OpKernelConstruction * context)
        : tensorflow::OpKernel(context),
          str_output_schema("(source,time,ant,chan,corr)")
    {
        OP_REQUIRES_OK(context, context->GetAttr("schemas",
                                                 &schemas));
        OP_REQUIRES_OK(context, context->GetAttr("N", &N));


        OP_REQUIRES_OK(context,
            parse_shape_schema(str_output_schema, output_schema));

        for(int i=0; i < output_schema.size(); ++i)
            { output_index.insert({output_schema[i], i}); }
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        std::unordered_map<std::string, int> output_sizes;
        tensorflow::OpInputList in_list;
        context->input_list("in", & in_list);

        OP_REQUIRES(context, in_list.size() == schemas.size(),
            tf::errors::InvalidArgument("Number of schemas ", schemas.size(),
                                        " does not match number of inputs ",
                                        in_list.size()));

        OP_REQUIRES(context, in_list.size() <= MAX_TENSORS,
            tf::errors::InvalidArgument("Only ", MAX_TENSORS,
                                        " Jones matrices supported"));

        OP_REQUIRES(context, output_schema.size() <= MAX_TENSOR_ELEMENTS,
            tf::errors::InvalidArgument("Only ", MAX_TENSOR_ELEMENTS,
                                        " output_schema elements supported"));


        std::vector<std::vector<tf::int64>> reshapes;
        reshapes.reserve(in_list.size());

        for(int i=0; i<in_list.size(); ++i)
        {
            // Get the tensor shape
            const tf::TensorShape shape = in_list[i].shape();

            // Get the associated shape schema
            std::vector<std::string> schema;
            OP_REQUIRES_OK(context, parse_shape_schema(schemas[i], schema));

            // Number of elements in shape and schema must match
            OP_REQUIRES(context, schema.size() == shape.dims(),
                tf::errors::InvalidArgument("schema ", schemas[i], " "
                                            "shape does not match "
                                            "in[", i, "].shape of ",
                                            shape.DebugString()));

            // Work out the dimension sizes needed to reshape
            // the tensor rank up to that of the output schema.
            // Introduce 1's for missing dimensions
            std::vector<tf::int64> reshape;
            reshape.reserve(output_schema.size());

            // Start out with all 1.
            for(int j=0; j<output_schema.size(); ++j)
                { reshape.push_back(1); }

            for(int j=0; j<schema.size(); ++j)
            {
                // Either set the output size for this
                // schema dimension or check that it matches
                // a previously found value
                auto size_it = output_sizes.find(schema[j]);

                if(size_it == output_sizes.end())
                {
                    output_sizes.insert({schema[j], shape.dim_size(j)});
                }
                else
                {
                    OP_REQUIRES(context,
                       size_it->second == shape.dim_size(j),
                       tf::errors::InvalidArgument("Existing size ",
                           size_it->second, " for dimension ", schema[j],
                           " does not match ", shape.dim_size(j),
                           " found in input tensor ", i));
                }


                // Find index of schema dimension in output schema
                auto it = output_index.find(schema[j]);

                OP_REQUIRES(context, it != output_index.end(),
                    tf::errors::InvalidArgument(schema[j], " is not part "
                                                "of the output schema ",
                                                str_output_schema));

                // Set the dimension size at the output index
                // to the shape size
                reshape[it->second] = shape.dim_size(j);
            }

            reshapes.emplace_back(reshape);
        }


        // Get pointers to flattened tensor data buffers
        using Tr = montblanc::kernel_traits<FT>;
        using LTr = LaunchTraits<FT>;

        // Determine output tensor shape
        tf::TensorShape output_shape;

        for(int i=0; i<output_schema.size(); ++i)
        {
            auto it = output_sizes.find(output_schema[i]);

            // Set to 1 if we couldn't infer the size
            if(it == output_sizes.end())
                { output_shape.AddDim(1); }
            else
                { output_shape.AddDim(it->second); }
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
                                          (long long) output_schema.size()});

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

        // Set the input array sizes
        for(int i=0; i < in_list.size(); ++i)
        {
            const tf::Tensor & tensor = in_list[i];
            auto & shape = reshapes[i];
            host_input_array_ptrs[i] = reinterpret_cast<const typename Tr::CT *>(
                            tensor.flat<CT>().data());

            for(int s=0; s < output_schema.size(); ++s)
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

        int nsrc = output_ptr->dim_size(0);
        int ntime = output_ptr->dim_size(1);
        int na = output_ptr->dim_size(2);
        int nchan = output_ptr->dim_size(3);
        int ncorr = output_ptr->dim_size(4);
        int npolchan = nchan*ncorr;

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
                in_list.size(),
                output_schema.size(),
                nsrc, ntime, na, npolchan);

    }
};

MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_JONES_MULTIPLY_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
