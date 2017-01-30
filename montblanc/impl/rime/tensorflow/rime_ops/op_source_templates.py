import string

# Template for the main header file
MAIN_HEADER_TEMPLATE = string.Template(
"""#ifndef ${main_header_guard}
#define ${main_header_guard}

// ${project} namespace start and stop defines
#define ${project_namespace_start} namespace ${project} {
#define ${project_namespace_stop} }

// ${snake_case} namespace start and stop defines
#define ${op_namespace_start} namespace ${snake_case} {
#define ${op_namespace_stop} }

${project_namespace_start}
${op_namespace_start}

// General definition of the ${opname} op, which will be specialised for CPUs and GPUs in
// ${cpp_header_file} and ${cuda_header_file} respectively, as well as float types (FT).
// Concrete template instantiations of this class should be provided in
// ${cpp_source_file} and ${cuda_source_file} respectively
template <typename Device, typename FT> class ${opname} {};

${op_namespace_stop}
${project_namespace_stop}

#endif // #ifndef ${main_header_guard}
""")





# Template for the c++ header file (CPU)
CPP_HEADER_TEMPLATE = string.Template(
"""#ifndef ${cpp_header_guard}
#define ${cpp_header_guard}

#include "${main_header_file}"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

${project_namespace_start}
${op_namespace_start}

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the ${opname} op for CPUs
template <typename FT>
class ${opname}<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit ${opname}(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_input = context->input(0);

        // Allocate an output tensor
        tf::Tensor * output_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, in_input.shape(), &output_ptr));


        int N = in_input.dim_size(0);
        auto input = in_input.tensor<FT, 1>();
        auto output = output_ptr->tensor<FT, 1>();

        for(int i=0; i<N; ++i)
            { output(i) = input(i) + FT(1.0); }
    }
};

${op_namespace_stop}
${project_namespace_stop}

#endif // #ifndef ${cpp_header_guard}
""")





# Template for the c++ source file (CPU)
CPP_SOURCE_TEMPLATE = string.Template(
"""#include "${cpp_header_file}"

#include "tensorflow/core/framework/shape_inference.h"

${project_namespace_start}
${op_namespace_start}

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    ShapeHandle in = c->input(0);

    // Assert that in has 1 dimension
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in, 1, &input),
        "in shape must be [N, ] but is " + c->DebugString(in));

    // Assert that in has a certain size
    // TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in, 0), N, &d),
    //     "in shape must be [N, ] but is " + c->DebugString(in));

    // Infer the shape of the output tensor,
    // in this case, the same shape as our input tensor
    ShapeHandle out = c->MakeShape({
        c->Dim(in, 0)
    });

    // Set the shape of the first output
    c->set_output(0, out);

    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the ${opname} operator.
REGISTER_OP("${opname}")
    .Input("in: FT")
    .Output("out: FT")
    .Attr("FT: {double, float} = DT_FLOAT")
    .SetShapeFn(shape_function);

// Register a CPU kernel for ${opname} that handles floats
REGISTER_KERNEL_BUILDER(
    Name("${opname}")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    ${opname}<CPUDevice, float>);

// Register a CPU kernel for ${opname} that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("${opname}")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    ${opname}<CPUDevice, double>);


${op_namespace_stop}
${project_namespace_stop}
""")





# Template for the cuda header file (GPU)
CUDA_HEADER_TEMPLATE = string.Template(
"""#if GOOGLE_CUDA

#ifndef ${cuda_header_guard}
#define ${cuda_header_guard}

#include "${main_header_file}"

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

${project_namespace_start}
${op_namespace_start}

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for floats and doubles
template <typename FT> struct LaunchTraits {};

template <> struct LaunchTraits<float>
    { static constexpr int BLOCKDIMX = 1024; };

template <> struct LaunchTraits<double>
    { static constexpr int BLOCKDIMX = 1024; };

// CUDA kernel outline
template <typename FT>
__global__ void ${kernel_name}(const FT * input, FT * output, int N)
{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using LTr = LaunchTraits<FT>;
    __shared__ FT buffer[LTr::BLOCKDIMX];

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= N)
        { return; }

    // Load in our input and add one to it
    buffer[threadIdx.x] = input[i];
    buffer[threadIdx.x] += FT(1.0);

    // Write to the outpu
    output[i] = buffer[threadIdx.x];
}

// Specialise the ${opname} op for GPUs
template <typename FT>
class ${opname}<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit ${opname}(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_input = context->input(0);

        int N = in_input.dim_size(0);

        // Allocate an output tensor
        tf::Tensor * output_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, in_input.shape(), &output_ptr));

        using LTr = LaunchTraits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block(LTr::BLOCKDIMX);
        dim3 grid((N + block.x - 1)/block.x);

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Get pointers to flattened tensor data buffers
        auto const input = in_input.flat<FT>().data();
        auto output = output_ptr->flat<FT>().data();

        // Call the ${kernel_name} CUDA kernel
        ${kernel_name}<<<grid, block, 0, device.stream()>>>(
            input, output, N);
    }
};

${op_namespace_stop}
${project_namespace_stop}

#endif // #ifndef ${cuda_header_guard}

#endif // #if GOOGLE_CUDA
""")





# Template for the cuda source file (GPU)
CUDA_SOURCE_TEMPLATE = string.Template(
"""#if GOOGLE_CUDA

#include "${cuda_header_file}"

${project_namespace_start}
${op_namespace_start}

// Register a GPU kernel for ${opname} that handles floats
REGISTER_KERNEL_BUILDER(
    Name("${opname}")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    ${opname}<GPUDevice, float>);

// Register a GPU kernel for ${opname} that handles doubles
REGISTER_KERNEL_BUILDER(
    Name("${opname}")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    ${opname}<GPUDevice, double>);

${op_namespace_stop}
${project_namespace_stop}

#endif // #if GOOGLE_CUDA
""")





# Template for the python test code
PYTHON_SOURCE_TEMPLATE = string.Template(
"""import os

import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
from montblanc.impl.rime.tensorflow import load_tf_lib
rime = load_tf_lib()

# Register the shape function for the operation
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
ops.RegisterShape("${opname}")(common_shapes.call_cpp_shape_fn)

# Create some input and wrap it in a tensorflow Variable
np_array = np.random.random(size=512*1024).astype(np.float32)
tf_array = tf.Variable(np_array)

# Pin the compute to the CPU
with tf.device('/cpu:0'):
    expr_cpu = ${module}.${snake_case}(tf_array)

# Pin the compute to the GPU
with tf.device('/gpu:0'):
    expr_gpu = ${module}.${snake_case}(tf_array)

init_op = tf.global_variables_initializer()

with tf.Session() as S:
    S.run(init_op)

    # Run our expressions on CPU and GPU
    result_cpu = S.run(expr_cpu)
    result_gpu = S.run(expr_gpu)

    # Check that 1.0 has been added to the input
    # and that CPU and GPU results agree
    assert np.allclose(result_cpu, np_array + 1.0)
    assert np.allclose(result_cpu, result_gpu)

""")