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
// ${cpp_header_file} and ${cuda_header_file} respectively.
// Concrete template instantiations of this class should be provided in
// ${cpp_source_file} and ${cuda_source_file} respectively
template <typename Device> class ${opname} {};

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
template <>
class ${opname}<CPUDevice> : public tensorflow::OpKernel
{
public:
    explicit ${opname}(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & input = context->input(0);

        // Allocate an output tensor
        tf::Tensor * output_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, input.shape(), &output_ptr));        
    }
};

${op_namespace_stop}
${project_namespace_stop}

#endif // #ifndef ${cpp_header_guard}
""")





# Template for the c++ source file (CPU)
CPP_SOURCE_TEMPLATE = string.Template(
"""#include "${cpp_header_file}"

${project_namespace_start}
${op_namespace_start}

REGISTER_OP("${opname}")
    .Input("in: float")
    .Output("out: float");

REGISTER_KERNEL_BUILDER(
    Name("${opname}")
    .Device(tensorflow::DEVICE_CPU),
    ${opname}<CPUDevice>);

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

// CUDA kernel outline
__global__ void ${kernel_name}(const float * input, float * output, int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= N)
        { return; }

    output[i] = 0.0f;
}

// Specialise the ${opname} op for GPUs
template <>
class ${opname}<GPUDevice> : public tensorflow::OpKernel
{
public:
    explicit ${opname}(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & input = context->input(0);

        int N = input.dim_size(0);

        // Allocate an output tensor
        tf::Tensor * output_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, input.shape(), &output_ptr));        

        // One block of 1024 threads
        dim3 block(1024, 0, 0);
        dim3 grid(1, 0, 0);
        
        const auto & stream = context->eigen_device<GPUDevice>().stream();

        ${kernel_name}<<<grid, block, 0, stream>>>(input.flat<float>().data(),
            output_ptr->flat<float>().data(), N);
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

REGISTER_KERNEL_BUILDER(
    Name("${opname}")
    .Device(tensorflow::DEVICE_GPU),
    ${opname}<GPUDevice>);

${op_namespace_stop}
${project_namespace_stop}

#endif // #if GOOGLE_CUDA
""")





# Template for the python test code
PYTHON_SOURCE_TEMPLATE = string.Template(
"""import numpy as np
import tensorflow as tf

rime = tf.load_op_library('${library}')

np_array = np.random.random(size=1000).astype(np.float32)
tf_array = tf.Variable(np_array)

result = rime.${snake_case}(tf_array)

with tf.Session() as S:
    S.run(tf.initialize_all_variables())
    res = S.run(result)
""")