from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype

self.mod = SourceModule(KERNEL_TEMPLATE % {
    # Huge assumption here. The handle sitting in
    # the stream object is a CUStream type.
    # (Check the stream class in src/cpp/cuda.hpp).
    # mgpu::CreateCudaDeviceAttachStream in KERNEL_TEMPLATE
    # wants a cudaStream_t. However, according to the following
    #  http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__DRIVER.html
    # 'The types CUstream and cudaStream_t are identical and may be used interchangeably.'
    'stream_handle' : solver.stream[0].handle,
    'block_size' : 256,
    'warp_size' : 32,
    'value_type' : dtype_to_ctype(np.float64),
    'index_type' : dtype_to_ctype(np.int32)
},
no_extern_c=1)

KERNEL_TEMPLATE = """
#include <pycuda-helpers.hpp>

//#define BLOCK_SIZE %(block_size)d
//#define WARP_SIZE %(warp_size)d

//typedef %(value_type)s value_type;
//typedef %(index_type)s index_type;

__global__ void seg_reduce_sum()
{
    
}

"""