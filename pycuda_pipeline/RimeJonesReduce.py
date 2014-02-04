import numpy as np
import pycuda
import pycuda.curandom
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype

from node import *

import segreduce

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

class RimeJonesReduce(Node):
    def __init__(self):
        super(RimeJonesReduce, self).__init__()
    def initialise(self, shared_data):
        self.mod = SourceModule(KERNEL_TEMPLATE % {
                # Huge assumption here. The handle sitting in
                # the stream object is a CUStream type.
                # (Check the stream class in src/cpp/cuda.hpp).
                # mgpu::CreateCudaDeviceAttachStream in KERNEL_TEMPLATE
                # wants a cudaStream_t. However, according to the following
                #  http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__DRIVER.html
                # 'The types CUstream and cudaStream_t are identical and may be used interchangeably.'
                'stream_handle' : shared_data.stream[0].handle,
                'block_size' : 256,
                'warp_size' : 32,
                'value_type' : dtype_to_ctype(np.float64),
                'index_type' : dtype_to_ctype(np.int32)
            },
            no_extern_c=1)
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        nbl = shared_data.nbl
        ndir = shared_data.ndir

        jones_shape = shared_data.jones_shape

        njones = np.product(jones_shape)
        keys = np.empty(jones_shape[0:2])

        for i in range(jones_shape[0]):
            for bl in range(jones_shape[1]):
#                for dir in jones_shape[2]:
                keys[i,bl] = bl*4 + i

        n_threads = 256
        n_blocks = (njones + n_threads - 1) / n_threads
        block, grid = (n_threads,1,1), (n_blocks,1,1)

        print 'njones=', njones, 'stream=', shared_data.stream[0].handle

        keys_gpu = gpuarray.to_gpu_async(keys, stream=shared_data.stream[0])
        sums_gpu = gpuarray.empty(shape=jones_shape, dtype=shared_data.jones_gpu.dtype.type)

        segreduce.segmented_reduce_complex128_sum(
            shared_data.jones_gpu,
            keys_gpu,
            sums_gpu,
            device_id=0,
            stream=shared_data.stream[0])

    def post_execution(self, shared_data):
        pass