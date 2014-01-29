import numpy as np
import pycuda
import pycuda.curandom
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

class RimeJonesReduce(Node):
    def __init__(self):
        super(RimeJonesReduce, self).__init__()
    def initialise(self, shared_data):
        self.mod = SourceModule("""
extern __shared__ double sres[];

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int * address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
                __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Prepare keys for segmented reduction
__global__ void seg_reduce_keyrange_per_block(int * keys, int* ranges,
                                              int n, int blockdim) {
    int i = threadIdx.x;
    if(i < blockdim) {
        ranges[2*i] = keys[i*blockdim];
        ranges[2*i+1] = keys[(i+1)*blockdim - 1];
    }
}

//template<class Method>
__global__
void segmented_reduction_kernel(double * values, int n_values, int* keys, int n_keys,
    int * keyranges,  double * result)
{
    __shared__ int skeys[256];
    __shared__ int svalues[256];
    int minkey = keyranges[2*blockIdx.x];
    int keydiff = keyranges[2*blockIdx.x + 1] - minkey;

    int thread = threadIdx.x;
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    // load keys and values
    svalues[thread] = values[index];
    skeys[thread] = keys[index];

    // result with the proper length
    if(thread <= keydiff)
        sres[thread] = 0;

    __syncthreads();
    for(int i = 1; i < blockDim.x; i *= 2) {
        if(thread % (2*i) == 0) {
            int w0 = skeys[thread];
            int w1 = skeys[thread + i];
            if(w0 != w1) {
                sres[w1 - minkey] += svalues[thread + i];
            }
            else {
                svalues[thread] += svalues[thread + i];
            }
        }
        __syncthreads();
    }
    // atomicAdd is fine here, as there are only few of those ops per
    // thread
    if(thread <= keydiff)
        atomicAdd(&result[minkey+thread], sres[thread]);
    __syncthreads();
    if(thread == 0)
        atomicAdd(&result[skeys[0]],svalues[0]);
}
""")
        self.seg_reduce_prep = self.mod.get_function('seg_reduce_keyrange_per_block')
        self.seg_reduce_kernel = self.mod.get_function('segmented_reduction_kernel')
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        N = 1024*1024
        keys = np.int32(np.random.rand(N)*N)
        n_threads = 256
        n_blocks = (N + n_threads - 1) / n_threads
        block, grid = (n_threads,1,1), (n_blocks,1,1)

        keys_gpu = gpuarray.to_gpu_async(keys, stream=shared_data.stream[0])
        key_ranges_gpu = gpuarray.empty([2*n_blocks],dtype=np.int32)
        a = pycuda.curandom.rand(N, dtype=np.float64)
        result_gpu = gpuarray.empty(N, dtype=np.float64)

        self.seg_reduce_prep(keys_gpu, key_ranges_gpu,
            np.int32(N), np.int32(1024),
            block=(1024,1,1), grid=(1,1,1),
            stream=shared_data.stream[0])

        self.seg_reduce_kernel(a, np.int32(N),
            keys_gpu, key_ranges_gpu, result_gpu,
            block=block, grid=grid,
            stream=shared_data.stream[0])

#        print result_gpu.get_async(stream=shared_data.stream[0])

    def post_execution(self, shared_data):
        pass