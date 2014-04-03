import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

class RimeMultiply(Node):
    def __init__(self):
        super(RimeMultiply, self).__init__()
        self.mod = SourceModule("""
#include <pycuda-complex.hpp>
#include \"math_constants.h\"

// Shared memory pointers used by the kernels.
extern __shared__ float smem_f[];
extern __shared__ double2 smem_d2[];

// Based on OSKAR cuda code
__device__ void
complex_multiply_double2(const double2 * lhs, const double2 * rhs, double2 * result)
{
    result->x = lhs->x*rhs->x;
    result->y = lhs->x*rhs->y;
    result->x -= lhs->y*rhs->y; /* RE*RE - IM*IM */
    result->y += lhs->y*rhs->x; /* RE*IM + IM*RE */
}

__device__ void
complex_multiply_add_double2(const double2 * lhs, const double2 * rhs, double2 * result)
{
    result->x += lhs->x*rhs->x;
    result->y += lhs->x*rhs->y;
    result->x -= lhs->y*rhs->y; /* RE*RE - IM*IM */
    result->y += lhs->y*rhs->x; /* RE*IM + IM*RE */
}

__global__
//__launch_bounds__( 256, 2 )
void rime_jones_multiply(
    double2 * lhs,
    double2 * rhs,
    double2 * out_jones,
    int njones)
{
    double2 * result = smem_d2;

    int i = (blockIdx.x*blockDim.x + threadIdx.x);

    if(i >= njones)
        { return; }

    const double2 a00 = lhs[i]; i += njones;
    const double2 a01 = lhs[i]; i += njones;
    const double2 a10 = lhs[i]; i += njones;
    const double2 a11 = lhs[i];

    i = (blockIdx.x*blockDim.x + threadIdx.x);

    const double2 b00 = rhs[i]; i += njones;
    const double2 b01 = rhs[i]; i += njones;
    const double2 b10 = rhs[i]; i += njones;
    const double2 b11 = rhs[i];

    complex_multiply_double2(&a00,&b00,&result[threadIdx.x]);
    complex_multiply_add_double2(&a01,&b10,&result[threadIdx.x]);
    i = blockIdx.x*blockDim.x + threadIdx.x;
    out_jones[i] = result[threadIdx.x];

    complex_multiply_double2(&a00,&b01,&result[threadIdx.x]);
    complex_multiply_add_double2(&a01,&b11,&result[threadIdx.x]);
    i += njones;
    out_jones[i] = result[threadIdx.x];

    complex_multiply_double2(&a10,&b00,&result[threadIdx.x]);
    complex_multiply_add_double2(&a11,&b10,&result[threadIdx.x]);
    i += njones;
    out_jones[i] = result[threadIdx.x];

    complex_multiply_double2(&a10,&b01,&result[threadIdx.x]);
    complex_multiply_add_double2(&a11,&b11,&result[threadIdx.x]);
    i += njones;
    out_jones[i] = result[threadIdx.x];
}
""",
options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_multiply')

    def initialise(self, shared_data):
        pass
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        njones = sd.nbl*sd.nchan*sd.ntime*sd.nsrc
        jones_per_block = 256 if njones > 256 else njones
        jones_blocks = (njones + jones_per_block - 1) / jones_per_block

        return {
            'block'  : (jones_per_block,1,1), \
            'grid'   : (jones_blocks,1,1), \
            'shared' : 1*jones_per_block*np.dtype(np.complex128).itemsize }

    def execute(self, shared_data):
        sd = shared_data

        # Output jones matrix
        njones = sd.nbl*sd.nchan*sd.ntime*sd.nsrc
        jsize = np.product(sd.jones_shape) # Number of complex  numbers
        jones_rhs = (np.random.random(jsize) + \
             1j*np.random.random(jsize)) \
            .astype(np.complex128).reshape(sd.jones_shape)

        jones_lhs_gpu = sd.jones_gpu
        jones_rhs_gpu = gpuarray.to_gpu(jones_rhs)
        jones_output_gpu = gpuarray.empty(shape=sd.jones_shape, dtype=np.complex128)

        self.kernel(jones_lhs_gpu, jones_rhs_gpu, jones_output_gpu, \
            np.int32(njones), **get_kernel_params())
            
        sd.jones_gpu = jones_output_gpu

    def post_execution(self, shared_data):
        pass