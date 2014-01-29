import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

class RimeJonesMultiply(Node):
    def __init__(self):
        super(RimeJonesMultiply, self).__init__()
    def initialise(self, shared_data):
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

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        ## Here I define my data, and my Jones matrices
        na=shared_data.na          # Number of antenna
        nbl=shared_data.nbl        # Number of baselines
        nchan=shared_data.nchan    # Number of channels
        ndir=shared_data.ndir      # Number of DDES

        # Output jones matrix
        jones_shape = (4,nbl,ndir)
        njones = nbl*ndir
        jsize = np.product(jones_shape) # Number of complex  numbers
        jones_lhs = (np.random.random(jsize) + 1j*np.random.random(jsize)).astype(np.complex128).reshape(jones_shape)
        jones_rhs = (np.random.random(jsize) + 1j*np.random.random(jsize)).astype(np.complex128).reshape(jones_shape)

        jones_per_block = 256 if njones > 256 else njones
        jones_blocks = (njones + jones_per_block - 1) / jones_per_block
        block, grid = (jones_per_block,1,1), (jones_blocks,1,1)

        print 'block', block, 'grid', grid

        foreground_stream,background_stream = shared_data.stream[0], shared_data.stream[1]

        jones_lhs_gpu = gpuarray.to_gpu_async(jones_lhs, stream=foreground_stream)
        jones_rhs_gpu = gpuarray.to_gpu_async(jones_rhs, stream=foreground_stream)
        jones_output_gpu = gpuarray.empty(shape=jones_shape, dtype=np.complex128)

        self.kernel(jones_lhs_gpu, jones_rhs_gpu, jones_output_gpu, np.int32(njones),
            stream=foreground_stream, block=block, grid=grid,
            shared=1*jones_per_block*np.dtype(np.complex128).itemsize)

        jones_output = jones_output_gpu.get_async(stream=foreground_stream)

        # Perform the calculation on the CPU
        jones_output_cpu = np.empty(shape=jones_shape, dtype=np.complex128)

        for baseline in range(nbl):
            for direction in range(ndir):
                jones_output_cpu[:,baseline,direction] = np.dot(
                    jones_lhs[:,baseline,direction].reshape(2,2),
                    jones_rhs[:,baseline,direction].reshape(2,2)).reshape(4)

        # Confirm similar results
        assert np.allclose(jones_output, jones_output_cpu)

    def post_execution(self, shared_data):
        pass