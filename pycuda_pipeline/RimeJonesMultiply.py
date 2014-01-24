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

__global__
//__launch_bounds__( 256, 2 )
void rime_jones_multiply(
    pycuda::complex<double> * lhs,
    pycuda::complex<double> * rhs,
    pycuda::complex<double> * out_jones,
	int njones)
{
    const int i = (blockIdx.x*blockDim.x + threadIdx.x);

    const pycuda::complex<double> a00 = lhs[i+0*njones];
    const pycuda::complex<double> a01 = lhs[i+1*njones];
    const pycuda::complex<double> a10 = lhs[i+2*njones];
    const pycuda::complex<double> a11 = lhs[i+3*njones];

    const pycuda::complex<double> b00 = rhs[i+0*njones];
    const pycuda::complex<double> b01 = rhs[i+1*njones];
    const pycuda::complex<double> b10 = rhs[i+2*njones];
    const pycuda::complex<double> b11 = rhs[i+3*njones];

    out_jones[i+0*njones]=a00*b00+a01*b10;
    out_jones[i+1*njones]=a00*b01+a01*b11;
    out_jones[i+2*njones]=a10*b00+a11*b10;
    out_jones[i+3*njones]=a10*b01+a11*b11;
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
            stream=foreground_stream, block=block, grid=grid)

#        print jones_gpu.get_async(stream=foreground_stream)

    def post_execution(self, shared_data):
        pass