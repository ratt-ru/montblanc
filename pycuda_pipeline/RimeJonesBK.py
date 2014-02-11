import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

class RimeJonesBK(Node):
    def __init__(self):
        super(RimeJonesBK, self).__init__()
    def initialise(self, shared_data):
        self.mod = SourceModule("""
#include <pycuda-complex.hpp>
#include \"math_constants.h\"

extern __shared__ double smem_d[];

__global__
void rime_jones_BK(
    double * UVW,
    double * LMA,
    double * sky,
    double wavelength,
    pycuda::complex<double> * jones,
    int ndir, int na, int nbl)
{
    // Our data space a 2D matrix of BL x DDE

    // Baseline
    const int BL = blockIdx.x*blockDim.x + threadIdx.x;
    // Direction Dependent Effect
    const int DDE = blockIdx.y*blockDim.y + threadIdx.y;

    if(BL >= nbl || DDE >= ndir)
        return;

    // Index into the jones array
    const int i = (BL*ndir + DDE)*4; 

    /* Cache input and output data from global memory. */
    double * u = smem_d;
    double * v = &u[blockDim.x];
    double * w = &v[blockDim.x];
    double * l = &w[blockDim.x];
    double * m = &l[blockDim.y];
    double * a = &m[blockDim.y];

    if (BL < nbl && threadIdx.y == 0)
    {
        u[threadIdx.x] = UVW[BL+0*nbl];
        v[threadIdx.x] = UVW[BL+1*nbl];
        w[threadIdx.x] = UVW[BL+2*nbl];
    }

    if (DDE < ndir && threadIdx.x == 0)
    {
        l[threadIdx.y] = LMA[DDE+0*ndir];
        m[threadIdx.y] = LMA[DDE+1*ndir];
        a[threadIdx.y] = LMA[DDE+2*ndir];
    }

    __syncthreads();

    // Calculate the n term first
    // n = sqrt(1.0 - l*l - m*m) - 1.0
    double phase = 1.0 - l[threadIdx.y]*l[threadIdx.y];
    phase -= m[threadIdx.y]*m[threadIdx.y];
    phase = sqrt(phase) - 1.0; 

    // u*l + v*m + w*n, in the wrong order :)
    phase *= w[threadIdx.x];                  // w*n
    phase += v[threadIdx.x]*m[threadIdx.y];   // v*m
    phase += u[threadIdx.x]*l[threadIdx.y];   // u*l

    // Multiply by 2*pi/wavelength
    phase *= (2. * CUDART_PI);
    phase /= wavelength;

    // Calculate the complex exponential from the phase
//    pycuda::complex<double> result = pycuda::exp(pycuda::complex<double>(0,phase));
    // Uses two registers less than the above approach.
    pycuda::complex<double> result;
    sincos(phase, &result._M_im, &result._M_re);

    // Multiply by the wavelength to the power of alpha
    result *= pow(1e6/wavelength,a[threadIdx.y]);

#if 0
    // Coalesced store of the computation
    jones[i+0]=pycuda::complex<double>(l[threadIdx.y],u[threadIdx.x]);
    jones[i+1]=pycuda::complex<double>(m[threadIdx.y],v[threadIdx.x]);
    jones[i+2]=pycuda::complex<double>(0.0,w[threadIdx.x]);
    jones[i+3]=pycuda::complex<double>(i,0);
#endif


#if 1
    const double fI = sky[DDE+0*ndir];
    const double fQ = sky[DDE+1*ndir];
    const double fU = sky[DDE+2*ndir];
    const double fV = sky[DDE+3*ndir];
    // TODO, this is uncoalesced
    jones[i+0]=pycuda::complex<double>(fI+fQ,0.)*result;
    jones[i+1]=pycuda::complex<double>(fU,fV)*result;
    jones[i+2]=pycuda::complex<double>(fU,-fV)*result;
    jones[i+3]=pycuda::complex<double>(fU-fQ,0.)*result;

#endif

    #undef SLICE_STRIDE
    #undef ROW_STRIDE

    #undef CUDA_XDIM
    #undef CUDA_YDIM
    #undef CUDA_ZDIM
}
""",
options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_BK')

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        sd = shared_data

        freqs=np.float64(np.linspace(1e6,2e6,sd.nchan))
        wavelength = 3e8/freqs

        baselines_per_block = 8 if sd.nbl > 8 else sd.nbl
        ddes_per_block = 128 if sd.ndir > 128 else sd.ndir

        baseline_blocks = (sd.nbl + baselines_per_block - 1) / baselines_per_block
        dde_blocks = (sd.ndir + ddes_per_block - 1) / ddes_per_block

        block=(baselines_per_block,ddes_per_block,1)
        grid=(baseline_blocks,dde_blocks,1)

        print 'block', block, 'grid', grid

        foreground_stream,background_stream = sd.stream[0], sd.stream[1]
        chan = 0

        self.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
            wavelength[chan],  sd.jones_gpu,
            np.int32(sd.ndir), np.int32(sd.na), np.int32(sd.nbl),
            stream=foreground_stream,
            block=block,
            grid=grid,
            shared=3*(baselines_per_block+ddes_per_block)
				*np.dtype(np.float64).itemsize)

        #print sd.jones_gpu.get_async(stream=foreground_stream)


    def post_execution(self, shared_data):
        pass