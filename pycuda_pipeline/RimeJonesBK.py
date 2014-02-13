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
#include \"math_constants.h\"

extern __shared__ double smem_d[];

// Based on OSKAR's implementation of the RIME K term.
// Baseline on the x dimension, source on the y dimension
__global__
void rime_jones_BK(
    double * UVW,
    double * LMA,
    double * sky,
    double wavelength,
    double2 * jones,
    int nsrc, int nbl)
{
    // Our data space is a 2D matrix of BL x SRC

    // Baseline
    int BL = blockIdx.x*blockDim.x + threadIdx.x;
    // Direction Dependent Effect
    int SRC = blockIdx.y*blockDim.y + threadIdx.y;

    if(BL >= nbl || SRC >= nsrc)
        return;

    /* Cache input and output data from global memory. */
    double * u = smem_d;
    double * v = &u[blockDim.x];
    double * w = &v[blockDim.x];
    double * l = &w[blockDim.x];
    double * m = &l[blockDim.y];
    double * a = &m[blockDim.y];

    // Index
    int i;

    if (BL < nbl && threadIdx.y == 0)
    {
        i = BL;   u[threadIdx.x] = UVW[i];
        i += nbl; v[threadIdx.x] = UVW[i];
        i += nbl; w[threadIdx.x] = UVW[i];
    }

    if (SRC < nsrc && threadIdx.x == 0)
    {
        i = SRC;   l[threadIdx.y] = LMA[i];
        i += nsrc; m[threadIdx.y] = LMA[i];
        i += nsrc; a[threadIdx.y] = LMA[i];
    }

    __syncthreads();

    // Calculate the n term first
    // n = sqrt(1.0 - l*l - m*m) - 1.0
    double phase = 1.0 - l[threadIdx.y]*l[threadIdx.y];
    phase -= m[threadIdx.y]*m[threadIdx.y];
    phase = sqrt(phase) - 1.0;
    // TODO: remove this superfluous variable
    // It only exists for debugging purposes
    double n = phase;

    // u*l + v*m + w*n, in the wrong order :)
    phase *= w[threadIdx.x];                  // w*n
    phase += v[threadIdx.x]*m[threadIdx.y];   // v*m
    phase += u[threadIdx.x]*l[threadIdx.y];   // u*l

    // sqrt(u*l + v*m + w*n)
    phase = sqrt(phase);

    // Multiply by 2*pi/wavelength
    phase *= (2. * CUDART_PI);
    phase /= wavelength;

    // Calculate the complex exponential from the phase
    double2 result;
    sincos(phase, &result.y, &result.x);

    // Multiply by the wavelength to the power of alpha
    phase = pow(1e6/wavelength, a[threadIdx.y]);
    result.x *= phase;
    result.y *= phase;

#if 0
    // Index into the jones array
    i = (BL*nsrc + SRC)*4;
    // Coalesced store of the computation
    jones[i+0]=make_double2(l[threadIdx.y],u[threadIdx.x]);
    jones[i+1]=make_double2(m[threadIdx.y],v[threadIdx.x]);
    jones[i+2]=make_double2(n,w[threadIdx.x]);
    jones[i+3]=result;
#endif


#if 1
    i = (BL*nsrc + SRC)*4;

    double fI = sky[SRC+0*nsrc];
    double fQ = sky[SRC+1*nsrc];
    double fU = sky[SRC+2*nsrc];
    double fV = sky[SRC+3*nsrc];

    // TODO, this is *still* uncoalesced
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = fI+fQ, b=0.0, c=result.x, d = result.y
    jones[i+0]=make_double2(
        (fI+fQ)*result.x - 0.0*result.y,
        (fI+fQ)*result.y + 0.0*result.x);

    // a=fU, b=fV, c=result.x, d = result.y 
    jones[i+1]=make_double2(
        fU*result.x - fV*result.y,
        fU*result.y + fV*result.x);

    // a=fU, b=-fV, c=result.x, d = result.y 
    jones[i+2]=make_double2(
        fU*result.x - -fV*result.y,
        fU*result.y + -fV*result.x);

    // a=fU-fQ, b=0.0, c=result.x, d = result.y 
    jones[i+3]=make_double2(
        (fU-fQ)*result.x - 0.0*result.y,
        (fU-fQ)*result.y + 0.0*result.x);
#endif
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
        srcs_per_block = 128 if sd.nsrc > 128 else sd.nsrc

        baseline_blocks = (sd.nbl + baselines_per_block - 1) / baselines_per_block
        src_blocks = (sd.nsrc + srcs_per_block - 1) / srcs_per_block

        block=(baselines_per_block,srcs_per_block,1)
        grid=(baseline_blocks,src_blocks,1)

        chan = 0

        self.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
            wavelength[chan],  sd.jones_gpu,
            np.int32(sd.nsrc), np.int32(sd.nbl),
            block=block,
            grid=grid,
            shared=3*(baselines_per_block+srcs_per_block)
				*np.dtype(np.float64).itemsize)

    def post_execution(self, shared_data):
        pass