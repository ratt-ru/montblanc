import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

FLOAT_KERNEL = """
#include \"math_constants.h\"

extern __shared__ float smem_d[];

// Based on OSKAR's implementation of the RIME K term.
// Baseline on the x dimension, source on the y dimension
__global__
void rime_jones_BK_float(
    float * UVW,
    float * LMA,
    float * sky,
    float * wavelength,
    float2 * jones,
    int nsrc, int nbl,
    int nchan, int ntime)
{
    // Our data space is a 4D matrix of BL x SRC x CHAN x TIME

    #define REFWAVE 1e6

    // Baseline, Source, Channel and Time indices
    int BL = blockIdx.x*blockDim.x + threadIdx.x;
    int SRC = blockIdx.y*blockDim.y + threadIdx.y;
    int CHAN = (blockIdx.z*blockDim.z + threadIdx.z) / ntime;
    int TIME = (blockIdx.z*blockDim.z + threadIdx.z) % ntime;

    if(BL >= nbl || SRC >= nsrc || CHAN >= nchan || TIME >= ntime)
        return;

    /* Cache input and output data from global memory. */
    float * u = smem_d;
    float * v = &u[blockDim.x];
    float * w = &v[blockDim.x];
    float * l = &w[blockDim.x];
    float * m = &l[blockDim.y];
    float * a = &m[blockDim.y];
    float * fI = &a[blockDim.y];
    float * fV = &fI[blockDim.y];
    float * fU = &fV[blockDim.y];
    float * fQ = &fU[blockDim.y];
    float * wave = &fQ[blockDim.y];

    // Index
    int i;

    if(threadIdx.y == 0)
    {
        i = BL;   u[threadIdx.x] = UVW[i];
        i += nbl; v[threadIdx.x] = UVW[i];
        i += nbl; w[threadIdx.x] = UVW[i];
    }

    if(threadIdx.x == 0)
    {
        i = SRC;   l[threadIdx.y] = LMA[i];
        i += nsrc; m[threadIdx.y] = LMA[i];
        i += nsrc; a[threadIdx.y] = LMA[i];

        i = SRC;   fI[threadIdx.y] = sky[i];
        i += nsrc; fU[threadIdx.y] = sky[i];
        i += nsrc; fV[threadIdx.y] = sky[i];
        i += nsrc; fQ[threadIdx.y] = sky[i];
    }

    if(threadIdx.z == 0)
    {
        i = CHAN; wave[threadIdx.z] = wavelength[i];
    }

    __syncthreads();

    // Calculate the n term first
    // n = sqrt(1.0 - l*l - m*m) - 1.0
    float phase = 1.0 - l[threadIdx.y]*l[threadIdx.y];
    phase -= m[threadIdx.y]*m[threadIdx.y];
    phase = sqrt(phase) - 1.0;
    // TODO: remove this superfluous variable
    // It only exists for debugging purposes
    // float n = phase;

    // u*l + v*m + w*n, in the wrong order :)
    phase *= w[threadIdx.x];                  // w*n
    phase += v[threadIdx.x]*m[threadIdx.y];   // v*m
    phase += u[threadIdx.x]*l[threadIdx.y];   // u*l

    // Multiply by 2*pi/wave[threadIdx.z]
    phase *= (2. * CUDART_PI);
    phase /= wave[threadIdx.z];

    // Calculate the complex exponential from the phase
    float real, imag;
    sincosf(phase, &imag, &real);

    // Multiply by the wavelength to the power of alpha
    phase = powf(REFWAVE/wave[threadIdx.z], a[threadIdx.y]);
    real *= phase; imag *= phase;

#if 0
    // Index into the jones array
    i = (BL*nsrc + SRC)*4;
    // Coalesced store of the computation
    jones[i+0]=make_float2(l[threadIdx.y],u[threadIdx.x]);
    jones[i+1]=make_float2(m[threadIdx.y],v[threadIdx.x]);
    jones[i+2]=make_float2(n,w[threadIdx.x]);
    jones[i+3]=result;
#endif


#if 1
    // Index into the jones matrices
    i = (BL*nchan*ntime*nsrc + CHAN*ntime*nsrc + TIME*nsrc + SRC);

    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = fI+fQ, b=0.0, c=real, d = imag
    jones[i]=make_float2(
        (fI[threadIdx.y]+fQ[threadIdx.y])*real - 0.0*imag,
        (fI[threadIdx.y]+fQ[threadIdx.y])*imag + 0.0*real);

    // a=fU, b=fV, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_float2(
        fU[threadIdx.y]*real - fV[threadIdx.y]*imag,
        fU[threadIdx.y]*imag + fV[threadIdx.y]*real);

    // a=fU, b=-fV, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_float2(
        fU[threadIdx.y]*real - -fV[threadIdx.y]*imag,
        fU[threadIdx.y]*imag + -fV[threadIdx.y]*real);

    // a=fI-fQ, b=0.0, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_float2(
        (fI[threadIdx.y]-fQ[threadIdx.y])*real - 0.0*imag,
        (fI[threadIdx.y]-fQ[threadIdx.y])*imag + 0.0*real);
#endif

    #undef REFWAVE
}
"""

class RimeJonesBKFloat(Node):
    def __init__(self):
        super(RimeJonesBKFloat, self).__init__()
    def initialise(self, shared_data):
        self.mod = SourceModule(FLOAT_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_BK_float')

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        baselines_per_block = 8 if sd.nbl > 8 else sd.nbl
        srcs_per_block = 16 if sd.nsrc > 16 else sd.nsrc
        time_chans_per_block = 1

        baseline_blocks = (sd.nbl + baselines_per_block - 1) / baselines_per_block
        src_blocks = (sd.nsrc + srcs_per_block - 1) / srcs_per_block
        time_chan_blocks = sd.ntime*sd.nchan

        return {
            'block' : (baselines_per_block,srcs_per_block,1), \
            'grid'  : (baseline_blocks,src_blocks,time_chan_blocks), \
            'shared' : (3*baselines_per_block + \
                        7*srcs_per_block + \
                        1*time_chans_per_block)*\
                            np.dtype(np.float64).itemsize }

    def execute(self, shared_data):
        sd = shared_data
        params = self.get_kernel_params(sd)

        self.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
            sd.wavelength_gpu,  sd.jones_gpu,
            np.int32(sd.nsrc), np.int32(sd.nbl), **params)

    def post_execution(self, shared_data):
        pass