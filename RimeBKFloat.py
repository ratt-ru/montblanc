import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

FLOAT_KERNEL = """
#include \"math_constants.h\"

extern __shared__ float smem_d[];

__global__
void rime_jones_BK_float(
    float * UVW,
    float * LM,
    float * brightness,
    float * wavelength,
    float2 * jones,
    float refwave,
    int nbl, int nchan, int ntime, int nsrc)
{
    // Our data space is a 4D matrix of BL x SRC x CHAN x TIME

    // Baseline, Source, Channel and Time indices
    int SRC = blockIdx.x*blockDim.x + threadIdx.x;
    int CHAN = (blockIdx.y*blockDim.y + threadIdx.y) / ntime;
    int TIME = (blockIdx.y*blockDim.y + threadIdx.y) % ntime;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= nbl || SRC >= nsrc || CHAN >= nchan || TIME >= ntime)
        return;

    /* Cache input and output data from global memory. */
    float * u = smem_d;
    float * v = &u[blockDim.z];
    float * w = &v[blockDim.z];
    float * l = &w[blockDim.z];
    float * m = &l[blockDim.x];
    float * a = &m[blockDim.x];
    float * fI = &a[blockDim.x];
    float * fV = &fI[blockDim.x];
    float * fU = &fV[blockDim.x];
    float * fQ = &fU[blockDim.x];
    float * wave = &fQ[blockDim.x];

    // Index
    int i;

    if(threadIdx.x == 0)
    {
        // UVW is a 3 x nbl x ntime matrix
        i = BL*ntime + TIME; u[threadIdx.z] = UVW[i];
        i += nbl*ntime;      v[threadIdx.z] = UVW[i];
        i += nbl*ntime;      w[threadIdx.z] = UVW[i];
    }

    if(threadIdx.z == 0)
    {
		// LM and brightness are 2 x nsrc and 5 x nsrc matrices
        i = SRC;   l[threadIdx.x] = LM[i];
        i += nsrc; m[threadIdx.x] = LM[i];

        i = SRC;   fI[threadIdx.x] = brightness[i];
        i += nsrc; fQ[threadIdx.x] = brightness[i];
        i += nsrc; fU[threadIdx.x] = brightness[i];
        i += nsrc; fV[threadIdx.x] = brightness[i];
        i += nsrc; a[threadIdx.x] = brightness[i];
    }

    if(threadIdx.y == 0)
    {
        i = CHAN; wave[threadIdx.y] = wavelength[i];
    }

    __syncthreads();

    // Calculate the n term first
    // n = sqrt(1.0 - l*l - m*m) - 1.0
    float phase = 1.0 - l[threadIdx.x]*l[threadIdx.x];
    phase -= m[threadIdx.x]*m[threadIdx.x];
    phase = sqrt(phase) - 1.0;
    // TODO: remove this superfluous variable
    // It only exists for debugging purposes
    // float n = phase;

    // u*l + v*m + w*n, in the wrong order :)
    phase *= w[threadIdx.z];                  // w*n
    phase += v[threadIdx.z]*m[threadIdx.x];   // v*m
    phase += u[threadIdx.z]*l[threadIdx.x];   // u*l

    // Multiply by 2*pi/wave[threadIdx.y]
    phase *= (2. * CUDART_PI);
    phase /= wave[threadIdx.y];

    // Calculate the complex exponential from the phase
    float real, imag;
    __sincosf(phase, &imag, &real);

    // Multiply by the wavelength to the power of alpha
    phase = __powf(refwave/wave[threadIdx.y], a[threadIdx.x]);
    real *= phase; imag *= phase;

#if 0
    // Index into the jones array
    i = (BL*nsrc + SRC)*4;
    // Coalesced store of the computation
    jones[i+0]=make_float2(l[threadIdx.x],u[threadIdx.z]);
    jones[i+1]=make_float2(m[threadIdx.x],v[threadIdx.z]);
    jones[i+2]=make_float2(n,w[threadIdx.z]);
    jones[i+3]=result;
#endif


#if 1
    // Index into the jones matrices
    i = BL*nchan*ntime*nsrc + CHAN*ntime*nsrc + TIME*nsrc + SRC;

    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = fI+fQ, b=0.0, c=real, d = imag
    jones[i]=make_float2(
        (fI[threadIdx.x]+fQ[threadIdx.x])*real - 0.0*imag,
        (fI[threadIdx.x]+fQ[threadIdx.x])*imag + 0.0*real);

    // a=fU, b=fV, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_float2(
        fU[threadIdx.x]*real - fV[threadIdx.x]*imag,
        fU[threadIdx.x]*imag + fV[threadIdx.x]*real);

    // a=fU, b=-fV, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_float2(
        fU[threadIdx.x]*real - -fV[threadIdx.x]*imag,
        fU[threadIdx.x]*imag + -fV[threadIdx.x]*real);

    // a=fI-fQ, b=0.0, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_float2(
        (fI[threadIdx.x]-fQ[threadIdx.x])*real - 0.0*imag,
        (fI[threadIdx.x]-fQ[threadIdx.x])*imag + 0.0*real);
#endif
}
"""

class RimeBKFloat(Node):
    def __init__(self):
        super(RimeBKFloat, self).__init__()
        self.mod = SourceModule(FLOAT_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_BK_float')
    def initialise(self, shared_data):
        pass
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        baselines_per_block = 8 if sd.nbl > 8 else sd.nbl
        srcs_per_block = 32 if sd.nsrc > 32 else sd.nsrc
        time_chans_per_block = 1

        baseline_blocks = (sd.nbl + baselines_per_block - 1) / baselines_per_block
        src_blocks = (sd.nsrc + srcs_per_block - 1) / srcs_per_block
        time_chan_blocks = sd.ntime*sd.nchan

        return {
            'block' : (srcs_per_block,1,baselines_per_block), \
            'grid'  : (src_blocks,time_chan_blocks,baseline_blocks), \
            'shared' : (3*baselines_per_block + \
                        7*srcs_per_block + \
                        1*time_chans_per_block)*\
                            np.dtype(sd.ft).itemsize }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu,
            sd.wavelength_gpu,  sd.jones_gpu, sd.refwave,
            np.int32(sd.nbl), np.int32(sd.nchan), np.int32(sd.ntime), np.int32(sd.nsrc),
            **self.get_kernel_params(sd))

    def post_execution(self, shared_data):
        pass
