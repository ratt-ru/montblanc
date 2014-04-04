import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

FLOAT_KERNEL = """
#include \"math_constants.h\"

extern __shared__ float smem_f[];

// Based on OSKAR's implementation of the RIME K term.
// Baseline on the x dimension, source on the y dimension
__global__
void rime_jones_EBK_sum_float(
    float * UVW,
    float * LM,
    float * brightness,
    float * wavelength,
    float * point_error,
    float2 * jones,
    int nsrc, int nbl, int nchan, int ntime, int na)
{
    // Our data space is a 4D matrix of BL x SRC x CHAN x TIME

    #define REFWAVE 1e6
    #define COS3_CONST 65*1e-9

    // Baseline, Channel and Time indices
    int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= nbl || CHAN >= nchan || TIME >= ntime)
        return;

    // Calculates the antenna pairs from the baseline!
    int ANT1 = int(floor((sqrtf(1+8*BL)-1)/2));
    int ANT2 = BL - (ANT1*ANT1+ANT1)/2;
    ANT1 += 1;

    float * l = smem_f;       // l is at beginning of shared mem
    float * m = &l[nsrc];     // m is at the end of l

    float * fI = &m[nsrc];    // fI is at the end of m
    float * fQ = &fI[nsrc];   // fQ is at the end of fI
    float * fV = &fQ[nsrc];   // fV is at the end of fQ
    float * fU = &fV[nsrc];   // fU is at the end of fV

    float * wave = &fU[nsrc]; // wave is at the end of fU

    float * ld_p = &wave[blockDim.x]; // Number of CHANS
    float * md_p = &ld_p[blockDim.z*blockDim.y]; // BL*TIME
    float * ld_q = &md_p[blockDim.z*blockDim.y]; // BL*TIME
    float * md_q = &ld_q[blockDim.z*blockDim.y]; // BL*TIME

    int i;

    // Keep our x (CHAN) dimension constant.
    // Then we can load in stuff required by our changing
    // y (TIME) and z (BL) dimensions. Specifically, pointing errors
    if(threadIdx.x == 0) 
    {
        int j = threadIdx.z*blockDim.y+threadIdx.y;
        i = ANT1*ntime + TIME; ld_p[j] = point_error[i];
        i += na*ntime;         md_p[j] = point_error[i];
        i = ANT2*ntime + TIME; ld_q[j] = point_error[i];
        i += na*ntime;         md_q[j] = point_error[i];
    }

    // Keep our y (TIME) and z (BL) dimensions constant
    // Then we can load in stuff required by our changing
    // x (CHAN) dimension. Specifically, wavelengths
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        i = CHAN; wave[threadIdx.x] = wavelength[i];
    }

    // Shouldn't need this because of __synchthreads() in loop
    // __syncthreads();

    for(int SRC=0; SRC<nsrc; ++SRC)
    {
        // Load point source data into shared memory using one thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC;   l[SRC] = LM[i]; fI[SRC] = brightness[i];
            i += nsrc; m[SRC] = LM[i]; fQ[SRC] = brightness[i];
            i += nsrc; fU[SRC] = brightness[i];
            i += nsrc; fV[SRC] = brightness[i];
        }

        __syncthreads();

        float phase = 1.0 - l[SRC]*l[SRC];
        phase -= m[SRC]*m[SRC];
        phase = sqrt(phase) - 1.0;

        // UVW is 3 x nbl x ntime matrix
        // u*l + v*m + w*n, in the wrong order :)
        i = BL*ntime + TIME + 2*nbl*ntime; phase *= UVW[i]; // w*n
        i -= nbl*ntime;             phase += UVW[i]*m[SRC]; // v*m
        i -= nbl*ntime;             phase += UVW[i]*l[SRC]; // u*l

        // Multiply by 2*pi/wave[threadIdx.x]
        phase *= (2. * CUDART_PI);
        phase /= wave[threadIdx.x];

        // Calculate the complex exponential from the phase
        float real, imag;
        __sincosf(phase, &imag, &real);

        // Multiply by the wavelength to the power of alpha
        i = SRC+nsrc*4; phase = __powf(REFWAVE/wave[threadIdx.x], brightness[i]);
        real *= phase; imag *= phase;        

        // Index into the jones matrices
        i = (BL*nchan*ntime*nsrc + CHAN*ntime*nsrc + TIME*nsrc + SRC);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // a = fI+fQ, b=0.0, c=real, d = imag
        jones[i]=make_float2(
            (fI[SRC]+fQ[SRC])*real - 0.0*imag,
            (fI[SRC]+fQ[SRC])*imag + 0.0*real);

        // a=fU, b=fV, c=real, d = imag 
        i += nbl*nsrc*nchan*ntime;
        jones[i]=make_float2(
            fU[SRC]*real - fV[SRC]*imag,
            fU[SRC]*imag + fV[SRC]*real);

        // a=fU, b=-fV, c=real, d = imag 
        i += nbl*nsrc*nchan*ntime;
        jones[i]=make_float2(
            fU[SRC]*real - -fV[SRC]*imag,
            fU[SRC]*imag + -fV[SRC]*real);

        // a=fI-fQ, b=0.0, c=real, d = imag 
        i += nbl*nsrc*nchan*ntime;
        jones[i]=make_float2(
            (fI[SRC]-fQ[SRC])*real - 0.0*imag,
            (fI[SRC]-fQ[SRC])*imag + 0.0*real);
    }

    #undef REFWAVE
}
"""

class RimeEBKSumFloat(Node):
    def __init__(self):
        super(RimeEBKSumFloat, self).__init__()
        self.mod = SourceModule(FLOAT_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_EBK_sum_float')

    def initialise(self, shared_data):
		pass	

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        chans_per_block = 4 if sd.nchan > 2 else sd.nchan
        times_per_block = 4 if sd.ntime > 2 else sd.ntime
        baselines_per_block = 16 if sd.nbl > 16 else sd.nbl

        chan_blocks = (sd.nchan + chans_per_block - 1) / chans_per_block
        time_blocks = (sd.ntime + times_per_block - 1) / times_per_block
        baseline_blocks = (sd.nbl + baselines_per_block - 1)/ baselines_per_block

        return {
            'block' : (chans_per_block,times_per_block,baselines_per_block),
            'grid'  : (chan_blocks,time_blocks,baseline_blocks),
            'shared' : (1*chans_per_block +
                6*sd.nsrc +
                4*baselines_per_block*times_per_block)*
                    np.dtype(sd.ft).itemsize }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu,
            sd.wavelength_gpu, sd.point_errors_gpu, sd.jones_gpu,
            np.int32(sd.nsrc), np.int32(sd.nbl),
            np.int32(sd.nchan), np.int32(sd.ntime), np.int32(sd.na),
            **self.get_kernel_params(sd))       

    def post_execution(self, shared_data):
        pass