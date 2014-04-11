import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

FLOAT_KERNEL = """
#include \"math_constants.h\"

extern __shared__ float smem_f[];

__global__
void rime_jones_EBK_float(
    float * UVW,
    float * LM,
    float * brightness,
    float * wavelength,
    float * point_error,
    float2 * jones,
    float refwave,
    int nbl, int nchan, int ntime, int nsrc, int na)
{
    // Our data space is a 4D matrix of BL x SRC x CHAN x TIME

    #define COS3_CONST 65*1e-9

    // Baseline, Source, Channel and Time indices
    int SRC = blockIdx.x*blockDim.x + threadIdx.x;
    int CHAN = (blockIdx.y*blockDim.y + threadIdx.y) / ntime;
    int TIME = (blockIdx.y*blockDim.y + threadIdx.y) % ntime;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;
    // Calculates the antenna pairs from the baseline!
    int ANT1 = int(floor((sqrtf(1+8*BL)-1)/2));
    int ANT2 = BL - (ANT1*ANT1+ANT1)/2;
    ANT1 += 1;

    if(BL >= nbl || SRC >= nsrc || CHAN >= nchan || TIME >= ntime)
        return;

    // Cache input and output data from global memory.

    // Pointing errors for antenna one (p) and two (q)
    float * ld_p = smem_f;
    float * md_p = &ld_p[blockDim.z];
    float * ld_q = &md_p[blockDim.z];
    float * md_q = &ld_q[blockDim.z];

    // Point source coordinates, their flux
    // and brightness matrix
    float * l = &md_q[blockDim.z];
    float * m = &l[blockDim.x];
    float * fI = &m[blockDim.x];
    float * fV = &fI[blockDim.x];
    float * fU = &fV[blockDim.x];
    float * fQ = &fU[blockDim.x];    

    // Wavelengths
    float * wave = &fQ[blockDim.x];

    // Index
    int i;

    if(threadIdx.x == 0)
    {
        i = ANT1*ntime + TIME; ld_p[threadIdx.z] = point_error[i];
        i += na*ntime;         md_p[threadIdx.z] = point_error[i];
        i = ANT2*ntime + TIME; ld_q[threadIdx.z] = point_error[i];
        i += na*ntime;         md_q[threadIdx.z] = point_error[i];
    }

    if(threadIdx.z == 0)
    {
        i = SRC;   l[threadIdx.x] = LM[i]; fI[threadIdx.x] = brightness[i];
        i += nsrc; m[threadIdx.x] = LM[i]; fQ[threadIdx.x] = brightness[i];
        i += nsrc; fU[threadIdx.x] = brightness[i];
        i += nsrc; fV[threadIdx.x] = brightness[i];        
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

    // UVW is 3 x nbl x ntime matrix
    // u*l + v*m + w*n, in the wrong order :)
    i = BL*ntime + TIME + 2*nbl*ntime;         phase *= UVW[i]; // w*n
    i -= nbl*ntime;             phase += UVW[i]*m[threadIdx.x]; // v*m
    i -= nbl*ntime;             phase += UVW[i]*l[threadIdx.x]; // u*l

    // Multiply by 2*pi/wave[threadIdx.y]
    phase *= (2. * CUDART_PI);
    phase /= wave[threadIdx.y];

    // Calculate the complex exponential from the phase
    float real, imag;
    __sincosf(phase, &imag, &real);

    // Multiply by the wavelength to the power of alpha
    i = SRC+nsrc*4; phase = __powf(refwave/wave[threadIdx.y], brightness[i]);
    real *= phase; imag *= phase;

    float E_p = (l[threadIdx.z]+ld_p[threadIdx.z])*(l[threadIdx.z]*ld_p[threadIdx.z]);
    E_p += (m[threadIdx.z]+md_p[threadIdx.z])*(m[threadIdx.z]*md_p[threadIdx.z]);
    E_p = sqrt(E_p);
    E_p = __cosf(COS3_CONST*wave[threadIdx.y]*E_p);
    E_p = E_p*E_p*E_p;
    real *= E_p; imag *= E_p;

    float E_q = (l[threadIdx.z]+ld_q[threadIdx.z])*(l[threadIdx.z]*ld_q[threadIdx.z]);
    E_q += (m[threadIdx.z]+md_q[threadIdx.z])*(m[threadIdx.z]*md_q[threadIdx.z]);
    E_q = sqrt(E_q);
    E_q = __cosf(COS3_CONST*wave[threadIdx.y]*E_q);
    E_q = E_q*E_q*E_q;
    real *= E_q; imag *= E_q;

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
    i = (BL*nchan*ntime*nsrc + CHAN*ntime*nsrc + TIME*nsrc + SRC);

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

class RimeEBKFloat(Node):
    def __init__(self):
        super(RimeEBKFloat, self).__init__()
        self.mod = SourceModule(FLOAT_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_EBK_float')

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
            'block' : (baselines_per_block,srcs_per_block,1), \
            'grid'  : (baseline_blocks,src_blocks,time_chan_blocks), \
            'shared' : (4*baselines_per_block + \
                        6*srcs_per_block + \
                        1*time_chans_per_block)*\
                            np.dtype(sd.ft).itemsize }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu,
            sd.wavelength_gpu, sd.point_errors_gpu, sd.jones_gpu, sd.refwave,
            np.int32(sd.nbl), np.int32(sd.nchan), np.int32(sd.ntime), np.int32(sd.nsrc),
            np.int32(sd.na), **self.get_kernel_params(sd))       

    def post_execution(self, shared_data):
        pass
