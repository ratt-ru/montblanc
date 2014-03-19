import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

DOUBLE_KERNEL = """
#include \"math_constants.h\"

extern __shared__ double smem_d[];

// Based on OSKAR's implementation of the RIME K term.
// Baseline on the x dimension, source on the y dimension
__global__
void rime_jones_EBK(
    double * UVW,
    double * LMA,
    double * sky,
    double * wavelength,
    double * point_error,
    double2 * jones,
    int nsrc, int nbl, int nchan, int ntime, int na)
{
    // Our data space is a 4D matrix of BL x SRC x CHAN x TIME

    #define REFWAVE 1e6
    #define COS3_CONST 45*1e-9

    // Baseline, Source, Channel and Time indices
    int BL = blockIdx.x*blockDim.x + threadIdx.x;
    int SRC = blockIdx.y*blockDim.y + threadIdx.y;
    int CHAN = (blockIdx.z*blockDim.z + threadIdx.z) / ntime;
    int TIME = (blockIdx.z*blockDim.z + threadIdx.z) % ntime;
    // Calculates the antenna pairs from the baseline!
    int ANT1 = int(floor((sqrtf(1+8*BL)-1)/2));
    int ANT2 = ANT1*(ANT1-1)/2;
    ANT1 += 1;

    if(BL >= nbl || SRC >= nsrc || CHAN >= nchan || TIME >= ntime)
        return;

    // Cache input and output data from global memory.

    // Pointing errors for antenna one (p) and two (q)
    double * ld_p = &smem_d[blockDim.x];
    double * md_p = &ld_p[blockDim.x];
    double * ld_q = &md_p[blockDim.x];
    double * md_q = &ld_q[blockDim.x];

    // Point source coordinates, their flux
    // and brightness matrix
    double * l = &md_q[blockDim.x];
    double * m = &l[blockDim.y];
    double * fI = &m[blockDim.y];
    double * fV = &fI[blockDim.y];
    double * fU = &fV[blockDim.y];
    double * fQ = &fU[blockDim.y];    

    // Wavelengths
    double * wave = &fQ[blockDim.y];

    // Index
    int i;

    if(threadIdx.y == 0)
    {
        i = ANT1;      ld_p[threadIdx.x] = point_error[i];
        i += na*ntime; md_p[threadIdx.x] = point_error[i];
        i = ANT2;      ld_q[threadIdx.x] = point_error[i];
        i += na*ntime; md_q[threadIdx.x] = point_error[i];
    }

    if(threadIdx.x == 0)
    {
        i = SRC;   l[threadIdx.y] = LMA[i]; fI[threadIdx.y] = sky[i];
        i += nsrc; m[threadIdx.y] = LMA[i]; fU[threadIdx.y] = sky[i];
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
    double phase = 1.0 - l[threadIdx.y]*l[threadIdx.y];
    phase -= m[threadIdx.y]*m[threadIdx.y];
    phase = sqrt(phase) - 1.0;
    // TODO: remove this superfluous variable
    // It only exists for debugging purposes
    // double n = phase;

    // UVW is 3 x nbl x ntime matrix
    // u*l + v*m + w*n, in the wrong order :)
    i = BL*ntime + TIME + 2*nbl*ntime;         phase *= UVW[i]; // w*n
    i -= nbl*ntime;             phase += UVW[i]*m[threadIdx.y]; // v*m
    i -= nbl*ntime;             phase += UVW[i]*l[threadIdx.y]; // u*l

    // Multiply by 2*pi/wave[threadIdx.z]
    phase *= (2. * CUDART_PI);
    phase /= wave[threadIdx.z];

    // Calculate the complex exponential from the phase
    double real, imag;
    sincos(phase, &imag, &real);

    // Multiply by the wavelength to the power of alpha
    i = SRC+nsrc*2; phase = pow(REFWAVE/wave[threadIdx.z], LMA[i]);
    real *= phase; imag *= phase;

    double E_p = (l[threadIdx.x]+ld_p[threadIdx.x])*(l[threadIdx.x]*ld_p[threadIdx.x]);
    E_p += (m[threadIdx.x]+md_p[threadIdx.x])*(m[threadIdx.x]*md_p[threadIdx.x]);
    E_p = sqrt(E_p);
    E_p = cos(COS3_CONST*wave[threadIdx.z]*E_p);
    E_p = E_p*E_p*E_p;
    real *= E_p; imag *= E_p;

    double E_q = (l[threadIdx.x]+ld_q[threadIdx.x])*(l[threadIdx.x]*ld_q[threadIdx.x]);
    E_q += (m[threadIdx.x]+md_q[threadIdx.x])*(m[threadIdx.x]*md_q[threadIdx.x]);
    E_q = sqrt(E_q);
    E_q = cos(COS3_CONST*wave[threadIdx.z]*E_q);
    E_q = E_q*E_q*E_q;
    real *= E_q; imag *= E_q;

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
    // Index into the jones matrices
    i = (BL*nchan*ntime*nsrc + CHAN*ntime*nsrc + TIME*nsrc + SRC);

    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = fI+fQ, b=0.0, c=real, d = imag
    jones[i]=make_double2(
        (fI[threadIdx.y]+fQ[threadIdx.y])*real - 0.0*imag,
        (fI[threadIdx.y]+fQ[threadIdx.y])*imag + 0.0*real);

    // a=fU, b=fV, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_double2(
        fU[threadIdx.y]*real - fV[threadIdx.y]*imag,
        fU[threadIdx.y]*imag + fV[threadIdx.y]*real);

    // a=fU, b=-fV, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_double2(
        fU[threadIdx.y]*real - -fV[threadIdx.y]*imag,
        fU[threadIdx.y]*imag + -fV[threadIdx.y]*real);

    // a=fI-fQ, b=0.0, c=real, d = imag 
    i += nbl*nsrc*nchan*ntime;
    jones[i]=make_double2(
        (fI[threadIdx.y]-fQ[threadIdx.y])*real - 0.0*imag,
        (fI[threadIdx.y]-fQ[threadIdx.y])*imag + 0.0*real);
#endif

    #undef REFWAVE
}
"""

class RimeJonesEBK(Node):
    def __init__(self):
        super(RimeJonesEBK, self).__init__()
    def initialise(self, shared_data):
        self.mod = SourceModule(DOUBLE_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_EBK')

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
            'shared' : (7*baselines_per_block + \
                        7*srcs_per_block + \
                        1*time_chans_per_block)*\
                            np.dtype(np.float64).itemsize }

    def execute(self, shared_data):
        sd = shared_data
        params = self.get_kernel_params(sd)

        self.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
            sd.wavelength_gpu, sd.point_errors_gpu, sd.jones_gpu,
            np.int32(sd.nsrc), np.int32(sd.nbl), np.int32(sd.na), **params)

    def post_execution(self, shared_data):
        pass