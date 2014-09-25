import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from montblanc.node import Node

FLOAT_KERNEL = """
#include \"math_constants.h\"

extern __shared__ float smem_f[];

__global__
void rime_jones_EBK_sum_float(
    float * UVW,
    float * LM,
    float * brightness,
    float * wavelength,
    float * point_error,
    float2 * visibilities,
    float ref_wave,
    int nbl, int nchan, int ntime, int npsrc, int na)
{
    // Our data space is a 4D matrix of BL x SRC x CHAN x TIME

    #define COS3_CONST 65*1e-9

    // Baseline, Channel and Time indices
    int TIME = blockIdx.x*blockDim.x + threadIdx.x;
    int CHAN = blockIdx.y*blockDim.y + threadIdx.y;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= nbl || CHAN >= nchan || TIME >= ntime)
        return;

    // Calculates the antenna pairs from the baseline!
    int ANT1 = int(floor((sqrtf(1+8*BL)-1)/2));
    int ANT2 = BL - (ANT1*ANT1+ANT1)/2;
    ANT1 += 1;

    float * l = smem_f;       // l is at beginning of shared mem
    float * m = &l[npsrc];     // m is at the end of l

    float * fI = &m[npsrc];    // fI is at the end of m
    float * fQ = &fI[npsrc];   // fQ is at the end of fI
    float * fV = &fQ[npsrc];   // fV is at the end of fQ
    float * fU = &fV[npsrc];   // fU is at the end of fV

    float * wave = &fU[npsrc]; // wave is at the end of fU

    float * ld_p = &wave[blockDim.y]; // Number of CHANS
    float * md_p = &ld_p[blockDim.z*blockDim.x]; // BL*TIME
    float * ld_q = &md_p[blockDim.z*blockDim.x]; // BL*TIME
    float * md_q = &ld_q[blockDim.z*blockDim.x]; // BL*TIME

    int i;

    // Keep our y (CHAN) dimension constant.
    // Then we can load in stuff required by our changing
    // x (TIME) and z (BL) dimensions. Specifically, pointing errors
    if(threadIdx.y == 0) 
    {
        int j = threadIdx.z*blockDim.x+threadIdx.x;
        i = ANT1*ntime + TIME; ld_p[j] = point_error[i];
        i += na*ntime;         md_p[j] = point_error[i];
        i = ANT2*ntime + TIME; ld_q[j] = point_error[i];
        i += na*ntime;         md_q[j] = point_error[i];
    }

    // Keep our x (TIME) and z (BL) dimensions constant
    // Then we can load in stuff required by our changing
    // y (CHAN) dimension. Specifically, wavelengths
    if(threadIdx.x == 0 && threadIdx.z == 0)
    {
        i = CHAN; wave[threadIdx.y] = wavelength[i];
    }

    // Shouldn't need this because of __synchthreads() in loop
    // __syncthreads();

    float2 jones_1_sum = make_float2(0.0f,0.0f);
    float2 jones_2_sum = make_float2(0.0f,0.0f);
    float2 jones_3_sum = make_float2(0.0f,0.0f);
    float2 jones_4_sum = make_float2(0.0f,0.0f);

    for(int SRC=0; SRC<npsrc; ++SRC)
    {
        // Load point source data into shared memory using one thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC;   l[SRC] = LM[i]; fI[SRC] = brightness[i];
            i += npsrc; m[SRC] = LM[i]; fQ[SRC] = brightness[i];
            i += npsrc; fU[SRC] = brightness[i];
            i += npsrc; fV[SRC] = brightness[i];
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

        // Multiply by 2*pi/wave[threadIdx.y]
        phase *= (2. * CUDART_PI);
        phase /= wave[threadIdx.y];

        // Calculate the complex exponential from the phase
        float real, imag;
        __sincosf(phase, &imag, &real);

        // Multiply by the wavelength to the power of alpha
        i = SRC+npsrc*4; phase = __powf(ref_wave/wave[threadIdx.y], brightness[i]);
        real *= phase; imag *= phase;        

        jones_1_sum.x += (fI[SRC]+fQ[SRC])*real - 0.0*imag;
        jones_1_sum.y += (fI[SRC]+fQ[SRC])*imag + 0.0*real;

        jones_2_sum.x += fU[SRC]*real - fV[SRC]*imag;
        jones_2_sum.y += fU[SRC]*imag + fV[SRC]*real;

        jones_3_sum.x += fU[SRC]*real - -fV[SRC]*imag;
        jones_3_sum.y += fU[SRC]*imag + -fV[SRC]*real;

        jones_4_sum.x += (fI[SRC]-fQ[SRC])*real - 0.0*imag;
        jones_4_sum.y += (fI[SRC]-fQ[SRC])*imag + 0.0*real;
    }

    // Index into the complex visibilities
    i = BL*nchan*ntime + CHAN*ntime + TIME;
    visibilities[i] = jones_1_sum;

    i += nbl*nchan*ntime;
    visibilities[i] = jones_2_sum;

    i += nbl*nchan*ntime;
    visibilities[i] = jones_3_sum;

    i += nbl*nchan*ntime;
    visibilities[i] = jones_4_sum;
}
"""

class RimeEBKSumFloat(Node):
    def __init__(self):
        super(RimeEBKSumFloat, self).__init__()

    def initialise(self, solver):
        self.mod = SourceModule(FLOAT_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_EBK_sum_float')

    def shutdown(self, solver):
        pass

    def pre_execution(self, solver):
        pass

    def get_kernel_params(self, solver):
        slvr = solver

        times_per_block = 16 if slvr.ntime > 2 else slvr.ntime
        chans_per_block = 8 if slvr.nchan > 2 else slvr.nchan
        baselines_per_block = 1 if slvr.nbl > 16 else slvr.nbl

        time_blocks = (slvr.ntime + times_per_block - 1) / times_per_block
        chan_blocks = (slvr.nchan + chans_per_block - 1) / chans_per_block
        baseline_blocks = (slvr.nbl + baselines_per_block - 1)/ baselines_per_block

        return {
            'block' : (times_per_block,chans_per_block,baselines_per_block),
            'grid'  : (time_blocks,chan_blocks,baseline_blocks),
            'shared' : (1*chans_per_block +
                6*slvr.npsrc +
                4*baselines_per_block*times_per_block)*
                    np.dtype(slvr.ft).itemsize }

    def execute(self, solver):
        slvr = solver

        self.kernel(slvr.uvw_gpu, slvr.lm_gpu, slvr.brightness_gpu,
            slvr.wavelength_gpu, slvr.point_errors_gpu, slvr.vis_gpu, slvr.ref_wave,
            np.int32(slvr.nbl), np.int32(slvr.nchan), np.int32(slvr.ntime), np.int32(slvr.npsrc),
            np.int32(slvr.na), **self.get_kernel_params(slvr))       

    def post_execution(self, solver):
        pass
