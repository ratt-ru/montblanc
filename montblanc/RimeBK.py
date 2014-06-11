import numpy as np
import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,
    'BLOCKDIMY' : 1,
    'BLOCKDIMZ' : 8
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,
    'BLOCKDIMY' : 1,
    'BLOCKDIMZ' : 8
}

KERNEL_TEMPLATE = string.Template("""
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>

#define NA ${na}
#define NBL ${nbl}
#define NCHAN ${nchan}
#define NTIME ${ntime}
#define NPSRC ${npsrc}

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_BK_impl(
    T * UVW,
    typename Tr::ft * LM,
    typename Tr::ft * brightness,
    typename Tr::ft * wavelength,
    typename Tr::ct * jones,
    typename Tr::ft ref_wave)
{
    // Our data space is a 4D matrix of BL x CHAN x TIME x SRC

    // Baseline, Source, Channel and Time indices
    int SRC = blockIdx.x*blockDim.x + threadIdx.x;
    int CHAN = (blockIdx.y*blockDim.y + threadIdx.y) / NTIME;
    int TIME = (blockIdx.y*blockDim.y + threadIdx.y) % NTIME;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;
    if(BL >= NBL || SRC >= NPSRC || CHAN >= NCHAN || TIME >= NTIME)
        return;

    /* Cache input and output data from global memory. */

    __shared__ typename Tr::ft u[BLOCKDIMZ];
    __shared__ typename Tr::ft v[BLOCKDIMZ];
    __shared__ typename Tr::ft w[BLOCKDIMZ];

    __shared__ typename Tr::ft l[BLOCKDIMX];
    __shared__ typename Tr::ft m[BLOCKDIMX];
    __shared__ typename Tr::ft fI[BLOCKDIMX];
    __shared__ typename Tr::ft fQ[BLOCKDIMX];
    __shared__ typename Tr::ft fV[BLOCKDIMX];
    __shared__ typename Tr::ft fU[BLOCKDIMX];
    __shared__ typename Tr::ft a[BLOCKDIMX];

    __shared__ typename Tr::ft wave[BLOCKDIMY];

    // Index
    int i;

    if(threadIdx.x == 0)
    {
        // UVW is a 3 x NBL x NTIME matrix
        i = BL*NTIME + TIME; u[threadIdx.z] = UVW[i];
        i += NBL*NTIME;      v[threadIdx.z] = UVW[i];
        i += NBL*NTIME;      w[threadIdx.z] = UVW[i];
    }

    if(threadIdx.z == 0)
    {
		// LM and brightness are 2 x NPSRC and 5 x NPSRC matrices
        i = SRC;   l[threadIdx.x] = LM[i];
        i += NPSRC; m[threadIdx.x] = LM[i];

        i = SRC;   fI[threadIdx.x] = brightness[i];
        i += NPSRC; fQ[threadIdx.x] = brightness[i];
        i += NPSRC; fU[threadIdx.x] = brightness[i];
        i += NPSRC; fV[threadIdx.x] = brightness[i];
        i += NPSRC; a[threadIdx.x] = brightness[i];
    }

    if(threadIdx.y == 0)
    {
        i = CHAN; wave[threadIdx.y] = wavelength[i];
    }

    __syncthreads();

    // Calculate the n term first
    // n = sqrt(1.0 - l*l - m*m) - 1.0
    typename Tr::ft phase = 1.0 - l[threadIdx.x]*l[threadIdx.x];
    phase -= m[threadIdx.x]*m[threadIdx.x];
    phase = Po::sqrt(phase) - 1.0;
    // TODO: remove this superfluous variable
    // It only exists for debugging purposes
    //  typename Tr::ft n = phase;

    // u*l + v*m + w*n, in the wrong order :)
    phase *= w[threadIdx.z];                  // w*n
    phase += v[threadIdx.z]*m[threadIdx.x];   // v*m
    phase += u[threadIdx.z]*l[threadIdx.x];   // u*l

    // Multiply by 2*pi/wave[threadIdx.y]
    phase *= (2. * Tr::cuda_pi);
    phase /= wave[threadIdx.y];

    // Calculate the complex exponential from the phase
    typename Tr::ft real, imag;
    Po::sincos(phase, &imag, &real);

    // Multiply by the wavelength to the power of alpha
    phase = Po::pow(ref_wave/wave[threadIdx.y], a[threadIdx.x]);
    real *= phase; imag *= phase;

    // Index into the jones matrices
    i = BL*NCHAN*NTIME*NPSRC + CHAN*NTIME*NPSRC + TIME*NPSRC + SRC;

    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = fI+fQ, b=0.0, c=real, d = imag
    jones[i]=Po::make_ct(
        (fI[threadIdx.x]+fQ[threadIdx.x])*real - 0.0*imag,
        (fI[threadIdx.x]+fQ[threadIdx.x])*imag + 0.0*real);

    // a=fU, b=fV, c=real, d = imag 
    i += NBL*NPSRC*NCHAN*NTIME;
    jones[i]=Po::make_ct(
        fU[threadIdx.x]*real - fV[threadIdx.x]*imag,
        fU[threadIdx.x]*imag + fV[threadIdx.x]*real);

    // a=fU, b=-fV, c=real, d = imag 
    i += NBL*NPSRC*NCHAN*NTIME;
    jones[i]=Po::make_ct(
        fU[threadIdx.x]*real - -fV[threadIdx.x]*imag,
        fU[threadIdx.x]*imag + -fV[threadIdx.x]*real);

    // a=fI-fQ, b=0.0, c=real, d = imag 
    i += NBL*NPSRC*NCHAN*NTIME;
    jones[i]=Po::make_ct(
        (fI[threadIdx.x]-fQ[threadIdx.x])*real - 0.0*imag,
        (fI[threadIdx.x]-fQ[threadIdx.x])*imag + 0.0*real);
}

extern "C" {

__global__ void rime_jones_BK_float(
    float * UVW,
    float * LM,
    float * brightness,
    float * wavelength,
    float2 * jones,
    float ref_wave)
{
    rime_jones_BK_impl(UVW, LM, brightness, wavelength,
        jones, ref_wave);
}


__global__ void rime_jones_BK_double(
    double * UVW,
    double * LM,
    double * brightness,
    double * wavelength,
    double2 * jones,
    double ref_wave)
{
    rime_jones_BK_impl(UVW, LM, brightness, wavelength,
        jones, ref_wave);
}

} // extern "C" {}
""")

class RimeBK(Node):
    def __init__(self):
        super(RimeBK, self).__init__()

    def initialise(self, shared_data):
        sd = shared_data

        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS
        D.update(sd.get_params())

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo'],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        kname = 'rime_jones_BK_float' \
            if sd.is_float() is True else \
            'rime_jones_BK_double'

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, shared_data):
        pass

    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS

        psrcs_per_block = D['BLOCKDIMX'] if sd.npsrc > D['BLOCKDIMX'] else sd.npsrc
        time_chans_per_block = D['BLOCKDIMY']
        baselines_per_block = D['BLOCKDIMZ'] if sd.nbl > D['BLOCKDIMZ'] else sd.nbl

        psrc_blocks = self.blocks_required(sd.npsrc,psrcs_per_block)
        baseline_blocks = self.blocks_required(sd.nbl,baselines_per_block)
        time_chan_blocks = sd.ntime*sd.nchan

        return {
            'block' : (psrcs_per_block,time_chans_per_block,baselines_per_block),
            'grid'  : (psrc_blocks,time_chan_blocks,baseline_blocks), 
        }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu,
            sd.wavelength_gpu,  sd.jones_gpu, sd.ref_wave,
            **self.get_kernel_params(sd))

    def post_execution(self, shared_data):
        pass
