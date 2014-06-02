import numpy as np
import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKX' : 32,
    'BLOCKY' : 1,
    'BLOCKZ' : 8
}

DOUBLE_PARAMS = {
    'BLOCKX' : 32,
    'BLOCKY' : 1,
    'BLOCKZ' : 8
}

KERNEL_TEMPLATE = string.Template("""
#include <cstdio>
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>

#define NA ${na}
#define NBL ${nbl}
#define NCHAN ${nchan}
#define NTIME ${ntime}
#define NSRC ${nsrc}

#define BLOCKDIMX ${BLOCKX}
#define BLOCKDIMY ${BLOCKY}
#define BLOCKDIMZ ${BLOCKZ}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_EK_impl(
    const T * __restrict__ UVW,
    const typename Tr::ft * __restrict__ LM,
    const typename Tr::ft * __restrict__ brightness,
    const typename Tr::ft * __restrict__ wavelength,
    const typename Tr::ft * __restrict__ point_error,
    const int * __restrict__ ant_pairs,
    typename Tr::ct * __restrict__ jones_scalar,
    typename Tr::ft ref_wave,
    typename Tr::ft E_beam_width,
    typename Tr::ft E_beam_clip)
{
    // Our data space is a 4D matrix of BL x CHAN x TIME x SRC
    // Baseline, Source, Channel and Time indices
    int SRC = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = (blockIdx.y*blockDim.y + threadIdx.y) % NTIME;
    int CHAN = (blockIdx.y*blockDim.y + threadIdx.y) / NTIME;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= NBL || SRC >= NSRC || CHAN >= NCHAN || TIME >= NTIME)
        return;

    // Cache input and output data from global memory.

    // Pointing errors for antenna one (p) and two (q)
    // This is technically bl x time, but since
    // our Y block dimension (time) is only 1 unit,
    // this works
    __shared__ typename Tr::ft ld_p[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft md_p[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft ld_q[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft md_q[BLOCKDIMZ*BLOCKDIMY];

    // Point source coordinates
    __shared__ typename Tr::ft l[BLOCKDIMX];
    __shared__ typename Tr::ft m[BLOCKDIMX];

    // Wavelengths
    // Should only have one here
    __shared__ typename Tr::ft wave[BLOCKDIMY];

    // Index
    int i;

    // Varies by time (y) and antenna (baseline) (z) 
    if(threadIdx.x == 0)
    {
        // Determine antenna pairs for this baseline
        i = BL*NTIME + TIME; int ANT1 = ant_pairs[i];
        i += NBL*NTIME;      int ANT2 = ant_pairs[i];

        // Pointing error index
        // baseline x BLOCKDIMY + TIME;
        // At present BLOCKDIMY should be 1 and threadIdx.y 0.
        int j = threadIdx.z*BLOCKDIMY + threadIdx.y;

        // Load in the pointing errors
        i = ANT1*NTIME + TIME; ld_p[j] = point_error[i];
        i += NA*NTIME;         md_p[j] = point_error[i];
        i = ANT2*NTIME + TIME; ld_q[j] = point_error[i];
        i += NA*NTIME;         md_q[j] = point_error[i];
    }

    // Varies by source (x)
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        i = SRC;    l[threadIdx.x] = LM[i];
        i += NSRC;  m[threadIdx.x] = LM[i];
    }

    // Varies by channel (y)
    if(threadIdx.x == 0 && threadIdx.z == 0)
    {
        wave[threadIdx.y] = wavelength[CHAN];
    }

    __syncthreads();

    // Calculate the n term first
    // n = Po::sqrt(1.0 - l*l - m*m) - 1.0
    typename Tr::ft phase = 1.0 - l[threadIdx.x]*l[threadIdx.x];
    phase -= m[threadIdx.x]*m[threadIdx.x];
    phase = Po::sqrt(phase) - 1.0;
    // TODO: remove this superfluous variable
    // It only exists for debugging purposes
    // Tr::ft n = phase;

    // UVW is 3 x NBL x NTIME matrix
    // u*l + v*m + w*n, in the wrong order :)
    i = BL*NTIME + TIME + 2*NBL*NTIME;  phase *= UVW[i]; // w*n
    i -= NBL*NTIME;      phase += UVW[i]*m[threadIdx.x]; // v*m
    i -= NBL*NTIME;      phase += UVW[i]*l[threadIdx.x]; // u*l

    // Multiply by 2*pi/wave[threadIdx.y]
    phase *= (2. * Tr::cuda_pi);
    phase /= wave[threadIdx.y];

    // Calculate the complex exponential from the phase
    typename Tr::ft real, imag;
    Po::sincos(phase, &imag, &real);

    // Multiply by the wavelength to the power of alpha
    i = SRC+NSRC*4; phase = Po::pow(ref_wave/wave[threadIdx.y], brightness[i]);
    real *= phase; imag *= phase;

    {
        typename Tr::ft diff = l[threadIdx.x]-ld_p[threadIdx.z];
        typename Tr::ft E_p = diff*diff;
        diff = m[threadIdx.x]-md_p[threadIdx.z];
        E_p += diff*diff;
        E_p = Po::sqrt(E_p);
        E_p *= E_beam_width*1e-9*wave[threadIdx.y];
        E_p = Po::min(E_p, E_beam_clip);
        E_p = Po::cos(E_p);
        E_p = E_p*E_p*E_p;
        real *= E_p; imag *= E_p;
    }

    {
        typename Tr::ft diff = l[threadIdx.x]-ld_q[threadIdx.z];
        typename Tr::ft E_q = diff*diff;
        diff = m[threadIdx.x]-md_q[threadIdx.z];
        E_q += diff*diff;
        E_q = Po::sqrt(E_q);
        E_q *= E_beam_width*1e-9*wave[threadIdx.y];
        E_q = Po::min(E_q, E_beam_clip);
        E_q = Po::cos(E_q);
        E_q = E_q*E_q*E_q;
        real *= E_q; imag *= E_q;
    }

    // Index into the jones matrices
    i = (BL*NCHAN*NTIME + CHAN*NTIME + TIME)*NSRC + SRC;

    jones_scalar[i]=Po::make_ct(real, imag);
}

extern "C" {

__global__
void rime_jones_EK_float(
    const float *  __restrict__ UVW,
    const float *  __restrict__ LM,
    const float  *  __restrict__ brightness,
    const float  *  __restrict__ wavelength,
    const float  *  __restrict__ point_error,
    const int  *  __restrict__ ant_pairs,
    float2 * __restrict__ jones_scalar,
    float ref_wave,
    float E_beam_width,
    float E_beam_clip)
{
    rime_jones_EK_impl(UVW, LM, brightness, wavelength, point_error,
        ant_pairs, jones_scalar, ref_wave, E_beam_width, E_beam_clip);
}

__global__
void rime_jones_EK_double(
    const double *  __restrict__ UVW,
    const double *  __restrict__ LM,
    const double *  __restrict__ brightness,
    const double *  __restrict__ wavelength,
    const double *  __restrict__ point_error,
    const int * __restrict__ ant_pairs,
    double2 *  __restrict__ jones_scalar,
    double ref_wave,
    double E_beam_width,
    double E_beam_clip)
{
    rime_jones_EK_impl(UVW, LM, brightness, wavelength, point_error,
        ant_pairs, jones_scalar, ref_wave, E_beam_width, E_beam_clip);
}

} // extern "C" {
""")

class RimeEK(Node):
    def __init__(self):
        super(RimeEK, self).__init__()

    def initialise(self, shared_data):
        sd = shared_data

        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS
        D.update(sd.get_params())

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo'],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        kname = 'rime_jones_EK_float' \
            if sd.is_float() is True else \
            'rime_jones_EK_double'

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, shared_data):
        pass

    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data
        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS

        srcs_per_block = D['BLOCKX'] if sd.nsrc > D['BLOCKX'] else sd.nsrc
        time_chans_per_block = D['BLOCKY']
        baselines_per_block = D['BLOCKZ'] if sd.nbl > D['BLOCKZ'] else sd.nbl

        src_blocks = self.blocks_required(sd.nsrc,srcs_per_block)
        time_chan_blocks = sd.ntime*sd.nchan
        baseline_blocks = self.blocks_required(sd.nbl, baselines_per_block)

        return {
            'block' : (srcs_per_block,time_chans_per_block,baselines_per_block), \
            'grid'  : (src_blocks,time_chan_blocks,baseline_blocks)
        }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu,
            sd.wavelength_gpu, sd.point_errors_gpu, sd.ant_pairs_gpu, sd.jones_scalar_gpu,
            sd.ref_wave, sd.beam_width, sd.E_beam_clip ,
            **self.get_kernel_params(sd))       

    def post_execution(self, shared_data):
        pass
