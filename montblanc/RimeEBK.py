import numpy as np
import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 4,
    'BLOCKDIMY' : 8,
    'BLOCKDIMZ' : 4
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 4,
    'BLOCKDIMY' : 8,
    'BLOCKDIMZ' : 4
}

KERNEL_TEMPLATE = string.Template("""
#include <cstdio>
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>

#define NA ${na}
#define NBL ${nbl}
#define NCHAN ${nchan}
#define NTIME ${ntime}
#define NPSRC ${npsrc}
#define NGSRC ${ngsrc}
#define NSRC ${nsrc}

#define GAUSS_SCALE ${gauss_scale}

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

template <
    typename T,
    unsigned int NISRC,
    bool gaussian=false,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_EBK_impl(
    typename Tr::ft * UVW,
    typename Tr::ft * LM,
    typename Tr::ft * brightness,
    typename Tr::ft * gauss_shape,
    typename Tr::ft * wavelength,
    typename Tr::ft * point_error,
    int * ant_pairs,
    typename Tr::ct * jones,
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

    if(BL >= NBL || SRC >= NISRC || CHAN >= NCHAN || TIME >= NTIME)
        return;

    // Cache input and output data from global memory.

    // UVW coordinates and
    // Pointing errors for antenna one (p) and two (q)
    // These are technically bl x time, but since
    // our Y block dimension (time) is only 1 unit,
    // this works
    __shared__ typename Tr::ft u[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft v[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft w[BLOCKDIMZ*BLOCKDIMY];

    __shared__ typename Tr::ft ld_p[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft md_p[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft ld_q[BLOCKDIMZ*BLOCKDIMY];
    __shared__ typename Tr::ft md_q[BLOCKDIMZ*BLOCKDIMY];

    // Point source coordinates, their flux
    // and brightness matrix
    __shared__ typename Tr::ft l[BLOCKDIMX];
    __shared__ typename Tr::ft m[BLOCKDIMX];
    __shared__ typename Tr::ft fI[BLOCKDIMX*BLOCKDIMY];
    __shared__ typename Tr::ft fQ[BLOCKDIMX*BLOCKDIMY];
    __shared__ typename Tr::ft fV[BLOCKDIMX*BLOCKDIMY];
    __shared__ typename Tr::ft fU[BLOCKDIMX*BLOCKDIMY];

    // Wavelengths
    // Should only have one here
    __shared__ typename Tr::ft wave[BLOCKDIMY];

    // Index
    int i;

    // Varies by time (y) and antenna (baseline) (z) 
    if(threadIdx.x == 0)
    {
        // UVW and Pointing error indices
        // baseline x BLOCKDIMY + TIME;
        // At present BLOCKDIMY should be 1 and threadIdx.y 0.
        int j = threadIdx.z*BLOCKDIMY + threadIdx.y;

        // Load in UVW coordinates and antenna pairs for this baseline
        i = BL*NTIME + TIME; u[j] = UVW[i]; int ANT1 = ant_pairs[i];
        i += NBL*NTIME;      v[j] = UVW[i]; int ANT2 = ant_pairs[i];
        i += NBL*NTIME;      w[j] = UVW[i];

        // Load in the pointing errors
        i = ANT1*NTIME + TIME; ld_p[j] = point_error[i];
        i += NA*NTIME;         md_p[j] = point_error[i];
        i = ANT2*NTIME + TIME; ld_q[j] = point_error[i];
        i += NA*NTIME;         md_q[j] = point_error[i];
    }

    // Varies by source (x)
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        i = SRC;   l[threadIdx.x] = LM[i];
        i += NSRC; m[threadIdx.x] = LM[i];
    }

    // Varies by time (y) and source (x)
    if(threadIdx.z == 0)
    {
        // TIME*NSRC + SRC;
        int j = threadIdx.y*BLOCKDIMX + threadIdx.x;

        i = TIME*NSRC + SRC; fI[j] = brightness[i];
        i += NTIME*NSRC;     fQ[j] = brightness[i];
        i += NTIME*NSRC;     fU[j] = brightness[i];
        i += NTIME*NSRC;     fV[j] = brightness[i];
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
    // typename Tr::ft n = phase;

    // u*l + v*m + w*n, in the wrong order :)
    int j = threadIdx.z*BLOCKDIMY + threadIdx.y;
    phase *= w[j];                // w*n
    phase += v[j]*m[threadIdx.x]; // v*m
    phase += u[j]*l[threadIdx.x]; // u*l

    // Multiply by 2*pi/wave[threadIdx.y]
    phase *= (2. * Tr::cuda_pi);
    phase /= wave[threadIdx.y];

    // Calculate the complex exponential from the phase
    typename Tr::ft real, imag;
    Po::sincos(phase, &imag, &real);

    // Multiply by the wavelength to the power of alpha
    i = TIME*NSRC + SRC + 4*NTIME*NSRC;
    phase = Po::pow(ref_wave/wave[threadIdx.y], brightness[i]);
    real *= phase; imag *= phase;

    if(gaussian)
    {
        // Load in the el and em parameters of the gaussian shape
        i = SRC;    typename Tr::ft el = gauss_shape[i];
        i += NGSRC; typename Tr::ft em = gauss_shape[i];

        int j = threadIdx.z*BLOCKDIMY + threadIdx.y;

        typename Tr::ft u1 = u[j]*em - v[j]*el;
        typename Tr::ft v1 = u[j]*el + v[j]*em;

        // Work out the scaling factor.
        typename Tr::ft scale_uv = GAUSS_SCALE/wave[threadIdx.y];

        // Load in the ratio parameter of the gaussian shape
        i += NGSRC; typename Tr::ft R = gauss_shape[i];

        // Scale u1 and u2
        u1 *= R*scale_uv;
        v1 *= scale_uv;

        typename Tr::ft exp = Po::exp(-(u1*u1 + v1*v1));
        real *= exp; imag *= exp;
    }

    {
        int j = threadIdx.z*BLOCKDIMY + threadIdx.y;
        typename Tr::ft diff = l[threadIdx.x]-ld_p[j];
        typename Tr::ft E_p = diff*diff;
        diff = m[threadIdx.x]-md_p[j];
        E_p += diff*diff;
        E_p = Po::sqrt(E_p);
        E_p *= E_beam_width*1e-9*wave[threadIdx.y];
        E_p = Po::min(E_p, E_beam_clip);
        E_p = Po::cos(E_p);
        E_p = E_p*E_p*E_p;
        real *= E_p; imag *= E_p;
    }

    {
        int j = threadIdx.z*BLOCKDIMY + threadIdx.y;
        typename Tr::ft diff = l[threadIdx.x]-ld_q[j];
        typename Tr::ft E_q = diff*diff;
        diff = m[threadIdx.x]-md_q[j];
        E_q += diff*diff;
        E_q = Po::sqrt(E_q);
        E_q *= E_beam_width*1e-9*wave[threadIdx.y];
        E_q = Po::min(E_q, E_beam_clip);
        E_q = Po::cos(E_q);
        E_q = E_q*E_q*E_q;
        real /= E_q; imag /= E_q;
    }

    // Index into the jones matrices
    i = (BL*NCHAN*NTIME + CHAN*NTIME + TIME)*NSRC + SRC;
    // TIME*NSRC + SRC;
    j = threadIdx.y*BLOCKDIMX + threadIdx.x;

    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = fI+fQ, b=0.0, c=real, d = imag
    jones[i]=Po::make_ct(
        (fI[j]+fQ[j])*real - 0.0*imag,
        (fI[j]+fQ[j])*imag + 0.0*real);

    // a=fU, b=fV, c=real, d = imag 
    i += NBL*NCHAN*NTIME*NSRC;
    jones[i]=Po::make_ct(
        fU[j]*real - fV[j]*imag,
        fU[j]*imag + fV[j]*real);

    // a=fU, b=-fV, c=real, d = imag 
    i += NBL*NCHAN*NTIME*NSRC;
    jones[i]=Po::make_ct(
        fU[j]*real - -fV[j]*imag,
        fU[j]*imag + -fV[j]*real);

    // a=fI-fQ, b=0.0, c=real, d = imag 
    i += NBL*NCHAN*NTIME*NSRC;
    jones[i]=Po::make_ct(
        (fI[j]-fQ[j])*real - 0.0*imag,
        (fI[j]-fQ[j])*imag + 0.0*real);
}

extern "C" {

// Macro that stamps out different kernels, depending on
// - whether we're handling floats or doubles
// - point sources or gaussian sources
// Arguments
// - ft: The floating point type. Should be float/double.
// - ct: The complex type. Should be float2/double2.
// - gaussian: boolean indicating whether we're handling gaussians or point sources
// - symbol: p or g depending on whether we're handling point or gaussian sources.
// - NISRC: Number of internal sources that the kernel operates on.
//          Set to NPSRC or NGSRC depending on point/gaussian sources.
// - src_offset: Number of sources to offset within the,
//          LM, brightness and jones matrices. Generally 0 for
//          point sources and NPSRC for gaussian sources.
//            
#define stamp_EBK_fn(ft, ct, gaussian, symbol, NISRC, SRC_OFFSET) \
__global__ void \
rime_jones_ ## symbol ## EBK_ ## ft( \
    ft * UVW, \
    ft * LM, \
    ft * brightness, \
    ft * gauss_shape, \
    ft * wavelength, \
    ft * point_error, \
    int * ant_pairs, \
    ct * jones, \
    ft ref_wave, \
    ft E_beam_width, \
    ft E_beam_clip) \
{ \
    rime_jones_EBK_impl<ft,NISRC,gaussian>( \
        UVW, \
        LM + SRC_OFFSET, \
        brightness + SRC_OFFSET, \
        gauss_shape, wavelength, \
        point_error, ant_pairs, \
        jones + SRC_OFFSET, \
        ref_wave, E_beam_width, E_beam_clip); \
}

stamp_EBK_fn(float,float2,false, p, NPSRC, 0)
stamp_EBK_fn(double,double2,false, p, NPSRC, 0)
stamp_EBK_fn(float,float2,true, g, NGSRC, NPSRC)
stamp_EBK_fn(double,double2,true, g, NGSRC, NPSRC)

} // extern "C" {
""")

class RimeEBK(Node):
    def __init__(self, gaussian=False):
        super(RimeEBK, self).__init__()
        self.gaussian = gaussian

    def initialise(self, shared_data):
        sd = shared_data

        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS
        D.update(sd.get_params())

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo'],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        # Choose our kernel form a 2x2 matrix
        # of choices.
        # - float or double
        # - point or gaussian sources?
        if sd.is_float() is True:
            kname = 'rime_jones_pEBK_float' \
                if not self.gaussian \
                else 'rime_jones_gEBK_float'
        else:
            kname = 'rime_jones_pEBK_double' \
                if not self.gaussian \
                else 'rime_jones_gEBK_double'

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, shared_data):
        pass

    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data
        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS

        # Are we dealing with point sources or gaussian sources?
        nisrc = sd.ngsrc if self.gaussian else sd.npsrc

        srcs_per_block = D['BLOCKDIMX'] if nisrc > D['BLOCKDIMX'] else nisrc
        time_chans_per_block = D['BLOCKDIMY']
        baselines_per_block = D['BLOCKDIMZ'] if sd.nbl > D['BLOCKDIMZ'] else sd.nbl

        src_blocks = self.blocks_required(nisrc,srcs_per_block)
        time_chan_blocks = sd.ntime*sd.nchan
        baseline_blocks = self.blocks_required(sd.nbl,baselines_per_block)

        return {
            'block' : (srcs_per_block,time_chans_per_block,baselines_per_block),
            'grid'  : (src_blocks,time_chan_blocks,baseline_blocks)
        }

    def execute(self, shared_data):
        sd = shared_data

        # Setup our kernel call, depending on whether we're
        # doing point or gaussian sources, and whether there
        # are indeed sources for us to compute!
        if not self.gaussian and sd.npsrc > 0:
            # Note the null pointer passed for gauss_shape here.
            self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu, np.intp(0),
                sd.wavelength_gpu, sd.point_errors_gpu, sd.ant_pairs_gpu, sd.jones_gpu,
                sd.ref_wave, sd.beam_width, sd.E_beam_clip,
    			**self.get_kernel_params(sd))
        elif self.gaussian and sd.ngsrc > 0:
            self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu, sd.gauss_shape_gpu,
                sd.wavelength_gpu, sd.point_errors_gpu, sd.ant_pairs_gpu, sd.jones_gpu,
                sd.ref_wave, sd.beam_width, sd.E_beam_clip,
                **self.get_kernel_params(sd))

    def post_execution(self, shared_data):
        pass