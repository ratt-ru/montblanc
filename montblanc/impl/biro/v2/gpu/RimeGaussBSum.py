import numpy as np
import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 8,    # Number of timesteps
    'BLOCKDIMZ' : 1,    # Number of baselines
    'maxregs'   : 40    # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 4,    # Number of timesteps
    'BLOCKDIMZ' : 1,    # Number of baselines
    'maxregs'   : 64    # Maximum number of registers
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

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

#define REFWAVE ${ref_wave}
#define BEAMCLIP ${beam_clip}
#define BEAMWIDTH ${beam_width}
#define GAUSS_SCALE ${gauss_scale}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_gauss_B_sum_impl(
    typename Tr::ft * uvw,
    typename Tr::ft * brightness,
    typename Tr::ft * gauss_shape,
    typename Tr::ft * wavelength,
    int * ant_pairs,
    typename Tr::ct * jones_EK_scalar,
    typename Tr::ct * visibilities,
    typename Tr::ct * output)
{
    int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= NBL || TIME >= NTIME || CHAN >= NCHAN)
        return;   

    __shared__ T u[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T v[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T w[BLOCKDIMZ][BLOCKDIMY];

    __shared__ T el[1];
    __shared__ T em[1];
    __shared__ T eR[1];

    __shared__ T I[BLOCKDIMY];
    __shared__ T Q[BLOCKDIMY];
    __shared__ T U[BLOCKDIMY];
    __shared__ T V[BLOCKDIMY];

    __shared__ T wl[BLOCKDIMX];

    int i;

    // Figure out the antenna pairs
    i = BL*NTIME + TIME; int ANT1 = ant_pairs[i];
    i += NBL*NTIME;      int ANT2 = ant_pairs[i];

    // UVW coordinates vary by baseline and time, but not channel
    if(threadIdx.x == 0)
    {
        // UVW, calculated from u_pq = u_p - u_q
        // baseline x BLOCKDIMY + TIME;
        i = ANT1*NTIME + TIME; u[threadIdx.z][threadIdx.y] = uvw[i];
        i += NA*NTIME;         v[threadIdx.z][threadIdx.y] = uvw[i];
        i += NA*NTIME;         w[threadIdx.z][threadIdx.y] = uvw[i];

        i = ANT2*NTIME + TIME; u[threadIdx.z][threadIdx.y] -= uvw[i];
        i += NA*NTIME;         v[threadIdx.z][threadIdx.y] -= uvw[i];
        i += NA*NTIME;         w[threadIdx.z][threadIdx.y] -= uvw[i];
    }

    // Wavelength varies by channel, but not baseline and time
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { wl[threadIdx.x] = wavelength[CHAN]; }

    typename Tr::ct Isum = Po::make_ct(0.0, 0.0);
    typename Tr::ct Qsum = Po::make_ct(0.0, 0.0);
    typename Tr::ct Usum = Po::make_ct(0.0, 0.0);
    typename Tr::ct Vsum = Po::make_ct(0.0, 0.0);

    for(int SRC=0;SRC<NPSRC;++SRC)
    {
        // The following loads effect the global load efficiency.

        // brightness varies by time (and source), not baseline or channel
        if(threadIdx.x == 0 && threadIdx.z == 0)
        {
            i = TIME*NSRC + SRC;  I[threadIdx.y] = brightness[i];
            i += NTIME*NSRC;      Q[threadIdx.y] = brightness[i];
            i += NTIME*NSRC;      U[threadIdx.y] = brightness[i];
            i += NTIME*NSRC;      V[threadIdx.y] = brightness[i];
        }

        __syncthreads();

        // Get the complex scalars for antenna one and multiply
        // in the exponent term
        i = (ANT1*NTIME*NSRC + TIME*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_one = jones_EK_scalar[i];
        // Get the complex scalars for antenna two
        i = (ANT2*NTIME*NSRC + TIME*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_two = jones_EK_scalar[i];

        // Divide the first antenna scalar by the second
        T div = (ant_two.x*ant_two.x + ant_two.y*ant_two.y);
        typename Tr::ct value = Po::make_ct(
            (ant_one.x*ant_two.x + ant_one.y*ant_two.y)/div,
            (ant_one.y*ant_two.x - ant_one.x*ant_two.y)/div);

        i = (BL*NTIME*NSRC + TIME*NSRC + SRC)*NCHAN + CHAN;
        output[i] = value;

        Isum.x += (I[threadIdx.y]+Q[threadIdx.y])*value.x + 0.0*value.y;
        Isum.y += (I[threadIdx.y]+Q[threadIdx.y])*value.y + 0.0*value.x;

        Qsum.x += (I[threadIdx.y]-Q[threadIdx.y])*value.x - 0.0*value.y;
        Qsum.y += (I[threadIdx.y]-Q[threadIdx.y])*value.y - 0.0*value.x;

        Usum.x += U[threadIdx.y]*value.x - -V[threadIdx.y]*value.y;
        Usum.y += U[threadIdx.y]*value.y + -V[threadIdx.y]*value.x;

        Vsum.x += U[threadIdx.y]*value.x - V[threadIdx.y]*value.y;
        Vsum.y += U[threadIdx.y]*value.y + V[threadIdx.y]*value.x;
    }

    for(int SRC=NPSRC;SRC<NSRC;++SRC)
    {
        // The following loads effect the global load efficiency.

        // brightness varies by time (and source), not baseline or channel
        if(threadIdx.x == 0 && threadIdx.z == 0)
        {
            i = TIME*NSRC + SRC;  I[threadIdx.y] = brightness[i];
            i += NTIME*NSRC;      Q[threadIdx.y] = brightness[i];
            i += NTIME*NSRC;      U[threadIdx.y] = brightness[i];
            i += NTIME*NSRC;      V[threadIdx.y] = brightness[i];
        }

        // gaussian shape only varies by source. Shape parameters
        // thus apply to the entire block and we can load them with
        // only the first thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC-NPSRC; el[0] = gauss_shape[i];
            i += NGSRC;     em[0] = gauss_shape[i];
            i += NGSRC;     eR[0] = gauss_shape[i];
        }

        __syncthreads();

        // Calculate the gaussian
        T scale_uv = T(GAUSS_SCALE)/wl[threadIdx.x];

        T u1 = (u[threadIdx.z][threadIdx.y]*em[0] -
            v[threadIdx.z][threadIdx.y]*el[0])*eR[0]*scale_uv;
        T v1 = (u[threadIdx.z][threadIdx.y]*el[0] +
            v[threadIdx.z][threadIdx.y]*em[0])*scale_uv;
        T exp = Po::exp(-(u1*u1 +v1*v1));

        // Get the complex scalars for antenna one and multiply
        // in the exponent term
        i = (ANT1*NTIME*NSRC + TIME*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_one = jones_EK_scalar[i];
        ant_one.x *= exp; ant_one.y *= exp;
        // Get the complex scalars for antenna two
        i = (ANT2*NTIME*NSRC + TIME*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_two = jones_EK_scalar[i];

        // Divide the first antenna scalar by the second
        T div = (ant_two.x*ant_two.x + ant_two.y*ant_two.y);
        typename Tr::ct value = Po::make_ct(
            (ant_one.x*ant_two.x + ant_one.y*ant_two.y)/div,
            (ant_one.y*ant_two.x - ant_one.x*ant_two.y)/div);

        i = (BL*NTIME*NSRC + TIME*NSRC + SRC)*NCHAN + CHAN;
        output[i] = value;

        Isum.x += (I[threadIdx.y]+Q[threadIdx.y])*value.x + 0.0*value.y;
        Isum.y += (I[threadIdx.y]+Q[threadIdx.y])*value.y + 0.0*value.x;

        Qsum.x += (I[threadIdx.y]-Q[threadIdx.y])*value.x - 0.0*value.y;
        Qsum.y += (I[threadIdx.y]-Q[threadIdx.y])*value.y - 0.0*value.x;

        Usum.x += U[threadIdx.y]*value.x - -V[threadIdx.y]*value.y;
        Usum.y += U[threadIdx.y]*value.y + -V[threadIdx.y]*value.x;

        Vsum.x += U[threadIdx.y]*value.x - V[threadIdx.y]*value.y;
        Vsum.y += U[threadIdx.y]*value.y + V[threadIdx.y]*value.x;
    }


    i = BL*NTIME*NCHAN + TIME*NCHAN + CHAN;
    visibilities[i] = Isum;

    i += 3*NBL*NTIME*NCHAN;
    visibilities[i] = Qsum;

    i -= NBL*NTIME*NCHAN;
    visibilities[i] = Usum;

    i -= NBL*NTIME*NCHAN;
    visibilities[i] = Vsum;
}

extern "C" {

// Macro that stamps out different kernels, depending
// on whether we're handling floats or doubles
#define stamp_gauss_b_sum_fn(ft, ct) \
__global__ void \
rime_gauss_B_sum_ ## ft( \
    ft * uvw, \
    ft * brightness, \
    ft * gauss_shape, \
    ft * wavelength, \
    int * ant_pairs, \
    ct * jones_EK_scalar, \
    ct * visibilities, \
    ct * output) \
{ \
    rime_gauss_B_sum_impl<ft>(uvw, brightness, gauss_shape, \
        wavelength, ant_pairs, jones_EK_scalar, visibilities, output); \
}

stamp_gauss_b_sum_fn(float, float2)
stamp_gauss_b_sum_fn(double, double2)

} // extern "C" {
""")

class RimeGaussBSum(Node):
    def __init__(self):
        super(RimeGaussBSum, self).__init__()

    def initialise(self, shared_data):
        sd = shared_data

        D = sd.get_properties()
        D.update(FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS)

        regs = str(FLOAT_PARAMS['maxregs'] \
            if sd.is_float() else DOUBLE_PARAMS['maxregs'])

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        kname = 'rime_gauss_B_sum_float' \
            if sd.is_float() is True else \
            'rime_gauss_B_sum_double'

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, shared_data):
        pass

    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS

        chans_per_block = D['BLOCKDIMX'] if sd.nchan > D['BLOCKDIMX'] else sd.nchan
        times_per_block = D['BLOCKDIMY'] if sd.ntime > D['BLOCKDIMY'] else sd.ntime
        bl_per_block = D['BLOCKDIMZ'] if sd.nbl > D['BLOCKDIMZ'] else sd.nbl

        chan_blocks = self.blocks_required(sd.nchan, chans_per_block)
        time_blocks = self.blocks_required(sd.ntime, times_per_block)
        bl_blocks = self.blocks_required(sd.nbl, bl_per_block)

        return {
            'block' : (chans_per_block, times_per_block, bl_per_block),
            'grid'  : (chan_blocks, time_blocks, bl_blocks), 
        }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.uvw_gpu, sd.brightness_gpu, sd.gauss_shape_gpu, 
            sd.wavelength_gpu, sd.ant_pairs_gpu, sd.jones_scalar_gpu,
            sd.vis_gpu, sd.output_gpu,
            **self.get_kernel_params(sd))

    def post_execution(self, shared_data):
        pass
