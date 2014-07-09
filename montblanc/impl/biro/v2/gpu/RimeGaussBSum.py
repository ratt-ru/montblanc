import numpy as np
import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 8,    # Number of timesteps
    'BLOCKDIMZ' : 2,    # Number of antennas
    'maxregs'   : 32    # Maximum number of registers
}

# 44 registers results in some spillage into
# local memory
DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 8,    # Number of timesteps
    'BLOCKDIMZ' : 1,    # Number of antennas
    'maxregs'   : 40    # Maximum number of registers
}

KERNEL_TEMPLATE = string.Template("""
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>

#define NA ${na}
#define NBL ${nbl}
#define NCHAN ${nchan}
#define NTIME ${ntime}
#define NSRC ${nsrc}

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

#define REFWAVE ${ref_wave}
#define BEAMCLIP ${beam_clip}
#define BEAMWIDTH ${beam_width}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_gauss_B_sum_impl(
    typename Tr::ft * uvw,
    typename Tr::ft * brightness,
    typename Tr::ft * gauss_shape,
    int * ant_pairs,
    typename Tr::ct * jones_EK_scalar,
    typename Tr::ct * visibilities)
{
    int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= NBL || TIME >= NTIME || CHAN >= NCHAN)
        return;   
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
    int * ant_pairs, \
    ct * jones_EK_scalar, \
    ct * visibilities) \
{ \
    rime_gauss_B_sum_impl<ft>(uvw, brightness, gauss_shape, \
        ant_pairs, jones_EK_scalar, visibilities); \
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
            sd.ant_pairs_gpu, sd.jones_scalar_gpu, sd.vis_gpu,
            **self.get_kernel_params(sd))

    def post_execution(self, shared_data):
        pass
