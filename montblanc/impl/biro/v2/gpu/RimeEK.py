import numpy as np
import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 8,    # Number of antennas
    'BLOCKDIMZ' : 2,    # Number of timesteps
    'maxregs'   : 32    # Maximum number of registers
}

# 44 registers results in some spillage into
# local memory
DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 4,    # Number of antennas
    'BLOCKDIMZ' : 1,    # Number of timesteps
    'maxregs'   : 44    # Maximum number of registers
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
void rime_jones_EK_impl(
    T * uvw,
    T * lm,
    T * brightness,
    T * wavelength,
    T * point_errors,
    typename Tr::ct * jones_scalar)
{
	int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
	int TIME = blockIdx.z*blockDim.z + threadIdx.z;

	if(ANT >= NA || TIME >= NTIME || CHAN >= NCHAN)
		return;

	__shared__ T u[BLOCKDIMZ][BLOCKDIMY];
	__shared__ T v[BLOCKDIMZ][BLOCKDIMY];
	__shared__ T w[BLOCKDIMZ][BLOCKDIMY];

	// Shared Memory produces a faster kernel than
	// registers for some reason!
	__shared__ T l[1];
	__shared__ T m[1];

	__shared__ T ld[BLOCKDIMZ][BLOCKDIMY];
	__shared__ T md[BLOCKDIMZ][BLOCKDIMY];

	__shared__ T wl[BLOCKDIMX];

	int i;

	// UVW coordinates vary by antenna and time, but not channel
	if(threadIdx.x == 0)
	{
		i = TIME*NA + ANT;
		u[threadIdx.z][threadIdx.y] = uvw[i];
		ld[threadIdx.z][threadIdx.y] = point_errors[i];
		i += NA*NTIME;
		v[threadIdx.z][threadIdx.y] = uvw[i];
		md[threadIdx.z][threadIdx.y] = point_errors[i];
		i += NA*NTIME;
		w[threadIdx.z][threadIdx.y] = uvw[i];
	}

	// Wavelengths vary by channel, not by time and antenna
	if(threadIdx.y == 0 && threadIdx.z == 0)
		{ wl[threadIdx.x] = wavelength[CHAN]; }

	for(int SRC=0;SRC<NSRC;++SRC)
	{
		// LM coordinates vary only by source, not antenna and time
		if(threadIdx.x == 0 && threadIdx.y == 0)
		{
			i = SRC;   l[0] = lm[i];
			i += NSRC; m[0] = lm[i];
		}

		__syncthreads();

		// Calculate the phase term for this antenna
		T phase = Po::sqrt(T(1.0) - l[0]*l[0] - m[0]*m[0]) - T(1.0);

		phase = w[threadIdx.z][threadIdx.y]*phase
			+ v[threadIdx.z][threadIdx.y]*m[0]
			+ u[threadIdx.z][threadIdx.y]*l[0];

	    phase *= (T(2.0) * Tr::cuda_pi / wl[threadIdx.x]);

		T real, imag;
		Po::sincos(phase, &imag, &real);

		// Calculate the beam term for this antenna
		T diff = l[0] - ld[threadIdx.z][threadIdx.y];
		T E = diff*diff;
		diff = m[0] - md[threadIdx.z][threadIdx.y];
		E += diff*diff;
		E = Po::sqrt(E);
		E *= T(BEAMWIDTH*1e-9)*wl[threadIdx.x];
		E = Po::min(E, T(BEAMCLIP));
		E = Po::cos(E);
		E = E*E*E;

		// Write out the phase and beam values multiplied together
        i = (TIME*NA*NSRC + ANT*NSRC + SRC)*NCHAN + CHAN;
		jones_scalar[i] = Po::make_ct(real*E, imag*E);
        __syncthreads();
	}
}

extern "C" {

__global__ void rime_jones_EK_float(
    float * UVW,
    float * LM,
    float * brightness,
    float * wavelength,
    float * point_errors,
    float2 * jones)
{
    rime_jones_EK_impl<float>(UVW, LM, brightness, wavelength,
        point_errors, jones);
}


__global__ void rime_jones_EK_double(
    double * UVW,
    double * LM,
    double * brightness,
    double * wavelength,
    double * point_errors,
    double2 * jones)
{
    rime_jones_EK_impl<double>(UVW, LM, brightness, wavelength,
        point_errors, jones);
}

} // extern "C" {
""")

class RimeEK(Node):
    def __init__(self):
        super(RimeEK, self).__init__()

    def initialise(self, solver):
        slvr = solver

        D = slvr.get_properties()
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        regs = str(FLOAT_PARAMS['maxregs'] \
        	if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        kname = 'rime_jones_EK_float' \
            if slvr.is_float() is True else \
            'rime_jones_EK_double'

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, solver):
        pass

    def pre_execution(self, solver):
        pass

    def get_kernel_params(self, solver):
        slvr = solver

        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS

        chans_per_block = D['BLOCKDIMX'] if slvr.nchan > D['BLOCKDIMX'] else slvr.nchan
        ants_per_block = D['BLOCKDIMY'] if slvr.na > D['BLOCKDIMY'] else slvr.na
        times_per_block = D['BLOCKDIMZ'] if slvr.ntime > D['BLOCKDIMZ'] else slvr.ntime

        chan_blocks = self.blocks_required(slvr.nchan, chans_per_block)
        ant_blocks = self.blocks_required(slvr.na, ants_per_block)
        time_blocks = self.blocks_required(slvr.ntime, times_per_block)

        return {
            'block' : (chans_per_block, ants_per_block, times_per_block),
            'grid'  : (chan_blocks, ant_blocks, time_blocks), 
        }

    def execute(self, solver):
        slvr = solver

        self.kernel(slvr.uvw_gpu, slvr.lm_gpu, slvr.brightness_gpu,
            slvr.wavelength_gpu, slvr.point_errors_gpu, slvr.jones_scalar_gpu,
            **self.get_kernel_params(slvr))

    def post_execution(self, solver):
        pass
