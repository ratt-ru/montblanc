import numpy as np
import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,
    'BLOCKDIMY' : 8,
    'BLOCKDIMZ' : 2
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,
    'BLOCKDIMY' : 4,
    'BLOCKDIMZ' : 1
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
    typename Tr::ft * uvw,
    typename Tr::ft * lm,
    typename Tr::ft * brightness,
    typename Tr::ft * wavelength,
    typename Tr::ft * point_errors,
    typename Tr::ct * jones_scalar)
{
	int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
	int TIME = blockIdx.y*blockDim.y + threadIdx.y;
	int ANT = blockIdx.z*blockDim.z + threadIdx.z;

	if(ANT >= NA || TIME >= NTIME || CHAN >= NCHAN)
		return;

	__shared__ typename Tr::ft u[BLOCKDIMZ][BLOCKDIMY];
	__shared__ typename Tr::ft v[BLOCKDIMZ][BLOCKDIMY];
	__shared__ typename Tr::ft w[BLOCKDIMZ][BLOCKDIMY];

	// Shared Memory produces a faster kernel than
	// registers for some reason!
	__shared__ typename Tr::ft l[1];
	__shared__ typename Tr::ft m[1];

	__shared__ typename Tr::ft ld[BLOCKDIMZ][BLOCKDIMY];
	__shared__ typename Tr::ft md[BLOCKDIMZ][BLOCKDIMY];

	__shared__ typename Tr::ft a[BLOCKDIMY];

	int i;

	// UVW coordinates vary by antenna and time, but not channel
	if(threadIdx.x == 0)
	{
		i = ANT*NTIME + TIME;
		u[threadIdx.z][threadIdx.y] = uvw[i];
		ld[threadIdx.z][threadIdx.y] = point_errors[i];
		i += NA*NTIME;
		v[threadIdx.z][threadIdx.y] = uvw[i];
		md[threadIdx.z][threadIdx.y] = point_errors[i];
		i += NA*NTIME;
		w[threadIdx.z][threadIdx.y] = uvw[i];
	}

	for(int SRC=0;SRC<NSRC;++SRC)
	{
		// LM coordinates vary only by source, not antenna and time
		if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		{
			i = SRC;   l[0] = lm[i];
			i += NSRC; m[0] = lm[i];
		}

		// brightness varies by time (and source), not baseline or channel
		if(threadIdx.x == 0 && threadIdx.z == 0)
		{
			i = TIME*NSRC + SRC + 4*NTIME*NSRC;
			a[threadIdx.y] = brightness[i];
		}

		__syncthreads();

		typename Tr::ft wl = wavelength[CHAN];

		// Calculate the beam term for this antenna
		typename Tr::ft diff = l[0] - ld[threadIdx.z][threadIdx.y];
		typename Tr::ft E = diff*diff;
		diff = m[0] - md[threadIdx.z][threadIdx.y];
		E += diff*diff;
		E = Po::sqrt(E);
		E *= BEAMWIDTH*1e-9*wl;
		E = Po::min(E, BEAMCLIP);
		E = Po::cos(E);
		// This is folded into phase further down
		//E = E*E*E;

		// Calculate the phase term for this antenna
		typename Tr::ft phase = 1.0 - l[0]*l[0];
		phase -= m[0]*m[0];
		phase = Po::sqrt(phase) - 1.0;

		phase *= w[threadIdx.z][threadIdx.y];
		phase += v[threadIdx.z][threadIdx.y]*m[0];
		phase += u[threadIdx.z][threadIdx.y]*l[0];

	    phase *= (2. * Tr::cuda_pi / wl);

		typename Tr::ft real, imag;
		Po::sincos(phase, &imag, &real);

		phase = Po::pow(REFWAVE/wl, a[threadIdx.y])*E*E*E;
		real *= phase; imag *= phase;

		// Write out the phase term.
		i = (ANT*NTIME*NSRC + TIME*NSRC + SRC)*NCHAN + CHAN;
		jones_scalar[i] = Po::make_ct(real, imag);
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

    def initialise(self, shared_data):
        sd = shared_data

        D = sd.get_properties()
        D.update(FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS)

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

        chans_per_block = D['BLOCKDIMX'] if sd.nchan > D['BLOCKDIMX'] else sd.nchan
        times_per_block = D['BLOCKDIMY'] if sd.ntime > D['BLOCKDIMY'] else sd.ntime
        ants_per_block = D['BLOCKDIMZ'] if sd.na > D['BLOCKDIMZ'] else sd.na

        chan_blocks = self.blocks_required(sd.nchan, chans_per_block)
        time_blocks = self.blocks_required(sd.ntime, times_per_block)
        ant_blocks = self.blocks_required(sd.na, ants_per_block)

        return {
            'block' : (chans_per_block, times_per_block, ants_per_block),
            'grid'  : (chan_blocks, time_blocks, ant_blocks), 
        }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.uvw_gpu, sd.lm_gpu, sd.brightness_gpu,
            sd.wavelength_gpu, sd.point_errors_gpu, sd.jones_scalar_gpu,
            **self.get_kernel_params(sd))

    def post_execution(self, shared_data):
        pass
