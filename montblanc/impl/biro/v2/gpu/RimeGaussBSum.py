import numpy as np
import string

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 8,    # Number of baselines
    'BLOCKDIMZ' : 1,    # Number of timesteps
    'maxregs'   : 48    # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 4,    # Number of baselines
    'BLOCKDIMZ' : 1,    # Number of timesteps
    'maxregs'   : 63    # Maximum number of registers
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

#define GAUSS_SCALE ${gauss_scale}

template <
    typename T,
    bool apply_weights=false,
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
    typename Tr::ft * weight_vector,
    typename Tr::ct * visibilities,
    typename Tr::ct * data_vis,
    typename Tr::ft * chi_sqrd_result)
{
    int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int BL = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= NBL || TIME >= NTIME || CHAN >= NCHAN)
        return;   

    __shared__ T u[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T v[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T w[BLOCKDIMZ][BLOCKDIMY];

    __shared__ T el[1];
    __shared__ T em[1];
    __shared__ T eR[1];

    __shared__ T I[BLOCKDIMZ];
    __shared__ T Q[BLOCKDIMZ];
    __shared__ T U[BLOCKDIMZ];
    __shared__ T V[BLOCKDIMZ];

    __shared__ T wl[BLOCKDIMX];

    int i;

    // Figure out the antenna pairs
    i = TIME*NBL + BL;   int ANT1 = ant_pairs[i];
    i += NBL*NTIME;      int ANT2 = ant_pairs[i];

    // UVW coordinates vary by baseline and time, but not channel
    if(threadIdx.x == 0)
    {
        // UVW, calculated from u_pq = u_p - u_q
        i = TIME*NA + ANT1;    u[threadIdx.z][threadIdx.y] = uvw[i];
        i += NA*NTIME;         v[threadIdx.z][threadIdx.y] = uvw[i];
        i += NA*NTIME;         w[threadIdx.z][threadIdx.y] = uvw[i];

        i = TIME*NA + ANT2;    u[threadIdx.z][threadIdx.y] -= uvw[i];
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
        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            i = TIME*NSRC + SRC;  I[threadIdx.z] = brightness[i];
            i += NTIME*NSRC;      Q[threadIdx.z] = brightness[i];
            i += NTIME*NSRC;      U[threadIdx.z] = brightness[i];
            i += NTIME*NSRC;      V[threadIdx.z] = brightness[i];
        }

        __syncthreads();

        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        i = (TIME*NA*NSRC + ANT2*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_two = jones_EK_scalar[i];
        // Get the complex scalar for antenna one and conjugate it
        i = (TIME*NA*NSRC + ANT1*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_one = jones_EK_scalar[i]; ant_one.y = -ant_one.y;

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // a = ant_two.x b=ant_two.y c=ant_one.x d = ant_one.y
        typename Tr::ct value = Po::make_ct(
            ant_two.x*ant_one.x - ant_two.y*ant_one.y,
            ant_two.x*ant_one.y + ant_two.y*ant_one.x);

        Isum.x += (I[threadIdx.z]+Q[threadIdx.z])*value.x + 0.0*value.y;
        Isum.y += (I[threadIdx.z]+Q[threadIdx.z])*value.y + 0.0*value.x;

        Qsum.x += (I[threadIdx.z]-Q[threadIdx.z])*value.x - 0.0*value.y;
        Qsum.y += (I[threadIdx.z]-Q[threadIdx.z])*value.y - 0.0*value.x;

        Usum.x += U[threadIdx.z]*value.x - -V[threadIdx.z]*value.y;
        Usum.y += U[threadIdx.z]*value.y + -V[threadIdx.z]*value.x;

        Vsum.x += U[threadIdx.z]*value.x - V[threadIdx.z]*value.y;
        Vsum.y += U[threadIdx.z]*value.y + V[threadIdx.z]*value.x;

        __syncthreads();
    }

    for(int SRC=NPSRC;SRC<NSRC;++SRC)
    {
        // The following loads effect the global load efficiency.

        // brightness varies by time (and source), not baseline or channel
        if(threadIdx.x == 0 && threadIdx.y == 0)
        {
            i = TIME*NSRC + SRC;  I[threadIdx.z] = brightness[i];
            i += NTIME*NSRC;      Q[threadIdx.z] = brightness[i];
            i += NTIME*NSRC;      U[threadIdx.z] = brightness[i];
            i += NTIME*NSRC;      V[threadIdx.z] = brightness[i];
        }

        // gaussian shape only varies by source. Shape parameters
        // thus apply to the entire block and we can load them with
        // only the first thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC-NPSRC;  el[0] = gauss_shape[i];
            i += NGSRC;     em[0] = gauss_shape[i];
            i += NGSRC;     eR[0] = gauss_shape[i];
        }

        __syncthreads();

        T u1 = u[threadIdx.z][threadIdx.y]*em[0] - v[threadIdx.z][threadIdx.y]*el[0];
        u1 *= T(GAUSS_SCALE)/wl[threadIdx.x];
        u1 *= eR[0];
        T v1 = u[threadIdx.z][threadIdx.y]*el[0] + v[threadIdx.z][threadIdx.y]*em[0];
        v1 *= T(GAUSS_SCALE)/wl[threadIdx.x];
        T exp = Po::exp(-(u1*u1 +v1*v1));

        // Get the complex scalar for antenna two and multiply
        // in the exponent term
        i = (TIME*NA*NSRC + ANT2*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_two = jones_EK_scalar[i];
        ant_two.x *= exp; ant_two.y *= exp;
        // Get the complex scalar for antenna one and conjugate it
        i = (TIME*NA*NSRC + ANT1*NSRC + SRC)*NCHAN + CHAN;
        typename Tr::ct ant_one = jones_EK_scalar[i]; ant_one.y = -ant_one.y;

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // a = ant_two.x b=ant_two.y c=ant_one.x d = ant_one.y
        typename Tr::ct value = Po::make_ct(
            ant_two.x*ant_one.x - ant_two.y*ant_one.y,
            ant_two.x*ant_one.y + ant_two.y*ant_one.x);
  
        Isum.x += (I[threadIdx.z]+Q[threadIdx.z])*value.x + 0.0*value.y;
        Isum.y += (I[threadIdx.z]+Q[threadIdx.z])*value.y + 0.0*value.x;

        Qsum.x += (I[threadIdx.z]-Q[threadIdx.z])*value.x - 0.0*value.y;
        Qsum.y += (I[threadIdx.z]-Q[threadIdx.z])*value.y - 0.0*value.x;

        Usum.x += U[threadIdx.z]*value.x - -V[threadIdx.z]*value.y;
        Usum.y += U[threadIdx.z]*value.y + -V[threadIdx.z]*value.x;

        Vsum.x += U[threadIdx.z]*value.x - V[threadIdx.z]*value.y;
        Vsum.y += U[threadIdx.z]*value.y + V[threadIdx.z]*value.x;

        __syncthreads();
    }

    i = (TIME*NBL + BL)*NCHAN + CHAN;
    visibilities[i] = Isum;
    typename Tr::ct delta = data_vis[i];
    delta.x -= Isum.x; delta.y -= Isum.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weight_vector[i]; delta.x *= w; delta.y *= w; }
    Isum.x = delta.x; Isum.y = delta.y;

    i += 3*NTIME*NBL*NCHAN;
    visibilities[i] = Qsum;
    delta = data_vis[i];
    delta.x -= Qsum.x; delta.y -= Qsum.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weight_vector[i]; delta.x *= w; delta.y *= w; }
    Isum.x += delta.x; Isum.y += delta.y;

    i -= NTIME*NBL*NCHAN;
    visibilities[i] = Usum;
    delta = data_vis[i];
    delta.x -= Usum.x; delta.y -= Usum.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weight_vector[i]; delta.x *= w; delta.y *= w; }
    Isum.x += delta.x; Isum.y += delta.y;

    i -= NTIME*NBL*NCHAN;
    visibilities[i] = Vsum;
    delta = data_vis[i];
    delta.x -= Vsum.x; delta.y -= Vsum.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weight_vector[i]; delta.x *= w; delta.y *= w; }
    Isum.x += delta.x; Isum.y += delta.y;

    i = (TIME*NBL + BL)*NCHAN + CHAN;    
    chi_sqrd_result[i] = Isum.x + Isum.y;
}

extern "C" {

// Macro that stamps out different kernels, depending
// on whether we're handling floats or doubles
// Arguments
// - ft: The floating point type. Should be float/double.
// - ct: The complex type. Should be float2/double2.
// - apply_weights: boolean indicating whether we're weighting our visibilities
// - symbol: u or w depending on whether we're handling unweighted/weighted visibilities.

#define stamp_gauss_b_sum_fn(ft, ct, apply_weights, symbol) \
__global__ void \
rime_gauss_B_sum_ ## symbol ## chi_ ## ft( \
    ft * uvw, \
    ft * brightness, \
    ft * gauss_shape, \
    ft * wavelength, \
    int * ant_pairs, \
    ct * jones_EK_scalar, \
    ft * weight_vector, \
    ct * visibilities, \
    ct * data_vis, \
    ft * chi_sqrd_result) \
{ \
    rime_gauss_B_sum_impl<ft, apply_weights>(uvw, brightness, gauss_shape, \
        wavelength, ant_pairs, jones_EK_scalar, \
        weight_vector, visibilities, data_vis, \
        chi_sqrd_result); \
}

stamp_gauss_b_sum_fn(float, float2, false, u)
stamp_gauss_b_sum_fn(float, float2, true, w)
stamp_gauss_b_sum_fn(double, double2, false, u)
stamp_gauss_b_sum_fn(double, double2, true, w)

} // extern "C" {
""")

class RimeGaussBSum(Node):
    def __init__(self, weight_vector=False):
        super(RimeGaussBSum, self).__init__()
        self.weight_vector = weight_vector

    def initialise(self, solver, stream=None):
        slvr = solver

        D = slvr.get_properties()
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        regs = str(FLOAT_PARAMS['maxregs'] \
            if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_gauss_B_sum_' + \
            ('w' if self.weight_vector else 'u') + 'chi_' + \
            ('float' if slvr.is_float() is True else 'double')

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, solver, stream=None):
        pass

    def pre_execution(self, solver, stream=None):
        pass

    def get_kernel_params(self, solver):
        slvr = solver

        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS

        chans_per_block = D['BLOCKDIMX'] if slvr.nchan > D['BLOCKDIMX'] else slvr.nchan
        bl_per_block = D['BLOCKDIMY'] if slvr.nbl > D['BLOCKDIMY'] else slvr.nbl
        times_per_block = D['BLOCKDIMZ'] if slvr.ntime > D['BLOCKDIMZ'] else slvr.ntime

        chan_blocks = self.blocks_required(slvr.nchan, chans_per_block)
        bl_blocks = self.blocks_required(slvr.nbl, bl_per_block)
        time_blocks = self.blocks_required(slvr.ntime, times_per_block)

        return {
            'block' : (chans_per_block, bl_per_block, times_per_block),
            'grid'  : (chan_blocks, bl_blocks, time_blocks), 
        }

    def execute(self, solver, stream=None):
        slvr = solver

        # The gaussian shape array can be empty if
        # no gaussian sources were specified.
        gauss = np.intp(0) if np.product(slvr.gauss_shape_shape) == 0 \
            else slvr.gauss_shape_gpu

        self.kernel(slvr.uvw_gpu, slvr.brightness_gpu, gauss, 
            slvr.wavelength_gpu, slvr.ant_pairs_gpu, slvr.jones_scalar_gpu,
            slvr.weight_vector_gpu,
            slvr.vis_gpu, slvr.bayes_data_gpu, slvr.chi_sqrd_result_gpu,
            **self.get_kernel_params(slvr))

        # Call the pycuda reduction kernel.
        # Divide by the single sigma squared value if a weight vector
        # is not required. Otherwise the kernel will incorporate the
        # individual sigma squared values into the sum
        gpu_sum = gpuarray.sum(slvr.chi_sqrd_result_gpu).get()

        if not self.weight_vector:
            slvr.set_X2(gpu_sum/slvr.sigma_sqrd)
        else:
            slvr.set_X2(gpu_sum)

    def post_execution(self, solver, stream=None):
        pass