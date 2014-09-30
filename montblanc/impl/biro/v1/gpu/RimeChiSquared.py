import numpy as np
import string
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 1024,
    'BLOCKDIMY' : 1,
    'BLOCKDIMZ' : 1
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 1024,
    'BLOCKDIMY' : 1,
    'BLOCKDIMZ' : 1
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

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

template <
    typename T,
    bool apply_weights=false,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_chi_squared_diff_impl(
    typename Tr::ct * data_vis,
    typename Tr::ct * model_vis,
    typename Tr::ft * weights,
    T * output)
{
    // data_vis and model_vis matrix have dimensions
    // 4 x BASELINE x CHAN x TIME.
    // We're computing (D - M)^2/sigma^2
    // output has dimension BASELINE x CHAN x TIME

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Quit if we're outside the problem range
    if(i >= NBL*NCHAN*NTIME)
        { return; }

    typename Tr::ct delta = data_vis[i];
    typename Tr::ct model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weights[i]; delta.x *= w; delta.y *= w; }
    typename Tr::ct sum = Po::make_ct(delta.x, delta.y);

    i += NBL*NCHAN*NTIME;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weights[i]; delta.x *= w; delta.y *= w; }
    sum.x += delta.x; sum.y += delta.y;

    i += NBL*NCHAN*NTIME;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weights[i]; delta.x *= w; delta.y *= w; }
    sum.x += delta.x; sum.y += delta.y;

    i += NBL*NCHAN*NTIME;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weights[i]; delta.x *= w; delta.y *= w; }
    sum.x += delta.x; sum.y += delta.y;

    output[blockIdx.x*blockDim.x + threadIdx.x] = sum.x + sum.y;
}

extern "C" {

// Macro that stamps out different kernels, depending on
// - whether we're handling floats or doubles
// - point sources or gaussian sources
// Arguments
// - ft: The floating point type. Should be float/double.
// - ct: The complex type. Should be float2/double2.
// - apply_weights: boolean indicating whether we're weighting our visibilities
// - symbol: u or w depending on whether we're handling unweighted/weighted visibilities.
//            
#define stamp_chi_sqrd_fn(ft, ct, apply_weights, symbol) \
__global__ void \
rime_chi_squared_ ## symbol ## diff_ ## ft( \
    ct * data_vis, \
    ct * model_vis, \
    ft * weights, \
    ft * output) \
{ \
    rime_chi_squared_diff_impl<ft, apply_weights>(data_vis,model_vis,weights,output); \
}

stamp_chi_sqrd_fn(float,float2,false,u)
stamp_chi_sqrd_fn(double,double2,false,u)
stamp_chi_sqrd_fn(float,float2,true,w)
stamp_chi_sqrd_fn(double,double2,true,w)

} // extern "C" {
""")

class RimeChiSquared(Node):
    def __init__(self, weight_vector=False):
        """
        Parameters:
        -----------
        weight_vector : boolean
            True if each value within the sum of a Chi Squared should
            be individually multiplied by a weight, and then summed.

            False if no weighting should be applied.
        """
        super(RimeChiSquared, self).__init__()
        self.weight_vector = weight_vector

    def initialise(self, solver, stream=None):
        slvr = solver

        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS
        D.update(slvr.get_properties())

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','--maxrregcount','32'],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        kname = 'rime_chi_squared_' + \
            ('w' if self.weight_vector else 'u') + \
            'diff_' + \
            ('float' if slvr.is_float() else 'double')

        #print 'kernel', kname

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, solver, stream=None):
        pass

    def pre_execution(self, solver, stream=None):
        pass

    def get_kernel_params(self, solver):
        slvr = solver
        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS

        vis_per_block = D['BLOCKDIMX'] if slvr.nvis > D['BLOCKDIMX'] else slvr.nvis
        vis_blocks = self.blocks_required(slvr.nvis,vis_per_block)

        return {
            'block'  : (vis_per_block,1,1), \
            'grid'   : (vis_blocks,1,1)
        }

    def execute(self, solver, stream=None):
        slvr = solver

        weight_vector_gpu = slvr.weight_vector_gpu if self.weight_vector is True \
            else np.intp(0)

        self.kernel(slvr.vis_gpu, slvr.bayes_data_gpu, weight_vector_gpu,
            slvr.chi_sqrd_result_gpu,
            **self.get_kernel_params(slvr))

        # If we're not catering for a weight vector,
        # call the simple reduction and divide by sigma squared.
        # Otherwise, call the more complicated reduction kernel that
        # internally divides by the noise vector
        gpu_sum = gpuarray.sum(slvr.chi_sqrd_result_gpu).get()

        if not self.weight_vector:
            slvr.set_X2(gpu_sum/slvr.sigma_sqrd)
        else:
            slvr.set_X2(gpu_sum)        
            
    def post_execution(self, solver, stream=None):
        pass