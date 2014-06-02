import string
from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 256,
    'BLOCKDIMY' : 1,
    'BLOCKDIMZ' : 1
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 256,
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
#define NSRC ${nsrc}

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_chi_squared_diff_impl(
    typename Tr::ct * data_vis,
    typename Tr::ct * model_vis,
    T * output)
{
    // data_vis and model_vis matrix have dimensions
    // 4 x BASELINE x CHAN x TIME.
    // We're computing (D - M)^2/sigma^2
    // output has dimension BASELINE x CHAN x TIME

    int i = (blockIdx.x*blockDim.x + threadIdx.x);
    int j = i;

    // Quit if we're outside the problem range
    if(i >= NBL*NCHAN*NTIME)
        { return; }

    typename Tr::ct delta = data_vis[i];
    typename Tr::ct model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    typename Tr::ct sum = Po::make_ct(delta.x*delta.x, delta.y*delta.y);

    i += NBL*NCHAN*NTIME;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    sum.x += delta.x*delta.x; sum.y += delta.y*delta.y;

    i += NBL*NCHAN*NTIME;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    sum.x += delta.x*delta.x; sum.y += delta.y*delta.y;

    i += NBL*NCHAN*NTIME;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    sum.x += delta.x*delta.x; sum.y += delta.y*delta.y;

    output[j] = (sum.x + sum.y);
}

extern "C" {

__global__
void rime_chi_squared_diff_float(
    float2 * data_vis,
    float2 * model_vis,
    float * output)
{
    rime_chi_squared_diff_impl(data_vis, model_vis, output);
}

__global__
void rime_chi_squared_diff_double(
    double2 * data_vis,
    double2 * model_vis,
    double * output)
{
    rime_chi_squared_diff_impl(data_vis, model_vis, output);
}

} // extern "C" {
""")

class RimeChiSquared(Node):
    def __init__(self):
        super(RimeChiSquared, self).__init__()
 
    def initialise(self, shared_data):
        sd = shared_data

        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS
        D.update(sd.get_params())

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-keep'],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        kname = 'rime_chi_squared_diff_float' \
            if sd.is_float() is True else \
            'rime_chi_squared_diff_double'

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, shared_data):
        pass

    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data
        D = FLOAT_PARAMS if sd.is_float() else DOUBLE_PARAMS

        vis_per_block = D['BLOCKDIMX'] if sd.nvis > D['BLOCKDIMX'] else sd.nvis
        vis_blocks = self.blocks_required(sd.nvis,vis_per_block)

        return {
            'block'  : (vis_per_block,1,1), \
            'grid'   : (vis_blocks,1,1)
        }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.vis_gpu, sd.bayes_data_gpu,
            sd.chi_sqrd_result_gpu,
            **self.get_kernel_params(sd))
            
    def post_execution(self, shared_data):
        pass