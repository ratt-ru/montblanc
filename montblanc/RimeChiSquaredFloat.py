import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from montblanc.node import Node

KERNEL = """
// Shared memory pointers used by the kernels.

__global__
void rime_chi_squared_diff(
    float2 * data_vis,
    float2 * model_vis,
    float * output,
    float sigma_squared,
    int nbl, int nchan, int ntime)
{
    // data_vis and model_vis matrix have dimensions
    // 4 x BASELINE x CHAN x TIME.
    // We're computing (D - M)^2/sigma^2
    // output has dimension BASELINE x CHAN x TIME

    int i = (blockIdx.x*blockDim.x + threadIdx.x);
    int j = i;
    int stride = nbl*nchan*ntime;

    // Quit if we're outside the problem range
    if(i >= stride)
        { return; }

    float2 delta = data_vis[i]; float2 model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    float2 sum = make_float2(delta.x*delta.x, delta.y*delta.y);

    i += stride;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    sum.x += delta.x*delta.x; sum.y += delta.y*delta.y;

    i += stride;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    sum.x += delta.x*delta.x; sum.y += delta.y*delta.y;

    i += stride;
    delta = data_vis[i]; model = model_vis[i];
    delta.x -= model.x; delta.y -= model.y;
    sum.x += delta.x*delta.x; sum.y += delta.y*delta.y;

    output[j] = (sum.x + sum.y)/sigma_squared;
}
"""

class RimeChiSquaredFloat(Node):
    def __init__(self):
        super(RimeChiSquaredFloat, self).__init__()
 
    def initialise(self, shared_data):
        self.mod = SourceModule(KERNEL,options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_chi_squared_diff')

    def shutdown(self, shared_data):
        pass

    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        vis_per_block = 256 if sd.nvis > 256 else sd.nvis
        vis_blocks = (sd.nvis + vis_per_block - 1) / vis_per_block

        return {
            'block'  : (vis_per_block,1,1), \
            'grid'   : (vis_blocks,1,1) }
#            'shared' : 0 }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.vis_gpu, sd.bayes_data_gpu, \
            sd.chi_sqrd_result_gpu, sd.sigma_sqrd, \
            np.int32(sd.nbl), np.int32(sd.nchan), np.int32(sd.ntime), \
            **self.get_kernel_params(sd))
            
    def post_execution(self, shared_data):
        pass