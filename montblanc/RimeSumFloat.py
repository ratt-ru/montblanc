import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

FLOAT_KERNEL = """
#include \"math_constants.h\"

extern __shared__ float2 smem_f[];

__global__
void rime_jones_sum_float(
    float2 * jones,
    float2 * visibilities,
    int nvis, int nsrc)
{
    // Our data space is a 4D matrix of BL x CHAN x TIME x SRC
    // V is the visibility
    int V = blockIdx.x*blockDim.x + threadIdx.x;

    if(V >= nvis)
        return;

    int J = V*nsrc;

    float2 sum = make_float2(0.0f, 0.0f);
	    
    for(int SRC=0; SRC<nsrc; ++SRC)
    {
    	float2 value = jones[J+SRC];
    	sum.x += value.x; sum.y += value.y;
    }

    visibilities[V] = sum;

    J += nvis*nsrc; V += nvis;
    sum = make_float2(0.0f, 0.0f);

    for(int SRC=0; SRC<nsrc; ++SRC)
    {
    	float2 value = jones[J+SRC];
    	sum.x += value.x; sum.y += value.y;
    }

    visibilities[V] = sum;

    J += nvis*nsrc; V += nvis;
    sum = make_float2(0.0f, 0.0f);

    for(int SRC=0; SRC<nsrc; ++SRC)
    {
    	float2 value = jones[J+SRC];
    	sum.x += value.x; sum.y += value.y;
    }

    visibilities[V] = sum;

    J += nvis*nsrc; V += nvis;
    sum = make_float2(0.0f, 0.0f);

    for(int SRC=0; SRC<nsrc; ++SRC)
    {
    	float2 value = jones[J+SRC];
    	sum.x += value.x; sum.y += value.y;
    }

    visibilities[V] = sum;

    #undef REFWAVE
}
"""

class RimeSumFloat(Node):
    def __init__(self):
        super(RimeSumFloat, self).__init__()

    def initialise(self, shared_data):
        self.mod = SourceModule(FLOAT_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_sum_float')

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        vis_per_block = 128 if sd.nvis > 128 else sd.nvis
        vis_blocks = (sd.nvis + vis_per_block - 1) / vis_per_block

        return {
            'block' : (vis_per_block,1,1),
            'grid'  : (vis_blocks,1,1),
            'shared' : 16*vis_per_block*np.dtype(sd.ct).itemsize }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.jones_gpu, sd.vis_gpu,
            np.int32(sd.nbl*sd.nchan*sd.ntime), np.int32(sd.nsrc),
            **self.get_kernel_params(sd))       

    def post_execution(self, shared_data):
        pass
