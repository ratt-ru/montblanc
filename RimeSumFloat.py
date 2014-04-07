import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

FLOAT_KERNEL = """
#include \"math_constants.h\"

extern __shared__ float smem_f[];

// Based on OSKAR's implementation of the RIME K term.
// Baseline on the x dimension, source on the y dimension
__global__
void rime_jones_sum_float(
    float2 * jones,
    float2 * visibilities,
    int nsrc, int nbl, int nchan, int ntime)
{
    // Our data space is a 4D matrix of BL x CHAN x TIME x SRC

    // Baseline, Channel and Time indices
    int TIME = blockIdx.x*blockDim.x + threadIdx.x;
    int CHAN = blockIdx.y*blockDim.y + threadIdx.y;
    int BL = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= nbl || CHAN >= nchan || TIME >= ntime)
        return;

    int J = BL*nchan*ntime*nsrc + CHAN*ntime*nsrc + TIME*nsrc;
    int V = BL*nchan*ntime + CHAN*ntime + TIME;

    float2 sum = make_float2(0.0f, 0.0f);
    float2 c = make_float2(0.0f, 0.0f);
	    
    for(int SRC=0; SRC<nsrc; ++SRC)
    {
        float2 y = jones[J+SRC];
        y.x -= c.x; y.y -= c.y;
        float2 t = make_float2(sum.x + y.x,sum.y + y.y);
        sum = t;
    }

    visibilities[V] = sum;

    J += nbl*nchan*ntime*nsrc; V += nbl*nchan*ntime;
    sum = make_float2(0.0f, 0.0f);
    c = make_float2(0.0f, 0.0f);

    for(int SRC=0; SRC<nsrc; ++SRC)
    {
        float2 y = jones[J+SRC];
        y.x -= c.x; y.y -= c.y;
        float2 t = make_float2(sum.x + y.x,sum.y + y.y);
        sum = t;
    }

    visibilities[V] = sum;

    J += nbl*nchan*ntime*nsrc; V += nbl*nchan*ntime;
    sum = make_float2(0.0f, 0.0f);
    c = make_float2(0.0f, 0.0f);

    for(int SRC=0; SRC<nsrc; ++SRC)
    {
        float2 y = jones[J+SRC];
        y.x -= c.x; y.y -= c.y;
        float2 t = make_float2(sum.x + y.x,sum.y + y.y);
        sum = t;
    }

    visibilities[V] = sum;

    J += nbl*nchan*ntime*nsrc; V += nbl*nchan*ntime;
    sum = make_float2(0.0f, 0.0f);
    c = make_float2(0.0f, 0.0f);

    for(int SRC=0; SRC<nsrc; ++SRC)
    {
        float2 y = jones[J+SRC];
        y.x -= c.x; y.y -= c.y;
        float2 t = make_float2(sum.x + y.x,sum.y + y.y);
        sum = t;
    }

    visibilities[V] = sum;

    #undef REFWAVE
}
"""

class RimeSumFloat(Node):
    def __init__(self):
        super(RimeSumFloat, self).__init__()
        self.mod = SourceModule(FLOAT_KERNEL, options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_sum_float')

    def initialise(self, shared_data):
		pass	

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass

    def get_kernel_params(self, shared_data):
        sd = shared_data

        times_per_block = 16 if sd.ntime > 2 else sd.ntime
        chans_per_block = 8 if sd.nchan > 2 else sd.nchan
        baselines_per_block = 1 if sd.nbl > 16 else sd.nbl

        time_blocks = (sd.ntime + times_per_block - 1) / times_per_block
        chan_blocks = (sd.nchan + chans_per_block - 1) / chans_per_block
        baseline_blocks = (sd.nbl + baselines_per_block - 1)/ baselines_per_block

        return {
            'block' : (times_per_block,chans_per_block,baselines_per_block),
            'grid'  : (time_blocks,chan_blocks,baseline_blocks)} #,
#            'shared' : 0 }

    def execute(self, shared_data):
        sd = shared_data

        self.kernel(sd.jones_gpu, sd.vis_gpu,
            np.int32(sd.nsrc), np.int32(sd.nbl),
            np.int32(sd.nchan), np.int32(sd.ntime),
            **self.get_kernel_params(sd))       

    def post_execution(self, shared_data):
        pass
