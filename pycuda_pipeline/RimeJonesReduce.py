import numpy as np
import pycuda
import pycuda.curandom
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from node import *

import segreduce

class RimeJonesReduce(Node):
    def __init__(self):
        super(RimeJonesReduce, self).__init__()
    def initialise(self, shared_data):
        pass
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        N = njones = 64
        jones = np.arange(N).astype(np.complex128);
        jones.imag = jones.real
        keys = np.array([0, 15, 31, 63],dtype=np.int32)

        jones_gpu = gpuarray.to_gpu(jones)
        keys_gpu = gpuarray.to_gpu(keys)
        sums_gpu = gpuarray.zeros(shape=keys.shape, dtype=jones_gpu.dtype.type)

        # http://blog.gmane.org/gmane.comp.python.cuda/month=20131101
        segreduce.segmented_reduce_complex128_sum(
            data=jones_gpu, seg_starts=keys_gpu, seg_sums=sums_gpu,
            device_id=0)

    def post_execution(self, shared_data):
        pass