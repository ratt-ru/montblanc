import numpy as np
import pycuda
import pycuda.curandom
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from node import *

import crimes

class RimeReduce(Node):
    def __init__(self):
        super(RimeReduce, self).__init__()
    def initialise(self, shared_data):
        pass
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        sd = shared_data

        crimes.segmented_reduce_complex128_sum(
            data=sd.jones_gpu, seg_starts=sd.keys_gpu,
            seg_sums=sd.vis_gpu, device_id=0)

    def post_execution(self, shared_data):
        pass

class RimeReduceFloat(Node):
    def __init__(self):
        super(RimeReduceFloat, self).__init__()
    def initialise(self, shared_data):
        pass
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        sd = shared_data

        crimes.segmented_reduce_complex64_sum(
            data=sd.jones_gpu, seg_starts=sd.keys_gpu,
            seg_sums=sd.vis_gpu, device_id=0)

    def post_execution(self, shared_data):
        pass        