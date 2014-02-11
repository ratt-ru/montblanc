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
        """
        nbl = shared_data.nbl
        ndir = shared_data.ndir

        jones_shape = shared_data.jones_shape

        njones = np.product(jones_shape)
#        keys = np.empty(jones_shape[0:2])
        keys = jones_shape[2]*np.arange(np.product(jones_shape[0:2])).reshape(jones_shape[0:2])

        print jones_shape

#        for i in range(jones_shape[0]):
#            for bl in range(jones_shape[1]):
#                for dir in jones_shape[2]:
#                keys[i,bl] = (bl*jones_shape[0] + i)*jones_shape[2]
        """

        N = njones = 64
        jones = np.arange(N).astype(np.complex128);
        jones.imag = jones.real
        #jones = np.empty([N],dtype=np.complex128)
        keys = np.array([0, 15, 31, 63],dtype=np.int32)


        """
        n_threads = 256
        n_blocks = (njones + n_threads - 1) / n_threads
        block, grid = (n_threads,1,1), (n_blocks,1,1)

        print 'jones=', jones
        print 'keys=', keys

        print 'njones=', njones, 'stream=', shared_data.stream[0].handle#, \
#            'gpudata=', int(shared_data.jones_gpu.gpudata)
        """

        jones_gpu = gpuarray.to_gpu_async(jones, stream=shared_data.stream[0])
        keys_gpu = gpuarray.to_gpu_async(keys, stream=shared_data.stream[0])
        sums_gpu = gpuarray.zeros(shape=keys.shape, dtype=jones_gpu.dtype.type)

        print 'shape=', sums_gpu.shape

        # http://blog.gmane.org/gmane.comp.python.cuda/month=20131101
        # At present, this creates a new context, which is not ideal
        segreduce.segmented_reduce_complex128_sum(
            data=jones_gpu, seg_starts=keys_gpu, seg_sums=sums_gpu,
            device_id=0, stream=shared_data.stream[0])

        del jones_gpu
        del keys_gpu

        sums = sums_gpu.get_async(stream=shared_data.stream[0])
        print 'sums', sums

        del sums_gpu
        """

        jones = np.arange(N).astype(np.complex64)
        jones.imag = jones.real
        keys = np.array([0, 15, 31, 63],dtype=np.int32)

        #print 'jones=', jones, 'keys=', keys
        print 'njones=', njones, 'stream=', shared_data.stream[0].handle

        jones_gpu = gpuarray.to_gpu_async(jones, stream=shared_data.stream[0])
        keys_gpu = gpuarray.to_gpu_async(keys, stream=shared_data.stream[0])
        sums_gpu = gpuarray.zeros(shape=keys.shape, dtype=jones.dtype.type)

        print 'shape=', sums_gpu.shape

        segreduce.segmented_reduce_complex64_sum(
            data=jones_gpu, seg_starts=keys_gpu, seg_sums=sums_gpu,
            device_id=0, stream=shared_data.stream[0])

        del jones_gpu
        del keys_gpu

        sums = sums_gpu.get_async(stream=shared_data.stream[0])
        print 'sums', sums

        del sums_gpu

        jones = np.arange(N).astype(np.float32);
        keys = np.array([0, 15, 31, 63],dtype=np.int32)

        print 'jones=', jones, 'keys=', keys
        print 'njones=', njones, 'stream=', shared_data.stream[0].handle

        jones_gpu = gpuarray.to_gpu_async(jones, stream=shared_data.stream[0])
        keys_gpu = gpuarray.to_gpu_async(keys, stream=shared_data.stream[0])
        sums_gpu = gpuarray.zeros(shape=keys.shape, dtype=jones.dtype.type)

        print 'jones.shape=',jones.shape, 'jones_gpu.shape=', jones_gpu.shape, 'sums_gpu.shape=', sums_gpu.shape, 'sums_gpu.shape=', sums_gpu.shape,

        segreduce.segmented_reduce_float32_sum(
            data=jones_gpu, seg_starts=keys_gpu, seg_sums=sums_gpu,
            device_id=0, stream=shared_data.stream[0])

        print 'jones_gpu.shape=', jones_gpu.shape, 'sums_gpu.shape=', sums_gpu.shape, 'sums_gpu.shape=', sums_gpu.shape,

        del jones_gpu
        del keys_gpu

        sums = sums_gpu.get_async(stream=shared_data.stream[0])
        print 'sums', sums

        del sums_gpu
        """

    def post_execution(self, shared_data):
        pass