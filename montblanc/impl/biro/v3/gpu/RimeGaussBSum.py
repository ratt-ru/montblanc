import numpy as np

import pycuda.driver
import pycuda.tools

import montblanc.impl.biro.v2.gpu.RimeGaussBSum

class RimeGaussBSum(montblanc.impl.biro.v2.gpu.RimeGaussBSum.RimeGaussBSum):
    def __init__(self, weight_vector=False):
        super(RimeGaussBSum, self).__init__(weight_vector)
    def initialise(self, solver, stream=None):
        super(RimeGaussBSum, self).initialise(solver,stream)
        self.dev_mem_pool = pycuda.tools.DeviceMemoryPool()
        self.dev_mem_pool.allocate(1024).free()
        self.start = pycuda.driver.Event(pycuda.driver.event_flags.DISABLE_TIMING)
        self.end = pycuda.driver.Event(pycuda.driver.event_flags.DISABLE_TIMING)

    def shutdown(self, solver, stream=None):
        super(RimeGaussBSum, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeGaussBSum, self).pre_execution(solver,stream)
    def post_execution(self, solver, stream=None):
        super(RimeGaussBSum, self).pre_execution(solver,stream)

    def execute(self, solver, stream=None):
        slvr = solver

        # The gaussian shape array can be empty if
        # no gaussian sources were specified.
        gauss = np.intp(0) if np.product(slvr.gauss_shape_shape) == 0 \
            else slvr.gauss_shape_gpu

        self.start.record(stream=stream)

        self.kernel(slvr.uvw_gpu, slvr.brightness_gpu, gauss, 
            slvr.wavelength_gpu, slvr.ant_pairs_gpu, slvr.jones_scalar_gpu,
            slvr.weight_vector_gpu, slvr.vis_gpu, slvr.bayes_data_gpu,
            slvr.chi_sqrd_result_gpu,
            stream=stream, **self.get_kernel_params(slvr))

        """
        # Compute the reduction over the chi squared terms and get the result
        # Since pycuda.gpuarray.sum uses the supplied arrays allocator
        # to allocate the result vector, we temporarily swap it out
        # for the memory pool allocator
        tmp_alloc = slvr.chi_sqrd_result_gpu.allocator
        slvr.chi_sqrd_result_gpu.allocator = self.dev_mem_pool.allocate
        pycuda.gpuarray.sum(slvr.chi_sqrd_result_gpu, stream=stream) \
            .get_async(ary=slvr.X2_cpu, stream=stream)
        slvr.chi_sqrd_result_gpu.allocator = tmp_alloc
        """

        self.end.record(stream=stream)