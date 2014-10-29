import numpy as np

import pycuda.driver
import pycuda.tools

import montblanc.impl.biro.v2.gpu.RimeEK

class RimeEK(montblanc.impl.biro.v2.gpu.RimeEK.RimeEK):
    def __init__(self):
        super(RimeEK, self).__init__()
    def initialise(self, solver, stream=None):
        super(RimeEK, self).initialise(solver,stream)
        self.start = pycuda.driver.Event(pycuda.driver.event_flags.DISABLE_TIMING)
        self.end = pycuda.driver.Event(pycuda.driver.event_flags.DISABLE_TIMING)
    def shutdown(self, solver, stream=None):
        super(RimeEK, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeEK, self).pre_execution(solver,stream)
    def post_execution(self, solver, stream=None):
        super(RimeEK, self).pre_execution(solver,stream)

    def execute(self, solver, stream=None):
        slvr = solver

        self.start.record(stream=stream)

        self.kernel(slvr.uvw_gpu, slvr.lm_gpu, slvr.brightness_gpu,
            slvr.wavelength_gpu, slvr.point_errors_gpu, slvr.jones_scalar_gpu,
            slvr.ref_wave, slvr.beam_width, slvr.beam_clip,
            stream=stream, **self.get_kernel_params(slvr))

        self.end.record(stream=stream)
